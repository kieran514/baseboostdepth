# python evaluate_depth_MD2.py --eval_mono --load_weights_folder /media/kieran/Extreme_SSD/Zeus/paper/Decomp/0.4/weights_19/ --kt_path /media/kieran/Extreme_SSD/data/KITTI_RAW --eval_split eigen
# python evaluate_depth_MD2.py --eval_mono --load_weights_folder /media/kieran/Extreme_SSD/Zeus/paper/Decomp/0.4/weights_19/ --kt_path /media/kieran/Extreme_SSD/data/KITTI_RAW --eval_split SYNS

from __future__ import absolute_import, division, print_function
from torchvision.utils import save_image

import os
import cv2
import numpy as np
import tqdm
import torch
from torch.utils.data import DataLoader
from scipy import ndimage

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import pdb
import torch.nn as nn
from chamfer_distance import ChamferDistance
cv2.setNumThreads(0) 
cham = ChamferDistance()
import time

# python evaluate_depth_MD2.py --eval_mono --load_weights_folder /media/kieran/Extreme_SSD/Zeus/paper/Decomp_0.4/ --kt_path /media/kieran/Extreme_SSD/data/KITTI_RAW --eval_split SYNS --chamfer

class BackprojectDepth(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.h, self.w = shape
        self.ones = nn.Parameter(torch.ones(1, 1, self.h*self.w), requires_grad=False)

        grid = torch.meshgrid(torch.arange(self.w), torch.arange(self.h))  # (h, w), (h, w)
        pix = torch.stack(grid).view(2, -1)[None]  # (1, 2, h*w) as (x, y)
        pix = torch.cat((pix, self.ones), dim=1)  # (1, 3, h*w)
        self.pix = nn.Parameter(pix, requires_grad=False)

    def forward(self, depth, K_inv):
        b = depth.shape[0]
        pts = K_inv[:, :3, :3] @ self.pix.repeat(b, 1, 1) # (b, 3, h*w) Cam rays.
        pts *= depth.flatten(-2)  # 3D points.
        pts = torch.cat((pts[0], self.ones.repeat(b, 1, 1)), dim=1)  # (b, 4, h*w) Add homogenous.
        return pts

splits_dir = os.path.join("splits")
STEREO_SCALE_FACTOR = 5.4

def to_log(depth):
    depth = (depth > 0) * np.log(depth.clip(min=1.1920928955078125e-07))
    return depth
def _metrics_pointcloud(pred, target, th):
    """Helper to compute F-Score and IoU with different correctness thresholds."""
    P = (pred < th).float().mean()  # Precision - How many predicted points are close enough to GT?
    R = (target < th).float().mean()  # Recall - How many GT points have a predicted point close enough?
    if (P < 1e-3) and (R < 1e-3): return P, P  # No points are correct.

    f = 2*P*R / (P + R)
    iou = P*R / (P + R - (P*R))
    return f, iou

def compute_errors(gt, pred, opt, gt_edge_org=None, pred_edge_org=None, inv_K=None, gt_org=None, pred_org=None, mask=None):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    if opt.eval_split == 'SYNS':
        err = np.abs(pred - gt)
        err = err.mean()

        pred_point = torch.as_tensor(pred_org, device='cuda:0')
        target_point = torch.as_tensor(gt_org, device='cuda:0')
        K_inv = torch.as_tensor(inv_K, device='cuda:0')[None]

        backproj = BackprojectDepth(pred_point.shape).to('cuda:0')
        pred_pts = backproj(pred_point[None, None], K_inv)[:, :3, mask.flatten()]
        target_pts = backproj(target_point[None, None], K_inv)[:, :3, mask.flatten()]

        ###################### CHAMFER DIST #################################
        if opt.chamfer:
            pred_nn, target_nn, _, _ = cham(pred_pts.permute(0, 2, 1), target_pts.permute(0, 2, 1))
            pred_nn, target_nn = pred_nn.sqrt(), target_nn.sqrt()
            f1, iou1 = _metrics_pointcloud(pred_nn, target_nn, th=0.1)

        ###################### CHAMFER DIST #################################

        mask = np.logical_and(mask, gt_edge_org[:,:,0])

        th_edges = 10
        D_target = ndimage.distance_transform_edt(1 - mask) 

        D_pred = ndimage.distance_transform_edt(1 - pred_edge_org[:,:,0]) 

        pred_edges = pred_edge_org[:,:,0] & (D_target < th_edges)  

        edge_Acc = D_target[pred_edges].mean() if pred_edges.sum() else th_edges

        edge_comp = D_pred[mask].mean() if pred_edges.sum() else th_edges

        if opt.chamfer:
            return abs_rel, err, sq_rel, rmse, rmse_log, edge_Acc, edge_comp, f1.item(), iou1.item()
        else:
            return abs_rel, err, sq_rel, rmse, rmse_log, edge_Acc, edge_comp
    else:
        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    if opt.eval_split == 'SYNS':
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 125
    else:
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80


    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))
    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    encoder_dict = torch.load(encoder_path, map_location='cuda:0')


    if opt.eval_split == 'SYNS':
        print("Using SYNS\n")
        dataset = datasets.SYNSRAWDataset(filenames, 0,
                                            encoder_dict['height'], encoder_dict['width'],
                                            [0], 4, is_train=False,
                                            valid_datatypes=['MS'], naive_mix = True)
    else:
        print("Using KITTI\n")
        dataset = datasets.MixedDataset(filenames, 0,
                                            encoder_dict['height'], encoder_dict['width'],
                                            [0], 4, kt_path=opt.kt_path , is_train=False,
                                            valid_datatypes=['MS'], kt=True, naive_mix = True)

    dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=4,
                        pin_memory=True, drop_last=False)
    
    if opt.ViT:
        import networksvit

        encoder_dict = torch.load(encoder_path, map_location='cuda:0')
        encoder = networksvit.mpvit_small() #networks.ResnetEncoder(opt.num_layers, False)
        encoder.num_ch_enc = [64,128,216,288,288]  # = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networksvit.DepthDecoder()

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path, map_location='cuda:0'), strict=False)
    elif opt.CA_depth:
        import networksCA
        encoder_dict = torch.load(encoder_path, map_location='cuda:0')
        encoder = networksCA.ResnetEncoder(50, False)
        depth_decoder = networksCA.DepthDecoder(encoder.num_ch_enc)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path, map_location='cuda:0'), strict=False)      
    elif opt.SQL:
        import networksSQL
        encoder_dict = torch.load(encoder_path, map_location='cuda:0')
        encoder = networksSQL.ResnetEncoderDecoder(num_layers=50, num_features=256, model_dim=32)
        depth_decoder = networksSQL.Lite_Depth_Decoder_QueryTr(in_channels=32, patch_size=16, dim_out=64, embedding_dim=32, 
                                                        query_nums=64, num_heads=4, min_val=0.001, max_val=80.0)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path, map_location='cuda:0'))    
    elif opt.SQL_L:
        import networksSQL
        encoder_dict = torch.load(encoder_path, map_location='cuda:0')
        encoder = networksSQL.ResnetEncoderDecoder(num_layers=50, num_features=256, model_dim=32)
        depth_decoder = networksSQL.Lite_Depth_Decoder_QueryTr(in_channels=32, patch_size=20, dim_out=128, embedding_dim=32, 
                                                        query_nums=128, num_heads=4, min_val=0.001, max_val=80.0)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path, map_location='cuda:0'))      
    elif opt.DIFFNet:
        import networksDIFF
        decoder_dict = torch.load(decoder_path, map_location = 'cuda:0')

        encoder = networksDIFF.test_hr_encoder.hrnet18(False)
        encoder.num_ch_enc = [ 64, 18, 36, 72, 144 ]
        depth_decoder = networksDIFF.HRDepthDecoder(encoder.num_ch_enc, opt.scales)
        model_dict = encoder.state_dict()
        dec_model_dict = depth_decoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in dec_model_dict})

    else:
        import networks
        encoder_dict = torch.load(encoder_path, map_location='cuda:0')

        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path, map_location='cuda:0'), strict=False)

    encoder.to("cuda:0")
    encoder.eval()
    depth_decoder.to("cuda:0")
    depth_decoder.eval()

    pred_disps = []

    print("-> Computing predictions with size {}x{}".format(
        encoder_dict['width'], encoder_dict['height']))

    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):
            input_color = data[("color", 0, 0)].to("cuda:0")

            output = depth_decoder(encoder(input_color))

            if opt.eval_split == 'SYNS':
                inv_K = data[("inv_K", -1)]


            if opt.SQL or opt.SQL_L:
                pred_disp = output[("disp", 0)]
            else:
                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)

            pred_disp = pred_disp.cpu()[:, 0]
            # pred_disp = pred_disp.cpu()[:, 0].numpy()

            pred_disps.append(pred_disp)

    pred_disps = np.concatenate(pred_disps)

    if opt.eval_split == 'SYNS':
        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

        gt_path_edge = os.path.join(splits_dir, opt.eval_split, "gt_edges.npz")
        gt_edges = np.load(gt_path_edge, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
    else:
        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in tqdm.tqdm(range(pred_disps.shape[0])):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        if opt.SQL or opt.SQL_L:
            pred_depth = pred_disp
        else:
            pred_depth = 1 / pred_disp
        pred_depth_org = pred_depth

        if opt.eval_split == 'SYNS':
            gt_edge = gt_edges[i]

            depth_e = to_log(pred_depth)
            depth_e = cv2.GaussianBlur(depth_e, (3, 3), sigmaX=1, sigmaY=1)
            dx = cv2.Sobel(src=depth_e, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
            dy = cv2.Sobel(src=depth_e, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
            edges = np.sqrt(dx**2 + dy**2)[..., None]
            pred_edge = edges > edges.mean()

        # mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < 10000)
        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        if opt.eval_split != 'SYNS':

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio
            if opt.eval_split == 'SYNS':
                pred_depth_org *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        pred_depth_org[pred_depth_org < MIN_DEPTH] = MIN_DEPTH
        pred_depth_org[pred_depth_org > MAX_DEPTH] = MAX_DEPTH

        if opt.eval_split == 'SYNS':
            errors.append(compute_errors(gt_depth, pred_depth, opt, gt_edge, pred_edge, inv_K, gt_depths[i], pred_depth_org, mask))
        else:
            errors.append(compute_errors(gt_depth, pred_depth, opt))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)
    if opt.eval_split != 'SYNS':
        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("\n-> Done!")
    else:
        if opt.chamfer:
            print("\n  " + ("{:>8} | " * 9).format('abs_rel', 'err', 'sq_rel', 'rmse','rmse_log', 'edge_Acc', 'edge_comp',  'f1', 'iou1'))
            print(("&{: 8.3f}  " * 9).format(*mean_errors.tolist()) + "\\\\")
            print("\n-> Done!")
        else:
            print("\n  " + ("{:>8} | " * 7).format('abs_rel', 'err', 'sq_rel', 'rmse','rmse_log', 'edge_Acc', 'edge_comp'))
            print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
            print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())