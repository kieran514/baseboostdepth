
# python validation.py --model_name MD2_row1 base_skip_row2 just_5_row3 rand_row4 rand_trimin_row5 rand_trimin_res_row6

from __future__ import absolute_import, division, print_function
from email.policy import strict

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import tqdm 
import torch
from torchvision import transforms, datasets
from pathlib import Path
import networks
from layers import disp_to_depth
import math
import cv2 
import pdb
from utils import readlines
from PIL import Image
import matplotlib.pyplot as plt

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
def colormap(inputs, color='plasma', normalize=True, torch_transpose=True):
    _DEPTH_COLORMAP = plt.get_cmap('plasma', 256)

    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        if color == 'plasma':
            vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        if color == 'plasma':
            vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        if color == 'plasma':
            vis = _DEPTH_COLORMAP(vis)

        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis

def parse_args():
    parser = argparse.ArgumentParser(
        description='Vis for Detailed Analysis.')
    
    parser.add_argument('--model_name', nargs='+', default=[])

    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
        
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity. (This only '
                             'makes sense for stereo-trained KITTI models).',
                        action='store_true')
    parser.add_argument("--ViT",
                        help=').',
                        action='store_true')
    parser.add_argument("--SQL",
                        help=').',
                        action='store_true')

    return parser.parse_args()

def compute_errors(gt, pred):
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

    return abs_rel

def test_simple(args, model, paths, num):

    print('Starting Depth Predictions')

    device = torch.device("cuda")
    model_path = os.path.join('/media/kieran/Extreme_SSD/Zeus/paper/ablation+vit', model)

    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    if args.ViT:
        import networksvit
        encoder_dict = torch.load(encoder_path, map_location='cuda:0')
        feed_height = encoder_dict['height']
        feed_width = encoder_dict['width']

        encoder = networksvit.mpvit_small() #networks.ResnetEncoder(opt.num_layers, False)
        encoder.num_ch_enc = [64,128,216,288,288]  # = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networksvit.DepthDecoder()

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(depth_decoder_path, map_location='cuda:0'), strict=False)
    elif args.SQL:
        import networksSQL
        encoder_dict = torch.load(encoder_path, map_location='cuda:0')
        feed_height = encoder_dict['height']
        feed_width = encoder_dict['width']

        encoder = networksSQL.ResnetEncoderDecoder(num_layers=50, num_features=256, model_dim=32)
        depth_decoder = networksSQL.Lite_Depth_Decoder_QueryTr(in_channels=32, patch_size=16, dim_out=64, embedding_dim=32, 
                                                        query_nums=64, num_heads=4, min_val=0.001, max_val=80.0)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(depth_decoder_path, map_location='cuda:0'))    

    else:
        encoder_dict = torch.load(encoder_path, map_location='cuda:0')
        encoder = networks.ResnetEncoder(18, False)
        feed_height = encoder_dict['height']
        feed_width = encoder_dict['width']

        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)


        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(depth_decoder_path, map_location='cuda:0'), strict=False)

    # encoder = networks.ResnetEncoder(18, False)
    # loaded_dict_enc = torch.load(encoder_path, map_location=device)
    # feed_height = loaded_dict_enc['height']
    # feed_width = loaded_dict_enc['width']
    # filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    # encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    # depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    # loaded_dict = torch.load(depth_decoder_path, map_location=device)
    # depth_decoder.load_state_dict(loaded_dict, strict=False)
    depth_decoder.to(device)
    depth_decoder.eval()

    output_directory = os.path.join('/media/kieran/Extreme_SSD/Zeus/validation_vis', model)
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    MIN_DEPTH = 0.1
    MAX_DEPTH = 80

    pred_disps = []
    abs_rel = []

    with torch.no_grad():
        for idx, image_path in tqdm.tqdm(enumerate(paths)):

            input_image_pil = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image_pil.size
            input_image = input_image_pil.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            if args.SQL:
                pred_disp = 1/ outputs[("disp", 0)]
            else:
                pred_disp, _ = disp_to_depth(outputs[("disp", 0)], MIN_DEPTH, MAX_DEPTH)

            pred_disp = pred_disp.cpu()[:, 0].numpy()

            pred_disps.append(pred_disp)
            if args.SQL:
                disp = 1 / outputs[("disp", 0)]
            else:
                disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=disp_resized_np.max())
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            # im = pil.fromarray(disp_resized_np*255).convert('L')
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}.jpg".format(str(idx).zfill(10)))
            im.save(name_dest_im)
            output_directory_new = os.path.join('/media/kieran/Extreme_SSD/Zeus/validation_vis', 'images')
            if num == 0:
                name_dest_inpur_image = os.path.join(output_directory_new, "{}.jpg".format(str(idx).zfill(10)))
                input_image_pil.save(name_dest_inpur_image)

    pred_disps = np.concatenate(pred_disps)

    gt_path = os.path.join('/media/kieran/Extreme_SSD/Zeus/validation_vis/depth_split', "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    ratios = []

    print('Starting Depth Eval')

    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]

        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                            0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        if num == 0:
            gt_depth_play = 1/gt_depth
            gt_depth_play[gt_depth_play > 80] = 0
            normalizer = mpl.colors.Normalize(vmin=gt_depth_play.min(), vmax=gt_depth_play.max())
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(gt_depth_play)[:, :, :3] * 255).astype(np.uint8)
            depther = pil.fromarray(colormapped_im)
            depther.save(f"/media/kieran/Extreme_SSD/Zeus/validation_vis/depth/{str(i).zfill(10)}.jpg")

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        # pred_depth *= opt.pred_depth_scale_factor
        ratio = np.median(gt_depth) / np.median(pred_depth)
        ratios.append(ratio)
        pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        abs_rel.append(compute_errors(gt_depth, pred_depth))
    print("   Finished Depth Eval")
    print(np.array(ratios).mean())

    return output_directory, abs_rel

if __name__ == '__main__':
    args = parse_args()
    paths = []

    absrel = {}

    inputs_vid = 'ffmpeg -y -hide_banner -loglevel error -i /media/kieran/Extreme_SSD/Zeus/validation_vis/images/imgs.mp4 -i /media/kieran/Extreme_SSD/Zeus/validation_vis/depth/depth.mp4'
    text = f"[0:v]scale=w=621:h=-1,drawtext=text='Images':x=10:y=10:fontsize=24:fontcolor=white:fontfile=/media/kieran/Extreme_SSD/Zeus/validation_vis/arial.ttf[Imgs];[1:v]scale=w=621:h=-1,drawtext=text='Depth':x=10:y=10:fontsize=24:fontcolor=white:fontfile=/media/kieran/Extreme_SSD/Zeus/validation_vis/arial.ttf[Depth];"
    hstack = '[Imgs][Depth]hstack=inputs=2[row0];'
    vstack = '[row0]'
    size = len(args.model_name)

    if size == 1:
        fin = f'" -c:v libx264 -crf 18 -preset veryfast {args.model_name[0]}.mp4'
    else:
        fin = f'" -c:v libx264 -crf 18 -preset veryfast {size}.mp4'
    h_stack_holder = ''

    filenames = readlines(os.path.join('/media/kieran/Extreme_SSD/Zeus/validation_vis/depth_split', "val_files.txt"))
    for file in filenames:
        file, num, side, kt, = file.split(' ')
        paths.append(os.path.join('/media/kieran/Extreme_SSD/data/KITTI_RAW', file, 'image_02/data', f'{num}.jpg'))
    
    vertical = math.ceil(size/2) + 1

    for num, model in tqdm.tqdm(enumerate(args.model_name)):
        output_directory, absrel = test_simple(args, model, paths, num)

        # for col, path in enumerate(paths):
        #     img_name = path.split('/')[-1]
        #     pather = f'/media/kieran/Extreme_SSD/Zeus/validation_vis/{model}/{str(col).zfill(10)}.jpg'
        #     code = f'''ffmpeg -i {pather} -y -hide_banner -loglevel error -vf "drawtext=text='{model.upper()} {absrel[col]}':x=10:y=10:fontsize=24:fontcolor=white:fontfile=/media/kieran/Extreme_SSD/Zeus/validation_vis/arial.ttf" {pather}'''
        #     os.system(code)

        code = f'''cd {output_directory} ; ffmpeg -y -hide_banner -loglevel error -framerate 0.5 -pattern_type glob -i '*.jpg' -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p {model}.mp4'''
        os.system(code)

        inputs_vid += f' -i {output_directory}/{model}.mp4'
        if num == (size-1) and num % 2 == 0:
            pass
        else:
            text += f"[{num+2}:v]scale=w=621:h=-1[{model}];"

        h_stack_holder += f'[{model}]'

        if num % 2 != 0 and num > 0:
            hstack += h_stack_holder + f'hstack=inputs=2[row{int(1+math.ceil(num/2))}];'
            h_stack_holder = ''
            vstack += f'[row{int(1+math.ceil(num/2))}]'
        elif num == (size-1):
            vstack += f'[{size}:v]'

    inputs_vid += ' -filter_complex "'

    vstack += f'vstack=inputs={vertical}'
    final = inputs_vid + text + hstack + vstack + fin
    print(final)
    os.system('cd /media/kieran/Extreme_SSD/Zeus/validation_vis ;' + final)
