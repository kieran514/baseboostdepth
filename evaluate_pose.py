from __future__ import absolute_import, division, print_function

import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import transformation_from_parameters
from utils import readlines
from options import MonodepthOptions
from datasets import KITTIOdomDataset
import networks
import tqdm
import pdb

# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):

    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse


def evaluate(opt):
    """Evaluate odometry on the KITTI dataset
    """
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    assert opt.eval_split == "odom_9" or opt.eval_split == "odom_10", \
        "eval_split should be either odom_9 or odom_10"

    sequence_id = int(opt.eval_split.split("_")[1])

    filenames = readlines(
        os.path.join(os.path.dirname(__file__), "splits", "odom",
                     "test_files_{:02d}.txt".format(sequence_id)))

    dataset = KITTIOdomDataset(filenames, 0, 192, 640, kt_path=opt.kt_path , is_train=False,
                                            kt=True, naive_mix = True)
    
    dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=8,
                        pin_memory=True, drop_last=False)

    pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
    pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")

    pose_encoder = networks.ResnetEncoder(opt.num_layers, False, 2)
    pose_encoder.load_state_dict(torch.load(pose_encoder_path, map_location='cuda:0'))

    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
    pose_decoder.load_state_dict(torch.load(pose_decoder_path, map_location='cuda:0'))

    pose_encoder.cuda()
    pose_encoder.eval()
    pose_decoder.cuda()
    pose_decoder.eval()

    pred_poses = []
    pred_poses_multi = []
    print("-> Computing pose predictions")

    skip_frame = 2

    opt.frame_ids = [0, skip_frame]  # pose network only takes two frames as input

    with torch.no_grad():
        for inputs in tqdm.tqdm(dataloader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            try:
                all_color_aug = torch.cat([inputs[("color", i, 0)] for i in opt.frame_ids], 1)

                features = [pose_encoder(all_color_aug)]
                axisangle, translation = pose_decoder(features)

                pred_poses.append(
                    transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())

                pred_poses_multi_step = []

                for number in range(skip_frame):
                    all_color_aug = torch.cat([inputs[("color", number, 0)], inputs[("color", number +1, 0)]], 1)

                    features = [pose_encoder(all_color_aug)]
                    axisangle, translation = pose_decoder(features)
                    pred_poses_multi_step.append(
                        transformation_from_parameters(axisangle[:, 0], translation[:, 0]))
                                    
                T_rel_cumulative = torch.eye(4)

                for pose_step in pred_poses_multi_step[::-1]:
                    T_rel_cumulative = torch.matmul(T_rel_cumulative.cuda(), pose_step.cuda())

                pred_poses_multi.append(T_rel_cumulative.cpu().numpy())
            except:
                print('passed')
                pass

    pred_poses = np.concatenate(pred_poses)
    pred_poses_multi = np.concatenate(pred_poses_multi)

    gt_poses_path = os.path.join('/media/kieran/Extreme_SSD/data/odom', "poses", "{:02d}.txt".format(sequence_id))
    gt_global_poses = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
    gt_global_poses = np.concatenate(
        (gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
    gt_global_poses[:, 3, 3] = 1

    gt_local_poses = []
    for i in range(skip_frame, len(gt_global_poses)):
        if skip_frame > 1:

            # list_poses = [np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1 - k]), gt_global_poses[i-k])) for k in range(skip_frame)]
            # T_rel_cumulative = np.eye(4)

            # for pose_step in list_poses:
            #     T_rel_cumulative = np.dot(T_rel_cumulative, pose_step)
            # gt_local_poses.append(T_rel_cumulative)

            gt_local_poses.append(
                np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - skip_frame]), gt_global_poses[i])))

        else:
            gt_local_poses.append(
                np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])))


    ates = []
    ates_2 = []
    # num_frames = gt_xyzs.shape[0]
    num_frames = pred_poses.shape[0]
    track_length = 1
    for i in tqdm.tqdm(range(0, num_frames - skip_frame)):
        local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length]))
        local_xyzs_2 = np.array(dump_xyz(pred_poses_multi[i:i + track_length]))
        gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length]))
        ates.append(compute_ate(gt_local_xyzs, local_xyzs))
        ates_2.append(compute_ate(gt_local_xyzs, local_xyzs_2))

    print("\n   Trajectory error: {:0.3f}, std: {:0.3f}\n".format(np.mean(ates), np.std(ates)))
    print("\n   Trajectory error: {:0.3f}, std: {:0.3f}\n".format(np.mean(ates_2), np.std(ates_2)))

    save_path = os.path.join(opt.load_weights_folder, "poses.npy")
    # np.save(save_path, pred_poses)
    # print("-> Predictions saved to", save_path)


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
