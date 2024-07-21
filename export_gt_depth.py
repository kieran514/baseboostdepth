# python export_gt_depth.py --data_path /media/kieran/HDD/data/Apollo --split ApolloScape
# python export_gt_depth.py --data_path /media/kieran/HDD/data/DSEC --split DSEC
# python export_gt_depth.py --data_path /media/kieran/HDD/data/Oxford --split Oxford

from __future__ import absolute_import, division, print_function

import os

import argparse
import numpy as np
import PIL.Image as pil

from utils import readlines
from kitti_utils import generate_depth_map
import tqdm
import cv2 
import pdb 
# python export_gt_depth_tests.py --data_path /media/kieran/Extreme_SSD/data/KITTI_RAW --split eigen_benchmark
# python export_gt_depth_tests.py --data_path /media/kieran/Extreme_SSD/Zeus/SYNS/SYNS --split SYNS

def export_gt_depths_all():

    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the KITTI data',
                        required=True)
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        required=True,
                        choices=["eigen", "eigen_benchmark", "ApolloScape", "DSEC", "Oxford", 'eigen_zhou/depth_test', 'SYNS'])
    # parser.add_argument('--train_file',
    #                     type=str,
    #                     help='file',
    #                     required=True)

    opt = parser.parse_args()

    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    lines = readlines(os.path.join(split_folder, "test_files.txt"))
    # lines = readlines(os.path.join(split_folder, f"{opt.train_file}.txt"))

    print("Exporting ground truth depths for {}".format(opt.split))

    gt_depths = []
    gt_edges = []
    for line in tqdm.tqdm(lines):

        if opt.split == 'DSEC' or opt.split == 'Oxford':
            folder, frame_id, side, data = line.split()
        elif opt.split == 'SYNS':
            folder, frame_id = line.split()
        elif opt.split == 'eigen_benchmark':
            folder, frame_id, _= line.split()
        else:
            folder, frame_id, _, _ = line.split()

        if not opt.split == 'SYNS':
            frame_id = int(frame_id)

        if opt.split == "eigen_zhou/depth_test":
            calib_dir = os.path.join(opt.data_path, folder.split("/")[0])
            velo_filename = os.path.join(opt.data_path, folder,
                                         "velodyne_points/data", "{:010d}.bin".format(frame_id))
            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
        elif opt.split == "eigen_benchmark":
            gt_depth_path = os.path.join(opt.data_path, folder, "proj_depth",
                                         "groundtruth", "image_02", "{:010d}.png".format(frame_id))
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 256

        elif opt.split == 'ApolloScape':
            gt_depth_path = os.path.join(opt.data_path, "test/depth",
                                         "{:010d}.png".format(frame_id))
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32)  #could be 256
        elif opt.split == 'DSEC':
            gt_depth_path = os.path.join(opt.data_path, 'test/test_depth', folder,
                                         "{:06d}.png".format(frame_id))
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32)
        elif opt.split == 'Oxford':
            gt_depth_path = os.path.join(opt.data_path, 'test/depth',
                                         "{:010d}.png".format(frame_id))
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) /5.1

        elif opt.split == 'SYNS':
            gt_depth_path = os.path.join(opt.data_path, "depth", folder, "{}.npy".format(frame_id))
            gt_depth = np.load(gt_depth_path)

            depth = to_log(gt_depth)
            depth = cv2.GaussianBlur(depth, (3, 3), sigmaX=1, sigmaY=1)
            dx = cv2.Sobel(src=depth, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
            dy = cv2.Sobel(src=depth, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)

            edges = np.sqrt(dx**2 + dy**2)[..., None]
            edges = edges > np.nanmean(edges)

        gt_depths.append(gt_depth.astype(np.float32))

        if opt.split == 'SYNS':
            gt_edges.append(edges.astype(np.float32))

    output_path = os.path.join(split_folder, "gt_depths.npz")
    if opt.split == 'SYNS':
        output_path_edge = os.path.join(split_folder, "gt_edges.npz")

    print("Saving to {}".format(opt.split))

    np.savez_compressed(output_path, data=np.array(gt_depths))
    if opt.split == 'SYNS':
        np.savez_compressed(output_path_edge, data=np.array(gt_edges))

def to_log(depth):
    """Convert linear depth into log depth."""
    depth = (depth > 0) * np.log(depth.clip(min=1.1920928955078125e-07))
    return depth


if __name__ == "__main__":
    export_gt_depths_all()
