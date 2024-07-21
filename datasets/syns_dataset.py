# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import torch
import cv2
from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class SYNSDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(SYNSDataset, self).__init__(*args, **kwargs)

    def load_intrinsic_syns(self):
        '''KITTI intricis loader'''
        KITTI_FOV = (25.46, 84.10)
        KITTI_SHAPE = (376, 1242)

        Fy, Fx = KITTI_FOV
        h, w = KITTI_SHAPE

        cx, cy = w//2, h//2
        fx = cx / np.tan(np.deg2rad(Fx)/2)
        fy = cy / np.tan(np.deg2rad(Fy)/2)

        K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]], dtype=np.float32)
        
        inv_K = np.linalg.pinv(K)

        return K, inv_K
    
    def get_color(self, data_path, folder, frame_index):
        color = self.loader(self.get_image_path(data_path, folder, frame_index))
        return color
    
    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits"""
        folder, frame_index = self.filenames[index].split()
        return folder, frame_index

class SYNSRAWDataset(SYNSDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(SYNSRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, data_path, folder, frame_index):
        image_path = os.path.join(
            data_path, 'images', folder, f'{frame_index}.png')
        return image_path
    