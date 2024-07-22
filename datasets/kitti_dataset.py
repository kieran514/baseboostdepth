from __future__ import absolute_import, division, print_function
import pdb
import os
import numpy as np
import PIL.Image as pil
from .mono_dataset import MonoDataset

class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

    def load_intrinsic_kt(self, scale):
        '''KITTI intricis loader'''
        K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        K[0, :] *= self.width // (2 ** scale)
        K[1, :] *= self.height // (2 ** scale)
        inv_K = np.linalg.pinv(K)
        return K, inv_K
    
    def index_to_folder_and_frame_idx_kt(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits"""
        if self.naive_mix:
            line = self.filenames[index].split()
        else:
            line = self.filenames['MS'][index].split()
        folder = line[0]
        if len(line) >= 3:
            frame_index = int(line[1])
        else:
            frame_index = 0
        if len(line) >= 3:
            side = line[2]
        else:
            side = None
        return folder, frame_index, side
    

class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path_kt(self, data_path, frame_index, side, folder):
        side_map = {"l": 2, "r": 3}
        f_str = "{:010d}{}".format(frame_index, '.jpg')
        image_path = os.path.join(
            data_path, folder, "image_0{}/data".format(side_map[side]), f_str)
        return image_path

    def get_color_kt(self, path, do_flip):
        color = self.loader(path)
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color
    
class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path_odom(self, data_path, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            os.path.dirname(data_path), 'odom/'
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format({"r":'3', "l": "2"}[side]), 'data', f_str)
        return image_path

    def index_to_folder_and_frame_idx(self, index):
        line = self.filenames[index].split()
        folder = line[0]
        if len(line) >= 3:
            frame_index = int(line[1])
        else:
            frame_index = 0
        if len(line) >= 3:
            side = line[2]
        else:
            side = None
        return folder, frame_index, side
    
    def get_color(self, data_path, folder, frame_index, side, do_flip, i=0):
        color = self.loader(self.get_image_path_odom(data_path, folder, int(frame_index) + i, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color
