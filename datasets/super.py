import os
import json
import numpy as np
import PIL.Image as pil
import torch
from glob import glob
import skimage.transform
from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset
# os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
# os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
# os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import yaml

import cv2

class MixedDataset(MonoDataset):
    """Superclass for different dataset loaders including:
    KITTI D
    CityScape 
    ApolloScape 
    Oxford Robotcar
    Audi (rectified and 1.29m baseline )
    Ford AV (rectified and )
    """
    def __init__(self, *args, **kwargs):
        super(MixedDataset, self).__init__(*args, **kwargs)

########################### Intrisics ############################### 

    def load_intrinsic_kt(self, folder, scale, side, res=False):
        '''KITTI intricis loader'''
        K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        if res:
            K[0, :] *= 1224
            K[1, :] *= 370

            inv_K = np.linalg.pinv(K)

            return K, inv_K
        else:
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            return K, inv_K
    
    def load_intrinsic_gb(self, folder, scale, side):
        '''KITTI intricis loader'''
        K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        
        K[0, :] *= self.width // (2 ** scale)
        K[1, :] *= self.height // (2 ** scale)

        inv_K = np.linalg.pinv(K)

        return K, inv_K
    def load_intrinsic_mal(self, folder, scale, side):
        '''Malaga intricis loader'''
        K = np.array([[0.77648035156, 0, 0.50500950195, 0],
                           [0, 1.03530713542, 0.34831734449, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        
        K[0, :] *= self.width // (2 ** scale)
        K[1, :] *= self.height // (2 ** scale)

        inv_K = np.linalg.pinv(K)

        return K, inv_K
    
    def load_intrinsic_fov(self, folder, scale, side, view):
        '''Malaga intricis loader'''

        if view == 36:
            K = np.array([[0.972222, 0, 0.5, 0],
                        [0, 1.4583, 0.5, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
            
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            return K, inv_K
        
        elif view == 50:
            K = np.array([[0.7, 0, 0.5, 0],
                        [0, 1.4583, 0.5, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
            
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            return K, inv_K
        
        elif view == 90:
            K = np.array([[0.3888888, 0, 0.5, 0],
                        [0, 1.4583, 0.5, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
            
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            return K, inv_K
        
        elif view == 150:
            K = np.array([[0.233333, 0, 0.5, 0],
                        [0, 1.4583, 0.5, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
            
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            return K, inv_K

    def load_intrinsic_cs(self, city_folder, scale, side):
        '''CityScape intricis loader'''
        camera_file = os.path.join(self.cs_path, city_folder, 'camera.json')

        with open(camera_file, 'r') as f:
            camera = json.load(f)
        fx = camera['intrinsic']['fx']
        fy = camera['intrinsic']['fy']
        u0 = camera['intrinsic']['u0']
        v0 = camera['intrinsic']['v0']
        K = np.array([[fx, 0, u0, 0],
                        [0, fy, v0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
        
        K[0, :] *= self.width // (2 ** scale)
        K[1, :] *= self.height // (2 ** scale)

        inv_K = np.linalg.pinv(K)

        return K, inv_K
    
    def load_intrinsic_ds(self, folder, scale, side):
        '''CityScape intricis loader'''
        camera_file = os.path.join(self.ds_path, 'calib', folder, 'calibration', 'cam_to_cam.yaml')

        with open(camera_file, 'r') as file:
            prime_service = yaml.safe_load(file)
        fx = (prime_service['intrinsics']['camRect1']['camera_matrix'][0] / 1440)
        fy = (prime_service['intrinsics']['camRect1']['camera_matrix'][1] / 1080)
        cx = (prime_service['intrinsics']['camRect1']['camera_matrix'][2] / 1440)
        cy = ((prime_service['intrinsics']['camRect1']['camera_matrix'][3] /1.25) -200) / 500
        K = np.array([[fx, 0, cx, 0],
                        [0, fy, cy, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
        
        K[0, :] *= self.width // (2 ** scale)
        K[1, :] *= self.height // (2 ** scale)

        inv_K = np.linalg.pinv(K)

        return K, inv_K
    
    def load_intrinsic_sim(self, folder, scale, side):
        '''Simulation intricis loader'''
        K = np.array([[0.972222, 0, 0.5, 0],
                        [0, 1.4583, 0.5, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
        
        K[0, :] *= self.width // (2 ** scale)
        K[1, :] *= self.height // (2 ** scale)

        inv_K = np.linalg.pinv(K)

        return K, inv_K
    
    def load_intrinsic_ddad(self, folder, scale, side):
        '''ddad intricis loader'''
        camera_file = os.path.join(self.ddad_path, folder, 'calib/camera.json')

        with open(camera_file, 'r') as f:
            camera = json.load(f)
        fx = camera['fx']
        fy = camera['fy']
        u0 = camera['cx']
        v0 = camera['cy']
        K = np.array([[fx, 0, u0, 0],
                        [0, fy, v0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
        
        K[0, :] *= self.width // (2 ** scale)
        K[1, :] *= self.height // (2 ** scale)

        inv_K = np.linalg.pinv(K)

        return K, inv_K
    
    def load_intrinsic_ms(self, folder, scale, side):
        '''ddad intricis loader'''
        camera_file = os.path.join(self.ms_path, folder, 'camera.json')

        with open(camera_file, 'r') as f:
            camera = json.load(f)
        fx = camera['fx']
        fy = camera['fy']
        u0 = camera['u0']
        v0 = camera['v0']
        K = np.array([[fx, 0, u0, 0],
                        [0, fy, v0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
        
        K[0, :] *= self.width // (2 ** scale)
        K[1, :] *= self.height // (2 ** scale)

        inv_K = np.linalg.pinv(K)

        return K, inv_K


    def load_intrinsic_ox(self, folder, scale, side):
        '''Oxford intricis loader'''
        K = np.array([[0.76800312968, 0, 0.50284919765, 0],
                        [0, 1.02400417292, 0.48896499666, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
        
        K[0, :] *= self.width // (2 ** scale)
        K[1, :] *= self.height // (2 ** scale)

        inv_K = np.linalg.pinv(K)

        return K, inv_K
        
    def load_intrinsic_fd(self, folder, scale, side):
        '''ford intricis loader'''
        if side == "left":
            K = np.array([[0.57088853019, 0, 0.51660798611, 0],
                        [0, 1.0996, 0.53327901501, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
            
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            return K, inv_K
        else:
            K = np.array([[0.57088853019, 0, 0.52757165579, 0],
                        [0, 1.0996, 0.53327901501, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
            
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            return K, inv_K
        
    def load_intrinsic_aps(self, folder, scale, side):
        '''Apolloscape has intriinsc paramters but we found these to be unreeliable for training and therefore not used for intrinsic supervision'''
        if side == "left":
            K = np.array([[0.68101, 0, 0.49829, 0],
                        [0, 0.85087, 0.25415, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
            
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            return K, inv_K
        else:
            K = np.array([[0.67978, 0, 0.506269, 0],
                        [0, 0.84919, 0.24409, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
            
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            return K, inv_K    
    def load_intrinsic_hol(self, folder, scale, side):
        '''HoloPix has intriinsc paramters but we found these to be unreeliable for training and therefore not used for intrinsic supervision'''
        K = np.array([[0,0,0,0],
                    [0,0,0,0],
                    [0,0,0,0],
                    [0,0,0,0]], dtype=np.float32)
        
        inv_K = np.array([[0,0,0,0],
                    [0,0,0,0],
                    [0,0,0,0],
                    [0,0,0,0]], dtype=np.float32)

        return K, inv_K
    
    def load_intrinsic_au(self, folder, scale, side):
        '''aurrigo intricis loader'''
        if side == "left": #K=Left and inv-K = Right
            K = np.array([[0.4167734375, 0, 0.4991875, 0],
                        [0, 0.86084677419, 0.42785483871, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
            
            inv_K = np.linalg.pinv(np.array([[0.41683984375 * self.width // (2 ** scale), 0, 0.5044296875 * self.width // (2 ** scale), 0],
                        [0, 0.86085483871 * self.height // (2 ** scale), 0.46231209677 * self.height // (2 ** scale), 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32))
        else:
            K = np.array([[0.41683984375, 0, 0.5044296875, 0],
                        [0, 0.86085483871, 0.46231209677, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
            
            inv_K = np.linalg.pinv(np.array([[0.4167734375 * self.width // (2 ** scale), 0, 0.4991875 * self.width // (2 ** scale), 0],
                        [0, 0.86084677419 * self.height // (2 ** scale), 0.42785483871 * self.height // (2 ** scale), 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32))
            
        K[0, :] *= self.width // (2 ** scale)
        K[1, :] *= self.height // (2 ** scale)

        return K, inv_K
########################### Baseline #############################

    def get_baseline_kt(self, folder):
        return 0.54
    
    def get_baseline_gb(self, folder):
        return 0.54
    
    def get_baseline_ds(self, folder):
        return 0.51    
    
    def get_baseline_ddad(self, folder):
        return 0    
    
    def get_baseline_sim(self, folder):
        return 0.5

    def get_baseline_cs(self, folder):
        camera_file = os.path.join(self.cs_path, folder, 'camera.json')

        with open(camera_file, 'r') as f:
            camera = json.load(f)
        return camera['extrinsic']['baseline']

    def get_baseline_aps(self, folder):
        return 0.622
    
    def get_baseline_ox(self, folder):
        return 0.24
    
    def get_baseline_ad(self, folder):
        return 1.29
    
    def get_baseline_fd(self, folder):
        return 0.577
    
    def get_baseline_au(self, folder):
        return 0.12
    
    def get_baseline_mal(self, folder):
        return 0.12
    
    def get_baseline_hol(self, folder):
        return 0
    
    def get_baseline_ms(self, folder):
        return 0
    
    def get_baseline_fov(self, folder):
        return 0.5
########################### frames ###############################
    '''Some datasets only prvide stereo data to train with and otehr just monocular. As to utilise the full scope of datasets avalable we make our Mixed optimsied version able to train with all datasets (Monocular, Stereo and Monocualr+Stereo) "9 = s" '''

    def get_frame_kt(self):
        # return [0, 's']
        # return [0, 1, 2, 3, 4, 5]
        # return [0, -1, 1]
        # return [0, 's']
        return [0, -1, 1]
        # return [0, 1, 2]
    
    def get_frame_gb(self):
        return [0, -1, 1]

    def get_frame_ds(self):
        return [0, 's']    
    
    def get_frame_ddad(self):
        return [0, -1, 1]
    
    def get_frame_sim(self):
        return [0, -1, 1, 's']

    def get_frame_cs(self):
        return [0, -1, 1, 's']
    
    def get_frame_ox(self):
        return [0, -1, 1, 's']
    
    def get_frame_ad(self):
        return [0, -1, 1, 's']    
    
    def get_frame_aps(self):
        return [0, -1, 1]
    
    def get_frame_fd(self):
        return [0, -1, 1]
    
    def get_frame_au(self):
        return [0, -1, 1, 's']

    def get_frame_mal(self):
        return [0, -1, 1, 's']
    
    def get_frame_hol(self):
        return [0, 's']
    
    def get_frame_ms(self):
        return [0, -1, 1, 's']
        # return [0, 's']
    
    def get_frame_fov(self):
        return [0, -1, 1, 's']
########################### Folders ###############################

    def index_to_folder_and_frame_idx_kt(self, index, data, is_test=False):
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

        skipper = line[-1]

        if self.skipping:
            return folder, frame_index, side, int(skipper)
        else:
            return folder, frame_index, side

    def index_to_folder_and_frame_idx_cs(self, index, data, is_test=False):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        txt file is of format:
            aache/000000 frame_idx side cs"""
        if self.naive_mix: 
            city_folder, frame_index, side, _ = self.filenames[index].split()
        else:
            city_folder, frame_index, side, _ = self.filenames['MS'][index].split()
        return city_folder, int(frame_index), side
    
    def index_to_folder_and_frame_idx_aps(self, index, data, is_test=False):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        txt file is of format:
            road01_ins/ColorImage/Record056 96 6 aps"""
        if is_test:
            test, frame_index, data = self.filenames[index].split()
            return test, int(frame_index), data
        else:
            if self.naive_mix: 
                folder, frame_index, side, _ = self.filenames[index].split()
            else:
                folder, frame_index, side, _ = self.filenames['M'][index].split()
            return folder, int(frame_index), side
    
    def index_to_folder_and_frame_idx_ox(self, index, data, is_test=False):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        txt file is of format:
            2014-05-19/13-05-38 96 left ox"""
        if is_test:
            test, frame_index, data, _ = self.filenames[index].split()
            return test, int(frame_index), data
        else:
            if self.naive_mix: 
                folder, frame_index, side, _ = self.filenames[index].split()
            else:
                folder, frame_index, side, _ = self.filenames['MS'][index].split()
            return folder, int(frame_index), side
    
    def index_to_folder_and_frame_idx_fd(self, index, data, is_test=False):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        txt file is of format:
            Log1 96 left fd"""
        if self.naive_mix: 
            folder, frame_index, side, _ = self.filenames[index].split()
        else:
            folder, frame_index, side, _ = self.filenames['M'][index].split()
        return folder, int(frame_index), side
    
    def index_to_folder_and_frame_idx_au(self, index, data, is_test=False):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        txt file is of format:
            2022-08-02-09-31-40 1136 left au"""
        if self.naive_mix: 
            folder, frame_index, side, _ = self.filenames[index].split()
        else:
            folder, frame_index, side, _ = self.filenames['MS'][index].split()
        return folder, int(frame_index), side
    
    def index_to_folder_and_frame_idx_ds(self, index, data, is_test=False):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        txt file is of format:
            """
        if self.naive_mix: 
            folder, frame_index, side, _ = self.filenames[index].split()
        else:
            folder, frame_index, side, _ = self.filenames['S'][index].split()
        return folder, int(frame_index), side
    
    def index_to_folder_and_frame_idx_sim(self, index, data, is_test=False):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        txt file is of format:
            """
        if self.naive_mix: 
            folder, frame_index, side, _ = self.filenames[index].split()
        else:
            folder, frame_index, side, _ = self.filenames['MS'][index].split()
        return folder, int(frame_index), side
    
    def index_to_folder_and_frame_idx_fov(self, index, data, is_test=False):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        txt file is of format:
            """
        if self.naive_mix: 
            folder, frame_index, side, _ = self.filenames[index].split()
        else:
            folder, frame_index, side, _ = self.filenames['MS'][index].split()
        return folder, int(frame_index), side
    
    def index_to_folder_and_frame_idx_ddad(self, index, data, is_test=False):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        txt file is of format:
            """
        if self.naive_mix: 
            folder, frame_index, side, _ = self.filenames[index].split()
        else:
            folder, frame_index, side, _ = self.filenames['M'][index].split()
        return folder, int(frame_index), side
    
    def index_to_folder_and_frame_idx_mal(self, index, data, is_test=False):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        txt file is of format:
            """
        if self.naive_mix: 
            folder, frame_index, side, _ = self.filenames[index].split()
        else:
            folder, frame_index, side, _ = self.filenames['MS'][index].split()
        return folder, int(frame_index), side
    
    def index_to_folder_and_frame_idx_hol(self, index, data, is_test=False):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        txt file is of format:
            """
        if self.naive_mix: 
            folder, frame_index, side, _ = self.filenames[index].split()
        else:
            folder, frame_index, side, _ = self.filenames['S'][index].split()
        return folder, int(frame_index), side
    
    def index_to_folder_and_frame_idx_gb(self, index, data, is_test=False):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        txt file is of format:
            """
        if self.naive_mix: 
            folder, frame_index, side, _ = self.filenames[index].split()
        else:
            folder, frame_index, side, _ = self.filenames['M'][index].split()
        return folder, int(frame_index), side

    def index_to_folder_and_frame_idx_ms(self, index, data, is_test=False):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        txt file is of format:
            """
        if self.naive_mix: 
            folder, frame_index, side, _ = self.filenames[index].split()
        else:
            folder, frame_index, side, _ = self.filenames['S'][index].split()
        return folder, int(frame_index), side
########################### Get Image Path ###############################

    def get_image_path_kt(self, data_path, folder, frame_index, side, is_test=False):
        side_map = {"l": 2, "r": 3}
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            data_path, folder, "image_0{}/data".format(side_map[side]), f_str)
        return image_path
    
    def get_image_path_cs(self, data_path, city_folder, frame_index, side, is_test=False):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            data_path, city_folder, side, f_str)
        return image_path
    
    def get_image_path_aps(self, data_path, folder, frame_index, side, is_test=False):
        if is_test:
            image_path = os.path.join(data_path, "test/img", "{:010d}.jpg".format(frame_index))
            return image_path
        else:
            f_str = "{:010d}{}".format(frame_index, self.img_ext)
            image_path = os.path.join(
                data_path, folder, side, f_str)
            return image_path
    
    def get_image_path_ox(self, data_path, folder, frame_index, side, is_test=False):
        if is_test:
            image_path = os.path.join(data_path, folder,
                                         "{:010d}.png".format(frame_index))
            return image_path
        else:
            f_str = "{:010d}{}".format(frame_index, self.img_ext)
            image_path = os.path.join(
                data_path, folder, side, f_str)
            return image_path
    
    def get_image_path_ad(self, data_path, folder, frame_index, side, is_test=False):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            data_path, folder, side, f_str)
        return image_path
    
    def get_image_path_fd(self, data_path, folder, frame_index, side, is_test=False):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            data_path, folder, side, f_str)
        return image_path
    
    def get_image_path_au(self, data_path, folder, frame_index, side, is_test=False):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            data_path, folder, side, f_str)
        return image_path
    
    def get_image_path_ds(self, data_path, folder, frame_index, side, is_test=False):
        if is_test:
            image_path = os.path.join(data_path, folder,side, "{:010d}.jpg".format(frame_index))
            return image_path
        else:
            f_str = "{:010d}{}".format(frame_index, self.img_ext)
            image_path = os.path.join(
                data_path, folder, side, f_str)
            return image_path

    def get_image_path_sim(self, data_path, folder, frame_index, side, is_test=False):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            data_path, folder, side, f_str)
        return image_path
    
    def get_image_path_fov(self, data_path, folder, frame_index, side, view, is_test=False):
        f_str = "{}{}".format(str(frame_index).zfill(3) + '-' + side, self.img_ext)
        image_path = os.path.join(
            data_path, str(view), str(view) + folder, side, f_str)
        return image_path
    
    def get_image_path_ddad(self, data_path, folder, frame_index, side, is_test=False):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            data_path, folder, side, f_str)
        return image_path
    
    def get_image_path_mal(self, data_path, folder, frame_index, side, is_test=False):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            data_path, folder, side, f_str)
        return image_path
    
    def get_image_path_hol(self, data_path, folder, frame_index, side, is_test=False):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            data_path, folder, side, f_str)
        return image_path
    
    def get_image_path_ms(self, data_path, folder, frame_index, side, is_test=False):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            data_path, folder, side, f_str)
        return image_path
    
    def get_image_path_gb(self, data_path, folder, frame_index, side, is_test=False):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            data_path, folder, side, f_str)
        return image_path

########################### Get Image ###############################

    def get_color_kt(self, data_path, folder, frame_index, side, do_flip, i=0, is_test=False):
        # try:
        #     color = self.loader(self.get_image_path_kt(data_path, folder, int(frame_index) + 40*i, side, is_test))
        # except:
        color = self.loader(self.get_image_path_kt(data_path, folder, int(frame_index) + i, side, is_test))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color
    
    def get_color_gb(self, data_path, folder, frame_index, side, do_flip, i=0, is_test=False):
        color = self.loader(self.get_image_path_gb(data_path, folder, int(frame_index) + i, side, is_test))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def get_color_cs(self, data_path, city_folder, frame_index, side, do_flip, i=0, is_test=False):
        if i == -1:
            frame_index = self.get_offset_framename(frame_index, offset=-2)
            color = self.loader(self.get_image_path_cs(data_path, city_folder, frame_index, side))
        elif i == 1:
            frame_index = self.get_offset_framename(frame_index, offset=2)
            color = self.loader(self.get_image_path_cs(data_path, city_folder, frame_index, side))
        else:
            color = self.loader(self.get_image_path_cs(data_path, city_folder, frame_index, side, is_test))

        # if do_flip:
        #     color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def get_color_aps(self, data_path, folder, frame_index, side, do_flip, i=0, is_test=False):
        if i == -1:
            frame_index = self.get_offset_framename(frame_index, offset=-3)
            color = self.loader(self.get_image_path_aps(data_path, folder, frame_index, side))
        elif i == 1:
            frame_index = self.get_offset_framename(frame_index, offset=3)
            color = self.loader(self.get_image_path_aps(data_path, folder, frame_index, side))
        else:
            color = self.loader(self.get_image_path_aps(data_path, folder, frame_index, side, is_test))

        # if do_flip:
        #     color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color
    
    def get_color_ox(self, data_path, folder, frame_index, side, do_flip, i=0, is_test=False):
        if i == -1:
            frame_index = self.get_offset_framename(frame_index, offset=-2)
            color = self.loader(self.get_image_path_ox(data_path, folder, frame_index, side))
        elif i == 1:
            frame_index = self.get_offset_framename(frame_index, offset=2)
            color = self.loader(self.get_image_path_ox(data_path, folder, frame_index, side))
        else:
            color = self.loader(self.get_image_path_ox(data_path, folder, frame_index, side, is_test))

        if do_flip and not is_test:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color
    
    def get_color_ad(self, data_path, folder, frame_index, side, do_flip, i=0, is_test=False):
        if i == -1:
            frame_index = self.get_offset_framename(frame_index, offset=-3)
            color = self.loader(self.get_image_path_ad(data_path, folder, frame_index, side))
        elif i == 1:
            frame_index = self.get_offset_framename(frame_index, offset=3)
            color = self.loader(self.get_image_path_ad(data_path, folder, frame_index, side))
        else:
            color = self.loader(self.get_image_path_ad(data_path, folder, frame_index, side, is_test))

        # if do_flip:
        #     color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color
    
    def get_color_fd(self, data_path, folder, frame_index, side, do_flip, i=0, is_test=False):
        if i == -1:
            frame_index = self.get_offset_framename(frame_index, offset=-2)
            color = self.loader(self.get_image_path_fd(data_path, folder, frame_index, side))
        elif i == 1:
            frame_index = self.get_offset_framename(frame_index, offset=2)
            color = self.loader(self.get_image_path_fd(data_path, folder, frame_index, side))
        else:
            color = self.loader(self.get_image_path_fd(data_path, folder, frame_index, side, is_test))

        # if do_flip:
        #     color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color
    
    def get_color_au(self, data_path, folder, frame_index, side, do_flip, i=0, is_test=False):
        if i == -1:
            frame_index = self.get_offset_framename(frame_index, offset=-2)
            color = self.loader(self.get_image_path_au(data_path, folder, frame_index, side))
        elif i == 1:
            frame_index = self.get_offset_framename(frame_index, offset=2)
            color = self.loader(self.get_image_path_au(data_path, folder, frame_index, side))
        else:
            color = self.loader(self.get_image_path_au(data_path, folder, frame_index, side, is_test))

        # if do_flip:
        #     color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color
    
    def get_color_ds(self, data_path, folder, frame_index, side, do_flip, i=0, is_test=False):
        if i == -1:
            frame_index = self.get_offset_framename(frame_index, offset=-2)
            color = self.loader(self.get_image_path_ds(data_path, folder, frame_index, side))
        elif i == 1:
            frame_index = self.get_offset_framename(frame_index, offset=2)
            color = self.loader(self.get_image_path_ds(data_path, folder, frame_index, side))
        else:
            # print(self.get_image_path_ds(data_path, folder, frame_index, side, is_test))
            color = self.loader(self.get_image_path_ds(data_path, folder, frame_index, side, is_test))

        # if do_flip:
        #     color = color.transpose(pil.FLIP_LEFT_RIGHT)
        # print('------------------------------------------------------')
        # print(self.get_image_path_ds(data_path, folder, frame_index, side, is_test))
        # print('------------------------------------------------------')
        return color
    
    def get_color_mal(self, data_path, folder, frame_index, side, do_flip, i=0, is_test=False):

        if i == -1:
            frame_index = self.get_offset_framename(frame_index, offset=-2)
            color = self.loader(self.get_image_path_mal(data_path, folder, frame_index, side))
        elif i == 1:
            frame_index = self.get_offset_framename(frame_index, offset=2)
            color = self.loader(self.get_image_path_mal(data_path, folder, frame_index, side))
        else:
            color = self.loader(self.get_image_path_mal(data_path, folder, frame_index, side, is_test))

        # if do_flip:
        #     color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color
    
    def get_color_sim(self, data_path, folder, frame_index, side, do_flip, i=0, is_test=False):
        color = self.loader(self.get_image_path_sim(data_path, folder, int(frame_index) + i, side, is_test))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color
    
    def get_color_fov(self, data_path, folder, frame_index, side, do_flip,view, i=0, is_test=False):
        color = self.loader(self.get_image_path_fov(data_path, folder, int(frame_index) + i, side, view, is_test))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color
    

    def get_color_ddad(self, data_path, city_folder, frame_index, side, do_flip, i=0, is_test=False):
        color = self.loader(self.get_image_path_ddad(data_path, city_folder, int(frame_index) + i, side, is_test))

        # if do_flip:
        #     color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def get_color_hol(self, data_path, city_folder, frame_index, side, do_flip, i=0, is_test=False):
        color = self.loader(self.get_image_path_hol(data_path, city_folder, int(frame_index) + i, side, is_test))

        # if do_flip:
        #     color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color
    
    def get_color_ms(self, data_path, city_folder, frame_index, side, do_flip, i=0, is_test=False):
        color = self.loader(self.get_image_path_ms(data_path, city_folder, int(frame_index) + i, side, is_test))

        # if do_flip:
        #     color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color
############################## Depth Hint ############################
    def get_depth_hint_kt(self, depth_hint_path, folder, frame_idx, side, is_flip):
        side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        depth_hint_path = os.path.join(depth_hint_path, folder, "image_0{}".format(side_map[side]),
                                       "{:010d}{}".format(frame_idx, ".npy"))
        depth_hint = np.load(depth_hint_path, allow_pickle=True)[0]  # (h, w)
        # print(depth_hint.shape, depth_hint)
        if is_flip:
            depth_hint = np.fliplr(depth_hint)
        depth_hint = cv2.resize(depth_hint, dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_hint = torch.from_numpy(depth_hint).float().unsqueeze(0)  # (1, h, w)
        return depth_hint
    
    def get_depth_hint_cs(self, depth_hint_path, city_folder, frame_idx, side, is_flip):
        depth_hint_path = os.path.join(
            depth_hint_path, city_folder, side, "{:010d}{}".format(frame_idx, ".npy"))
        depth_hint = np.load(depth_hint_path, allow_pickle=True)[0]  # (h, w)
        # if is_flip:
        #     depth_hint = np.fliplr(depth_hint)
        depth_hint = cv2.resize(depth_hint, dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_hint = torch.from_numpy(depth_hint).float().unsqueeze(0)  
        return depth_hint
    
    def get_depth_hint_aps(self, depth_hint_path, folder, frame_idx, side, is_flip):
        depth_hint_path = os.path.join(
            depth_hint_path, folder, "Camera {}/".format(int(side)), "{:010d}{}".format(frame_idx, ".npy"))
        depth_hint = np.load(depth_hint_path, allow_pickle=True)[0]  # (h, w)
        # if is_flip:
        #     depth_hint = np.fliplr(depth_hint)
        depth_hint = cv2.resize(depth_hint, dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_hint = torch.from_numpy(depth_hint).float().unsqueeze(0)  # (1, h, w)
        return depth_hint
    
    def get_depth_hint_ox(self, depth_hint_path, folder, frame_idx, side, is_flip):
        depth_hint_path = os.path.join(
            depth_hint_path, folder, side, "{:010d}{}".format(frame_idx, ".npy"))
        depth_hint = np.load(depth_hint_path, allow_pickle=True)[0]  # (h, w)
        if is_flip:
            depth_hint = np.fliplr(depth_hint)
        depth_hint = cv2.resize(depth_hint, dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_hint = torch.from_numpy(depth_hint).float().unsqueeze(0)  
        return depth_hint
    
    def get_depth_hint_ad(self, depth_hint_path, folder, frame_idx, side, is_flip):
        depth_hint_path = os.path.join(
            depth_hint_path, folder, side, "{:010d}{}".format(frame_idx, ".npy"))
        depth_hint = np.load(depth_hint_path, allow_pickle=True)[0]  # (h, w)
        # if is_flip:
        #     depth_hint = np.fliplr(depth_hint)
        depth_hint = cv2.resize(depth_hint, dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_hint = torch.from_numpy(depth_hint).float().unsqueeze(0)  
        return depth_hint

    def get_depth_hint_fd(self, depth_hint_path, folder, frame_idx, side, is_flip):
        depth_hint_path = os.path.join(
            depth_hint_path, folder, side, "{:010d}{}".format(frame_idx, ".npy"))
        depth_hint = np.load(depth_hint_path, allow_pickle=True)[0]  # (h, w)
        # if is_flip:
        #     depth_hint = np.fliplr(depth_hint)
        depth_hint = cv2.resize(depth_hint, dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_hint = torch.from_numpy(depth_hint).float().unsqueeze(0)  
        return depth_hint
    
    def get_depth_hint_au(self, depth_hint_path, folder, frame_idx, side, is_flip):
        depth_hint_path = os.path.join(
            depth_hint_path, folder, side, "{:010d}{}".format(frame_idx, ".npy"))
        depth_hint = np.load(depth_hint_path, allow_pickle=True)[0]  # (h, w)
        # if is_flip:
        #     depth_hint = np.fliplr(depth_hint)
        depth_hint = cv2.resize(depth_hint, dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_hint = torch.from_numpy(depth_hint).float().unsqueeze(0)  
        return depth_hint
    
    def get_depth_hint_ds(self, depth_hint_path, folder, frame_idx, side, is_flip):
        depth_hint_path = os.path.join(
            depth_hint_path, folder, side, "{:010d}{}".format(frame_idx, ".npy"))
        depth_hint = np.load(depth_hint_path, allow_pickle=True)  # (h, w)
        # if is_flip:
        #     depth_hint = np.fliplr(depth_hint)
        depth_hint = cv2.resize(depth_hint, dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_hint = torch.from_numpy(depth_hint).float().unsqueeze(0)  
        return depth_hint

    def get_depth_hint_sim(self, depth_hint_path, folder, frame_idx, side, is_flip):
        depth_hint_path = os.path.join(
            depth_hint_path, folder, side, "{:010d}{}".format(frame_idx, ".npy"))
        depth_hint = np.load(depth_hint_path, allow_pickle=True)  # (h, w)
        # if is_flip:
        #     depth_hint = np.fliplr(depth_hint)
        depth_hint = cv2.resize(depth_hint, dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_hint = torch.from_numpy(depth_hint).float().unsqueeze(0)  
        return depth_hint

########################### Misc ################
    
    def get_offset_framename(self, frame_index, offset):
        return int(frame_index) + offset
