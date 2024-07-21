# TODO
# Check if file exists rather than try except
# remomve void vairables
# Nothing stopping us from limiting the data input from the baseline valie 
# (for the boosting stage we take in 7 images, but if that given image has 
# a basline that hits the upper bound then we can lmit the input)
# Rewrite everything

from __future__ import absolute_import, division, print_function
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import random
import numpy as np
from PIL import Image  # using pillow-simd for increased speed
import pdb
import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms import InterpolationMode

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders
    """
    def __init__(self, 
                 filenames,
                 epoch,
                 height,
                 width,
                 kt_path=None,
                 syns_path=None,
                 rand=False,
                 scales=[0],
                 trimin= False,
                 kt=False,
                 is_train=False,
                 img_ext='.jpg',
                 naive_mix=False):
        super(MonoDataset, self).__init__()
        self.kt_path = kt_path
        self.syns_path = syns_path
        self.epoch = epoch
        self.rand = rand
        self.filenames = filenames
        self.height = height
        self.width = width
        self.interp = Image.LANCZOS
        self.is_train = is_train
        self.img_ext = img_ext
        self.kt = kt
        self.scales = scales
        self.naive_mix = naive_mix
        self.trimin = trimin
        self.kt_path = self.kt_path 
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)

        if self.epoch < 10:
            self.to_use = 2 if self.trimin else 1
            self.cutt_off = 0.1 + (0.04 * self.epoch)
        else:
            self.to_use = 7 if self.trimin else 5
            self.cutt_off = (0.15*self.epoch) - 0.9

        transforms.ColorJitter.get_params(
            self.brightness, self.contrast, self.saturation, self.hue)
        self.resize = {}
        for scale in self.scales:
            s = 2 ** scale 
            self.resize[scale] = transforms.Resize((self.height // s, self.width // s),
                                            interpolation=InterpolationMode.LANCZOS)
                        
    def __getitem__(self, index):
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
        if self.naive_mix:
            inputs = {}
            if type(self).__name__ == "KITTIRAWDataset":
                if self.rand and self.is_train:
                    data_splits = self.filenames[index].split()
                    baseline = data_splits[-1]
                else:
                    baseline = 0
                folder, frame_index, side = self.index_to_folder_and_frame_idx_kt(index)
                if self.is_train:
                    if self.rand:
                        frame_idxs = sorted([i for i in range(-self.to_use, self.to_use+1) if (abs(i) * float(baseline)) <= self.cutt_off], key=abs)
                        if max(frame_idxs) < 3:
                            frame_idxs.append('s')
                    else:
                        frame_idxs = [0,1,-1,'s']
                else:
                    frame_idxs = [0]

                if self.is_train:
                    mini = random.randint(1,6) if random.random() > 0.7 else 0
                    limit_pos = max([i for i in range(1,8 - mini) if os.path.isfile(self.get_image_path_kt(self.kt_path, frame_index+i, side, folder))])
                    limit_neg = max([abs(i) for i in range(-1,-8 + mini, -1) if os.path.isfile(self.get_image_path_kt(self.kt_path, frame_index+i, side, folder))])
                    limit = min([limit_pos, limit_neg])
                else:
                    limit = 7
                
                frame_idxs[:] = [x for x in frame_idxs if x != 's' and abs(x) <= abs(limit)]
                if max(frame_idxs) < 3:
                    frame_idxs.append('s')

                if self.is_train:
                    other_side = {"r": "l", "l": "r"}[side]
                    for i in frame_idxs:
                        if i == "s":   
                            inputs[("color", "s", -1)] = self.get_color_kt(self.get_image_path_kt(self.kt_path, frame_index, other_side, folder), do_flip)
                        else:
                            file_path = self.get_image_path_kt(self.kt_path, frame_index+i, side, folder)
                            inputs[("color", i, -1)] = self.get_color_kt(file_path, do_flip)                 

                    K, inv_K = self.load_intrinsic_kt(0)
                    inputs[("K", 0)] = torch.from_numpy(K)
                    inputs[("inv_K", 0)] = torch.from_numpy(inv_K)
                else:
                    inputs[("color", 0, -1)] = self.get_color_kt(self.get_image_path_kt(self.kt_path, frame_index, side, folder), do_flip)
                if do_color_aug:
                    color_aug = transforms.ColorJitter(
                        self.brightness, self.contrast, self.saturation, self.hue)
                else:
                    color_aug = (lambda x: x)
                self.preprocess(inputs, color_aug)
                if self.is_train:
                    for i in frame_idxs:
                        del inputs[("color", i, -1)]
                        if i != 's':
                            del inputs[("color_aug", i, -1)]

                stereo_T = np.eye(4, dtype=np.float32)
                baseline_sign = -1 if do_flip else 1
                side_sign = -1 if side in ["l"] else 1
                stereo_T[0, 3] = side_sign * baseline_sign * 0.1
                inputs["stereo_T"] = torch.from_numpy(stereo_T)
                if 's' in frame_idxs:
                    frame_idxs[frame_idxs.index('s')] = -50
                inputs["frames"] = torch.tensor(frame_idxs)
                inputs["cutt_off"] = torch.tensor(self.cutt_off)
                inputs["to_use"] = torch.tensor(self.to_use)
                return inputs
            
            elif type(self).__name__ == "SYNSRAWDataset":
                folder, frame_index = self.index_to_folder_and_frame_idx(index)
                inputs[("color", 0, -1)] = self.get_color(self.syns_path, folder, frame_index)
                color_aug = (lambda x: x)
                self.preprocess(inputs, color_aug)
                del inputs[("color", 0, -1)]
                del inputs[('color_aug', 0, 0)]
                K, inv_K = self.load_intrinsic_syns()
                inputs[("K", -1)] = torch.from_numpy(K)
                inputs[("inv_K", -1)] = torch.from_numpy(inv_K)
                return inputs

    def __len__(self):
        if self.naive_mix:
            return len(self.filenames)      
        else:
            lengths = []
            for type in self.valid_datatypes:
                lengths.append(len(self.filenames[type]))
            return max(lengths)

    def preprocess(self, inputs, color_aug):
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                for scale in self.scales:
                    inputs[(n, im, scale)] = self.resize[scale](inputs[(n, im, scale - 1)])

                # inputs[(n, im, 0)] = self.resize[0](inputs[(n, im, -1)])

        avoid_scale = [1,2,3] 

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                if im == 0 or im == 's':
                    inputs[(n, im, i)] = self.to_tensor(f)
                if i not in avoid_scale and im != 's':
                    inputs[(n, im, i)] = self.to_tensor(f)
                    inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
    
