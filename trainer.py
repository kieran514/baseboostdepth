from codecs import backslashreplace_errors
from locale import RADIXCHAR
from mimetypes import init
import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import pdb
import cv2
import numpy as np
import time
import re
from tqdm.auto import tqdm
import wandb
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from utils import readlines, sec_to_hm_str
from layers import SSIM, BackprojectDepth, Project3D, transformation_from_parameters, \
    disp_to_depth, get_smooth_loss, compute_depth_errors
import datasets, networks
import matplotlib.pyplot as plt

_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)
_BRIGHT_COLORMAP = plt.get_cmap('cool', 256)

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        if not self.opt.debug:
            wandb.init(project='BMVC')
            wandb.config.update(self.opt)

        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []
        self.device = torch.device("cpu" if self.opt.no_cuda else f"cuda:{self.opt.cuda}")
        self.num_scales = len(self.opt.scales)
        self.num_pose_frames = 2
        if self.opt.naive_mix:
            assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
            self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])
            if self.opt.use_stereo:
                self.opt.frame_ids.append("s")        

        if self.opt.ViT:
            import networksvit
            self.models["encoder"] = networksvit.mpvit_small() 
            self.models["encoder"].num_ch_enc = [64,128,216,288,288] 
            self.models["encoder"].to(self.device)

            self.models["depth"] = networksvit.DepthDecoder()
            self.models["depth"].to(self.device)
        elif self.opt.SQL:
            import networksSQL
            self.models["encoder"] = networksSQL.ResnetEncoderDecoder(num_layers=50, num_features=256, model_dim=32)
            self.models["depth"] = networksSQL.Lite_Depth_Decoder_QueryTr(in_channels=32, patch_size=16, dim_out=64, embedding_dim=32, 
                                                        query_nums=64, num_heads=4, min_val=0.001, max_val=80.0)
            self.models["encoder"].to(self.device)
            self.models["depth"].to(self.device)
            self.parameters_to_train += list(self.models["encoder"].parameters())
        elif self.opt.CA_depth:
            import networksCA
            self.models["encoder"] = networksCA.ResnetEncoder(50, self.opt.weights_init == "pretrained")
            self.models["depth"] = networksCA.DepthDecoder(self.models["encoder"].num_ch_enc, self.opt.scales)

            self.models["encoder"].to(self.device)
            self.models["depth"].to(self.device)
            self.parameters_to_train += list(self.models["encoder"].parameters())
        elif self.opt.DIFFNet:
            import networksDIFF
            self.models["encoder"] = networksDIFF.test_hr_encoder.hrnet18(False)
            self.models["encoder"].num_ch_enc = [ 64, 18, 36, 72, 144 ]
            self.models["encoder"].to(self.device)

            self.models["depth"] = networksDIFF.HRDepthDecoder(self.models["encoder"].num_ch_enc, self.opt.scales)
            self.models["depth"].to(self.device)
        else:
            self.models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.models["encoder"].to(self.device)
            self.models["depth"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales)
            self.models["depth"].to(self.device)
            self.parameters_to_train += list(self.models["encoder"].parameters())

        self.parameters_to_train += list(self.models["depth"].parameters())
        self.models["pose_encoder"] = networks.ResnetEncoder(
                                        18, 
                                        self.opt.weights_init == "pretrained",
                                        num_input_images=self.num_pose_frames)
        self.models["pose_encoder"].to(self.device)
        self.models["pose"] = networks.PoseDecoder(
                                    self.models["pose_encoder"].num_ch_enc,
                                    num_input_features=1,
                                    num_frames_to_predict_for=2)
        self.models["pose"].to(self.device)
        self.parameters_to_train += list(self.models["pose_encoder"].parameters())
        self.parameters_to_train += list(self.models["pose"].parameters())
        if self.opt.ViT:
            self.params = [ {"params":self.parameters_to_train, "lr": 1e-4},{"params": list(self.models["encoder"].parameters()), "lr": 5e-5} ]
            self.model_optimizer = optim.AdamW(self.params)
            self.model_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.model_optimizer, milestones=[11, 13, 15, 16, 17, 18, 19], gamma=0.4) 
        else:
            self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
            self.model_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.model_optimizer, milestones=[11, 13, 15, 16, 17, 18, 19], gamma=0.4) 

        if self.opt.load_weights_folder != 'None':
            print("Loading weights")
            self.load_model()
            
        print("Training model named:\n  ", self.opt.model_name)
        print("Training is using:\n  ", self.device)
        self.dataset = datasets.KITTIRAWDataset
        if self.opt.naive_mix:        
            filenames = readlines(os.path.join(os.path.dirname(__file__), "splits", 'eigen_zhou', f"{self.opt.training_file}.txt"))
            self.train_filenames = filenames
            self.img_ext = '.png' if self.opt.png else '.jpg'
        val_filenames = {}
        if self.opt.naive_mix:
            val_filenames = readlines(os.path.join(os.path.dirname(__file__), "splits", 'eigen_zhou', "val_files.txt"))
        val_dataset = self.dataset(val_filenames, 0, self.opt.height, self.opt.width, 
                                   kt_path=self.opt.kt_path, is_train=False, img_ext='.jpg', kt=self.opt.kt, naive_mix = self.opt.naive_mix)
        self.val_loader = DataLoader(val_dataset, 1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
        if self.opt.SYNS_eval:
            val_filenames_syns = {}
            val_filenames_syns = readlines(os.path.join(os.path.dirname(__file__), "splits", 'SYNS', "val_files.txt"))
            val_dataset_syns = datasets.SYNSRAWDataset(val_filenames_syns, 0, self.opt.height, self.opt.width,
                                                       syns_path=self.opt.syns_path, is_train=False, naive_mix=self.opt.naive_mix)
            self.val_loader_syns = DataLoader(val_dataset_syns, 1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
            self.depth_metric_names_syns = ["edge_Acc", "edge_comp"]
            self.best_syns = 100.0
        self.ssim = SSIM()
        self.ssim.to(self.device)
        self.backproject_depth = {}
        self.project_3d = {}
        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        if self.opt.naive_mix:
            self.len_train_dataset = len(self.train_filenames)
            self.num_total_steps = self.len_train_dataset // self.opt.batch_size * self.opt.num_epochs
            print("There are {:d} training items and {:d} validation items\n".format(
                self.len_train_dataset, len(val_filenames)))
        gt_path = os.path.join('splits', 'eigen_zhou', "gt_depths.npz")
        self.gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
        if self.opt.SYNS_eval:
            gt_path_syns_edges = os.path.join('splits', "SYNS", "gt_edges.npz")
            self.gt_edges_syns = np.load(gt_path_syns_edges, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
            gt_path_syns = os.path.join('splits', "SYNS", "gt_depths.npz")
            self.gt_depth_syns = np.load(gt_path_syns, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
        self.save_opts()
        self.best = 10.0
        
    def set_train(self):
        for m in self.models.values():
            m.train()

    def set_eval(self):
        for m in self.models.values():
            m.eval()

    def train(self):
        if self.opt.load_weights_folder != 'None':
            number = self.opt.load_weights_folder.split("_")[-1]
            if number == 'best':
                self.epoch = 10
                number = 0
            else:
                try:
                    self.epoch = int(number) + 1
                except:
                    self.epoch = 10
            start = self.epoch
            self.step = self.epoch * (self.len_train_dataset // self.opt.batch_size)
            for _ in range(start):
                self.model_lr_scheduler.step()
        else:
            self.epoch = 0
            self.step = 0
            start = 0
        self.start_time = time.time()
        for self.epoch in range(start, self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
        self.models = init
        if not self.opt.debug:
            wandb.join()
        
    def run_epoch(self):
        self.model_lr_scheduler.step()
        if self.opt.ViT:
            depth_lr = self.model_optimizer.param_groups[1]['lr']
            pose_lr = self.model_optimizer.param_groups[0]['lr']
            print(f'\nStarting from epoch {self.epoch} and current learning rate for depth is {depth_lr} and pose lr is {pose_lr}')
        else:
            starting_lr = self.model_optimizer.param_groups[0]['lr']
            print(f'\nStarting from epoch {self.epoch} and current learning rate is {starting_lr}')

        self.set_train() 
        if self.opt.naive_mix:
            if self.opt.rand:
                if self.epoch < 10 and not self.opt.SQL:
                    self.opt.scales = [0,1,2,3]
                else:
                    self.opt.scales = [0]
                
                train_dataset = self.dataset(self.train_filenames, self.epoch, self.opt.height, self.opt.width,
                                            kt_path=self.opt.kt_path, rand=self.opt.rand, is_train=True, scales=self.opt.scales,
                                            img_ext=self.img_ext, kt=self.opt.kt, naive_mix=self.opt.naive_mix, 
                                            trimin=self.opt.trimin)     
                self.train_loader = DataLoader(train_dataset, self.opt.batch_size, shuffle=True, 
                                               collate_fn=self.custom_collate, num_workers=self.opt.num_workers, 
                                               pin_memory=True, drop_last=True)
                
                self.num_total_steps = len(self.train_filenames) // self.opt.batch_size * self.opt.num_epochs
                
                for scale in self.opt.scales:
                    h = self.opt.height // (2 ** scale)
                    w = self.opt.width // (2 ** scale)
                    self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
                    self.backproject_depth[scale].to(self.device)
                    self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
                    self.project_3d[scale].to(self.device)

            for batch_idx, inputs in enumerate(tqdm(self.train_loader, mininterval=2)):
                before_op_time = time.time()
                self.batch_idx = batch_idx
                if self.opt.rand and batch_idx == 0 and self.epoch == 0:
                    print('----------------------------------!! BaseBoostDepth Is Training !!----------------------------------')
                    print(f"You are using:\n"
                        f"Curriculum Learning Optimization: {self.opt.rand}\n"
                        f"Tri-Minimization: {self.opt.trimin}\n"
                        f"Incremental Pose: {self.opt.incremental_skip}\n"
                        f"Partial Pose: {self.opt.partial_skip}\n"
                        f"Error-induced: {self.opt.decomp}\n")
                    
                if self.opt.rand:
                    def custom_key_func(item):
                        if isinstance(item, str):
                            return float('inf') 
                        else:
                            return abs(item) 
                    self.opt.frame_ids = sorted(inputs['frames'], key=custom_key_func)
                    if 's' in self.opt.frame_ids:
                        assert self.opt.frame_ids[-1] == 's'
                else:
                    frames_ = inputs['frames'][0].numpy().tolist()
                    if 50 in frames_:
                        frames_[frames_.index(50)] = 's'
                    self.opt.frame_ids = frames_

                self.early_phase = (batch_idx % self.opt.log_frequency)
                outputs, losses = self.process_batch(inputs)
                self.model_optimizer.zero_grad()
                losses["loss"].backward()
                self.model_optimizer.step()
                duration = time.time() - before_op_time

                if self.early_phase == 0 and batch_idx >0:
                    duration = time.time() - before_op_time
                    starting_lr = self.model_optimizer.param_groups[0]['lr']
                    print('---------------------------------------------------------------------------------')
                    self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                    print(f'Starting from epoch {self.epoch} and current learning rate is {starting_lr}')
                    print(f"Ordering: {inputs['ordering']}")
                    print(f"Scales: {self.opt.scales}")
                    print(f"Valid Frames: {self.valid_frames}")
                    print(f"Current Boosting Weight: {inputs['cutt']}")
                    print(f"Omega: {inputs['to_use']}")
                    # print('Incremental', {key: (100 * (torch.sum(torch.cat(values, 0))/(self.opt.height*self.opt.width)) /
                    #                             len(values[0])).item() for key, values in self.ident[0].items()})
                    # print('Full guided', {key: (100 * (torch.sum(torch.cat(values, 0))/(self.opt.height*self.opt.width)) /
                    #                             len(values[0])).item() for key, values in self.ident[1].items() if values})
                    print('---------------------------------------------------------------------------------')
                    self.log("train", inputs, outputs, losses)
                    self.val('run')
                self.step += 1

    def process_batch(self, inputs, batch_idx=None, is_train=True):
        for key, ipt in inputs.items():
            if key not in ["frames", "ordering", "cutt"]:
                inputs[key] = ipt.to(self.device)

        if is_train:
            self.valid_frames = list(set([el for sublist in inputs['ordering'] for el in sublist if el !=0]))
            self.valid_frames_trimin(inputs)
            outputs = self.predict_poses(inputs)
            feats = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs.update(self.models['depth'](feats))
            outputs.update(self.generate_images_pred(inputs, outputs))
            losses = self.compute_losses(inputs, outputs)
        else:
            outputs = {}
            feats = self.models["encoder"](inputs["color", 0, 0])
            outputs.update(self.models['depth'](feats))
            if self.opt.SQL:
                outputs["depth", 0, 0] = outputs["disp", 0]
            else:
                _, outputs["depth", 0, 0] = disp_to_depth(outputs["disp", 0], self.opt.min_depth, self.opt.max_depth)
            losses = None
        return outputs, losses
    
    def predict_poses(self, inputs):
        outputs = {}        
        self.valid_mask_dict_pose = {}
        self.valid_mask_pose = {}
        frame_ids = self.opt.frame_ids[1:]
        ordering = inputs['ordering']

        for frames in frame_ids:
            if frames != 's' and frames > 0:
                self.valid_mask_dict_pose[frames-1] = [f_i[1] >= frames if f_i[1] != 's' else False for f_i in ordering]
            if frames != 's' and frames == 1:
                self.valid_mask_pose[frames-1] = self.valid_mask_dict_pose[frames-1]
            elif frames != 's' and frames > 1:
                self.valid_mask_pose[frames-1] = [ba for num, ba in enumerate(self.valid_mask_dict_pose[frames-1]) if self.valid_mask_dict_pose[frames-2][num]]

        if self.opt.partial_skip:
            self.partial = {}
            self.partial_mask = {}
            self.partial_mask_dict = {}
            for frames in frame_ids:
                if frames != 's':
                    self.partial[frames] = [abs(frames) == f_i[1]-2 for f_i in ordering if f_i[1] != 's']
            for frames in frame_ids:
                if frames != 's':
                    if frames > 0:
                        self.partial_mask_dict[frames] = [f_i[1] >= frames if f_i[1] != 's' else False for f_i in ordering]
        
        if self.opt.trimin:
            hol = list(set([el for sublist in ordering for el in sublist if el !='s' and el != 0]))
            copyed_hol = hol.copy()
            hol = self.adding_to_hol(hol, copyed_hol)            
            self.valid_frames_pose = list(set(hol))
        else:
            self.valid_frames_pose = list(set([el for sublist in ordering for el in sublist if el !='s' and el !=0]))

        self.maxing_valid_frames = (bool(self.valid_frames_pose) and max(self.valid_frames_pose)>2)
        self.maxing_valid_frames = inputs['cutt'].item() > 0.5

        if self.opt.incremental_skip and self.maxing_valid_frames:
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids if f_i != "s"} 
            for f_i in frame_ids:
                if f_i != "s":
                    if abs(f_i) > 1:
                        changer = f_i + 1 if f_i < -1 else f_i - 1
                            
                        pose_feats[changer] = inputs["color_aug", changer, 0][self.valid_mask_pose[abs(changer)]]
                        pose_inputs = [pose_feats[f_i], pose_feats[f_i+1]] if f_i < 0 else [pose_feats[f_i-1], pose_feats[f_i]]
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                        axisangle_iter, translation_iter = self.models["pose"](pose_inputs)                            
                        outputs[("cam_T_cam_step", f_i+1 if f_i < 0 else f_i-1, f_i)] = transformation_from_parameters(
                            axisangle_iter[:, 0], translation_iter[:, 0], invert=(f_i < 0))
                        
                        if f_i in self.valid_frames_pose:
                            T_rel_cumulative = torch.eye(4).unsqueeze(0).expand(outputs[("cam_T_cam_step", f_i+1 if f_i < 0 else f_i-1, f_i)].shape[0], -1, -1).to(self.device)
                            for frame in range(f_i, 0, -1):
                                multi_all = torch.stack([batch for num, batch in enumerate(outputs[("cam_T_cam_step", frame-1 if f_i > 0 else frame+1, frame)]) 
                                                            if frame == f_i or self.valid_mask_pose[abs(frame)][num]], dim=0)
                                remain = abs(f_i - frame)
                                if remain >= 2:
                                    for extra in range(1, remain):
                                        multi_all = torch.stack([batch for num, batch in enumerate(multi_all) 
                                                                    if self.valid_mask_pose[abs(frame)+extra][num]], dim=0)
                                T_rel_cumulative = torch.matmul(T_rel_cumulative, multi_all)
                            outputs[("cam_T_cam", 0, f_i)] = T_rel_cumulative
                                                        
                            if self.opt.decomp:
                                outputs[("cam_T_cam_error", 0, f_i)] = outputs[("cam_T_cam", 0, f_i)].clone().detach()
                                outputs[("cam_T_cam_error", 0, f_i)][:,:3,3:] /= self.opt.pose_error

                    else:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]] if f_i < 0 else [pose_feats[0], pose_feats[f_i]]                            
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                        axisangle, translation = self.models["pose"](pose_inputs)
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                        outputs[("cam_T_cam_step", 0, f_i)] = outputs[("cam_T_cam", 0, f_i)].clone()
                        if self.opt.decomp:
                            outputs[("cam_T_cam_error", 0, f_i)] = outputs[("cam_T_cam", 0, f_i)].clone().detach()
                            outputs[("cam_T_cam_error", 0, f_i)][:,:3,3:] /= self.opt.pose_error
        else:
            for f_i in self.valid_frames:
                if f_i != "s":
                    if self.opt.trimin:
                        mid = inputs["color_aug", 0, 0][self.valid_tri_mask_dict[abs(f_i)]]
                        either_way = inputs["color_aug", f_i, 0][self.valid_tri_mask[abs(f_i)]]
                    else:
                        mid = inputs["color_aug", 0, 0][self.valid_mask_dict[abs(f_i)]]
                        either_way = inputs["color_aug", f_i, 0][self.valid_mask[abs(f_i)]]
                    pose_inputs = [either_way, mid] if f_i < 0 else [mid, either_way]
                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                    if self.opt.decomp:
                        outputs[("cam_T_cam_error", 0, f_i)] = outputs[("cam_T_cam", 0, f_i)].clone().detach()
                        outputs[("cam_T_cam_error", 0, f_i)][:,:3,3:] /= self.opt.pose_error

        if self.opt.partial_skip and self.maxing_valid_frames:
            for f_i in self.valid_frames:
                if f_i != 's' and abs(f_i) > 1:
                    mid = inputs["color_aug", 0, 0][self.partial_mask_dict[abs(f_i)]]
                    either_way = inputs["color_aug", f_i, 0] 
                    pose_inputs = [either_way, mid] if f_i < 0 else [mid, either_way]
                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    axisangle, translation = self.models["pose"](pose_inputs)
                    replaced = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                    replaced[:,:,:3] = outputs[("cam_T_cam", 0, f_i)][:,:,:3] 
                    outputs[("cam_T_cam", 0, f_i)] = torch.stack([camera_point if self.partial[f_i][num] else replaced[num] 
                                                                    for num, camera_point in enumerate(outputs[("cam_T_cam", 0, f_i)])], dim=0)
        return outputs
            
    def warping_block_for_easy_looking(self, frame, scale, inputs, outputs, frame_size, T, mask_valid, mask_valid_dict, T_error=None):
        source_scale = 0

        if frame == 's':
            images = inputs[("color", frame, source_scale)] 
            depth = outputs[("depth", 0, scale)][mask_valid_dict[frame]]
        else:
            images = inputs[("color", frame, source_scale)][mask_valid[abs(frame)]]
            depth = outputs[("depth", 0, scale)][mask_valid_dict[abs(frame)]]

        K = inputs[("K", source_scale)][:frame_size]
        inv_k = inputs[("inv_K", source_scale)][:frame_size]

        cam_points = self.backproject_depth[source_scale](
            depth, inv_k)
        
        if self.opt.decomp and frame != 's':
            pix_coords_error = self.project_3d[source_scale](cam_points, K, T_error)
            outputs[("color_D", frame, scale)] = F.grid_sample(images, pix_coords_error, align_corners=True, padding_mode="border")
            
        pix_coords = self.project_3d[source_scale](cam_points, K, T)
        outputs[("color", frame, scale)] = F.grid_sample(images, pix_coords, align_corners=True, padding_mode="border")

    def generate_images_pred(self, inputs, outputs):
        if self.opt.trimin:
            mask_valid=self.valid_tri_mask 
            mask_valid_dict=self.valid_tri_mask_dict
        else:
            mask_valid=self.valid_mask
            mask_valid_dict=self.valid_mask_dict
        
        iterations = self.valid_frames

        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            if self.opt.SQL:
                depth = disp
            else:
                _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth
            for frame in iterations:
                if frame == "s":
                    T = inputs["stereo_T"][mask_valid_dict['s']]
                    T_error = None
                else:
                    if self.opt.incremental_skip and self.maxing_valid_frames:
                        T = outputs[("cam_T_cam", 0, frame)][mask_valid[abs(frame)]]
                        T_error = outputs[("cam_T_cam_error", 0, frame)][mask_valid[abs(frame)]] if self.opt.decomp else None 
                    else:
                        T_error = outputs[("cam_T_cam_error", 0, frame)] if self.opt.decomp else None
                        T = outputs[("cam_T_cam", 0, frame)]
                frame_size = T.shape[0]
                self.warping_block_for_easy_looking(frame, scale, inputs, outputs, frame_size, T, mask_valid, mask_valid_dict, T_error)
        return outputs
    
    def compute_reprojection_loss(self, pred, target):
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return reprojection_loss
            
    def compute_losses(self, inputs, outputs):
        losses = {}
        total_loss = 0
        source_scale = 0
        frames = self.valid_frames

        if self.opt.trimin:
            mask_valid = self.valid_tri_mask
            mask_valid_dict = self.valid_tri_mask_dict
        else:
            mask_valid = self.valid_mask
            mask_valid_dict = self.valid_mask_dict

        target = {frame_id: inputs[("color", 0, source_scale)][mask_valid_dict[abs(frame_id)]] if frame_id != 's' 
                else inputs[("color", 0, source_scale)][mask_valid_dict[frame_id]]
                for frame_id in frames}

        ident_color = {frame_id: inputs[("color", frame_id, source_scale)][mask_valid[abs(frame_id)]] if frame_id != 's' 
                    else inputs[("color", frame_id, source_scale)] for frame_id in frames}

        identity_reprojection_losses = {frame_id: self.compute_reprojection_loss(ident_color[frame_id], target[frame_id]) for frame_id in frames}

        pattern = re.compile(r'-?\d+')

        if self.opt.trimin:
            identity_reprojection_losses = {keys if keys[0] != ' ' else  keys[1:] : identity_reprojection_losses[int(pattern.search(keys).group())][self.valid_tri_mask_reverse[keys]] if keys[0] != 's' 
                            else identity_reprojection_losses[keys[0]][self.valid_tri_mask_reverse[keys]]
                            for keys in self.valid_tri_mask_reverse.keys()} 
            iterds_ = list(set([el for sublist in inputs['ordering'] for el in sublist if el !=0]))
            temp_positive = [frame_id for frame_id in iterds_ if frame_id == 's' or frame_id > 0]
            noise = {frame_id: torch.randn(identity_reprojection_losses[f'{frame_id}'].shape, device=self.device) * 0.00001 
                    for frame_id in temp_positive}
        else:
            temp_positive = [frame_id for frame_id in frames if frame_id == 's' or frame_id > 0]
            noise = {frame_id: torch.randn(identity_reprojection_losses[frame_id].shape, device=self.device) * 0.00001 
                    for frame_id in temp_positive}

        for scale in self.opt.scales:
            loss = 0
            color = inputs[("color", 0, scale)]
            disp = outputs[("disp", scale)]
            reprojection_losses = {frame_id: self.compute_reprojection_loss(outputs[("color", frame_id, scale)], target[frame_id]) for frame_id in frames}
                
            if self.opt.decomp:
                reprojection_losses_D = {frame_id: self.compute_reprojection_loss(outputs[("color_D", frame_id, scale)], target[frame_id]) 
                                       for frame_id in frames if frame_id != 's'}
                
            if self.opt.trimin:
                reprojection_losses = {keys if keys[0] != ' ' else  keys[1:] : reprojection_losses[int(pattern.search(keys).group())][self.valid_tri_mask_reverse[keys]] if keys[0] != 's' 
                                else reprojection_losses[keys[0]][self.valid_tri_mask_reverse[keys]]
                                for keys in self.valid_tri_mask_reverse.keys()} 
                reprojection_losses_D = {keys if keys[0] != ' ' else  keys[1:]  : reprojection_losses_D[int(pattern.search(keys).group())][self.valid_tri_mask_reverse[keys]]
                                        for keys in self.valid_tri_mask_reverse.keys() if keys[0] != 's' } if self.opt.decomp else None
                                            
            to_cat = []
            dictor_norm = {(frame[1], 'norm') : [] for frame in inputs['ordering']}
            dictor_guide = {(frame[1], 'guide') : [] for frame in inputs['ordering']}
            if self.opt.trimin:
                to_optimise, ident = self.x_min_opt(temp_positive, reprojection_losses, identity_reprojection_losses, noise, reprojection_losses_D, to_cat, dictor_norm, dictor_guide)
                self.ident = ident                               
            else:
                to_optimise = torch.cat([torch.min(torch.cat([reprojection_losses[frame_id], reprojection_losses[-frame_id], 
                                identity_reprojection_losses[frame_id]+noise[frame_id], 
                                identity_reprojection_losses[-frame_id]+noise[frame_id]], dim=1), dim=1)[0] 
                                if frame_id != 's' else 
                                torch.min(torch.cat([reprojection_losses[frame_id],
                                identity_reprojection_losses[frame_id]+noise[frame_id]], dim=1), dim=1)[0]
                                for frame_id in temp_positive], dim=0)     
                               
            loss += to_optimise.mean()
            if color.shape[-2:] != disp.shape[-2:] and self.opt.SQL:
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, outputs, losses, idx, SYNS=False, accumulate=False):
        min_depth = 1e-3
        max_depth = 80
        depth_pred = outputs["depth", 0, 0]
        if SYNS: 
            gt_depth = self.gt_depth_syns[idx]
            gt_edge = self.gt_edges_syns[idx]
            gt_height, gt_width = gt_edge.shape[:2]
            depth_pred = torch.clamp(F.interpolate(depth_pred, [gt_height, gt_width], mode="bilinear", align_corners=False), 1e-3, 80)
            depth_pred = depth_pred.detach().squeeze().cpu().numpy()
            depth_e = self.to_log(depth_pred)
            depth_e = cv2.GaussianBlur(depth_e, (3, 3), sigmaX=1, sigmaY=1)
            dx = cv2.Sobel(src=depth_e, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
            dy = cv2.Sobel(src=depth_e, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
            edges = np.sqrt(dx**2 + dy**2)[..., None]
            pred_edge = edges > edges.mean()
            mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)
            depth_errors = compute_depth_errors(gt_edge, pred_edge, mask, SYNS)
            for i, metric in enumerate(self.depth_metric_names_syns):
                if accumulate:
                    losses[metric] += np.array(depth_errors[i])
                else:
                    losses[metric] = np.array(depth_errors[i])
        else:
            gt_depth = self.gt_depths[idx]
            gt_height, gt_width = gt_depth.shape[:2]
            depth_pred = torch.clamp(F.interpolate(depth_pred, [gt_height, gt_width], mode="bilinear", align_corners=False), 1e-3, 80)
            depth_pred = depth_pred.detach().squeeze()
            mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                            0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)                
            gt_depth = torch.from_numpy(gt_depth).to(self.device)
            mask = torch.from_numpy(mask).to(self.device)

            depth_pred *= torch.median(gt_depth[mask]) / torch.median(depth_pred[mask])
            depth_pred = torch.clamp(depth_pred, min=min_depth, max=max_depth)
            depth_errors = compute_depth_errors(gt_depth[mask], depth_pred[mask])

            for i, metric in enumerate(self.depth_metric_names):
                if accumulate:
                    losses[metric] += np.array(depth_errors[i].cpu())
                else:
                    losses[metric] = np.array(depth_errors[i].cpu())
    
    def to_log(self, depth):
        depth = (depth > 0) * np.log(depth.clip(min=1.1920928955078125e-07))
        return depth
    
    def val(self, is_init):
        self.set_eval()
        losses = {}
        for metric in self.depth_metric_names:
            losses[metric] = 0.0
        total_batches = 0.0
        with torch.no_grad():
            for batch_idx, inputs in enumerate(tqdm(self.val_loader, mininterval=5, colour='red')):
                total_batches += 1.0
                outputs, _ = self.process_batch(inputs, batch_idx, is_train=False)
                self.compute_depth_losses(outputs, losses, batch_idx, accumulate=True)
        print('Val result:')
        for metric in self.depth_metric_names:
            losses[metric] /= total_batches
            print(metric, ': ', losses[metric])
        self.log('val', inputs, outputs, losses)
        
        if losses["de/abs_rel"] < self.best:
            self.best = losses["de/abs_rel"]
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Saving best result: ", self.best)
        if not is_init == 'init':
            self.save_model(f'{self.epoch}_{self.batch_idx}_absrel_{losses["de/abs_rel"]}')

        if self.opt.SYNS_eval:
            losses = {}
            for metric in self.depth_metric_names_syns:
                losses[metric] = 0.0
            total_batches = 0.0
            with torch.no_grad():
                for batch_idx, inputs in enumerate(tqdm(self.val_loader_syns, mininterval=5, colour='red')):
                    total_batches += 1.0
                    outputs, _ = self.process_batch(inputs, batch_idx, is_train=False)
                    self.compute_depth_losses(outputs, losses, batch_idx, SYNS=self.opt.SYNS_eval, accumulate=True)
            print('Val result:')
            for metric in self.depth_metric_names_syns:
                losses[metric] /= total_batches
                print(metric, ': ', losses[metric])
            self.log('val', inputs, outputs, losses)
            if losses["edge_comp"] < self.best_syns:
                self.best_syns = losses["edge_comp"]
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Saving best result: ", self.best_syns)
        del inputs, outputs, losses
        self.set_train() 

    def log_time(self, batch_idx, duration, loss):
        samples_per_sec = self.opt.batch_size / duration

        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        if mode == 'val':
            frameIDs = [0]
            BS = 1
        else:
            if self.opt.rand:
                frameIDs = list(set([el for sublist in inputs['ordering'] for el in sublist if el !=0]))
            else:
                frameIDs = self.opt.frame_ids
                BS = self.opt.batch_size      

        if not self.opt.debug:

            for l, v in losses.items():
                if l != "reprojection_losses":
                    wandb.log({"{}_{}".format(mode, l): v}, step=self.step)

            if self.opt.rand:
                if mode == 'train':
                    if self.opt.trimin or self.opt.x_min:
                        if 's' in frameIDs:
                            special_s_mask = [sublist[1] == 's' for sublist in inputs['ordering'] if sublist[1] == 1 or sublist[1] == 2 or sublist[1] == 's']

                        targets = {frame_id: inputs[("color", 0, 0)][self.valid_tri_mask_dict[abs(frame_id)]] if frame_id != 's' 
                                    else inputs[("color", 0, 0)][self.valid_mask_dict[frame_id]]
                                    for frame_id in frameIDs}
                        targets = {frame_id: (targets[frame_id][self.valid_tri_mask_reverse[f"{frame_id}"]] if frame_id < 0
                                    else targets[frame_id][self.valid_tri_mask_reverse[f" {frame_id}"]])
                                    if frame_id != 's' else targets[frame_id]
                                    for frame_id in frameIDs}
                        inputs_frames = {frame_id: inputs[("color", frame_id, 0)][self.valid_tri_mask[abs(frame_id)]] if frame_id != 's' 
                                        else inputs[("color", frame_id, 0)]
                                    for frame_id in frameIDs}
                        inputs_frames = {frame_id: (inputs_frames[frame_id][self.valid_tri_mask_reverse[f"{frame_id}"]] if frame_id < 0
                                    else inputs_frames[frame_id][self.valid_tri_mask_reverse[f" {frame_id}"]])
                                    if frame_id != 's' else inputs_frames[frame_id][special_s_mask]
                                    for frame_id in frameIDs}
                        disp = {frame_id: outputs[("disp", 0)][self.valid_tri_mask_dict[abs(frame_id)]] if frame_id != 's' 
                                    else outputs[("disp", 0)][self.valid_mask_dict[frame_id]]
                                    for frame_id in frameIDs}
                        disp = {frame_id: (disp[frame_id][self.valid_tri_mask_reverse[f"{frame_id}"]] if frame_id < 0
                                    else disp[frame_id][self.valid_tri_mask_reverse[f" {frame_id}"]])
                                    if frame_id != 's' else disp[frame_id]
                                    for frame_id in frameIDs}
                        warps = {frame_id: outputs[("color", frame_id, 0)][self.valid_tri_mask_reverse[f"{frame_id}"]] if frame_id == 's' or frame_id < 0
                                else outputs[("color", frame_id, 0)][self.valid_tri_mask_reverse[f" {frame_id}"]]
                                for frame_id in frameIDs}
                    else:
                        warps = {frame_id: outputs[("color", frame_id, 0)] for frame_id in frameIDs}
                        targets = {frame_id: inputs[("color", 0, 0)][self.valid_mask_dict[abs(frame_id)]] if frame_id != 's' 
                                    else inputs[("color", 0, 0)][self.valid_mask_dict[frame_id]] 
                                    for frame_id in frameIDs}         
                        inputs_frames = {frame_id: inputs[("color", frame_id, 0)][self.valid_mask[abs(frame_id)]] if frame_id != 's' 
                                        else inputs[("color", frame_id, 0)]
                                        for frame_id in frameIDs}
                        disp = {frame_id: outputs[("disp", 0)][self.valid_mask_dict[abs(frame_id)]] if frame_id != 's' 
                                    else outputs[("disp", 0)][self.valid_mask_dict[frame_id]]
                                    for frame_id in frameIDs}
                for frame_id in frameIDs:
                    if mode == 'train':
                        logimg = wandb.Image(targets[frame_id][0].data.permute(1,2,0).cpu().numpy())
                        wandb.log({"{}/trail_{}/{}".format(mode, frame_id, 'target'): logimg}, step=self.step)
                        logimg = wandb.Image(inputs_frames[frame_id][0].data.permute(1,2,0).cpu().numpy())
                        wandb.log({"{}/trail_{}/{}".format(mode, frame_id, 'input'): logimg}, step=self.step)
                        logimg = wandb.Image(warps[frame_id][0].data.permute(1,2,0).cpu().numpy())
                        wandb.log({"{}/trail_{}/{}".format(mode, frame_id, 'warp'): logimg}, step=self.step)
                        if self.opt.SQL:
                            logimg = wandb.Image(colormap(1/ disp[frame_id][0, 0]).transpose(1,2,0))
                            wandb.log({"{}/trail_{}/{}".format(mode, frame_id, 'disp'): logimg}, step=self.step)
                        else:
                            logimg = wandb.Image(colormap(disp[frame_id][0, 0]).transpose(1,2,0))
                            wandb.log({"{}/trail_{}/{}".format(mode, frame_id, 'disp'): logimg}, step=self.step)
                    else:
                        logimg = wandb.Image(inputs[("color", 0, 0)][0].data.permute(1,2,0).cpu().numpy())
                        wandb.log({"{}/trail_{}/{}".format(mode, frame_id, 'target'): logimg}, step=self.step)
                        if self.opt.SQL or self.opt.SQL_L:
                            logimg = wandb.Image(colormap(1/outputs[("disp", 0)][0, 0]).transpose(1,2,0))
                            wandb.log({"{}/trail_{}/{}".format(mode, frame_id, 'disp'): logimg}, step=self.step)
                        else:
                            logimg = wandb.Image(colormap(outputs[("disp", 0)][0, 0]).transpose(1,2,0))
                            wandb.log({"{}/trail_{}/{}".format(mode, frame_id, 'disp'): logimg}, step=self.step)
            else:
                for j in range(min(4, BS)): 
                    s = 0  
                    for frame_id in frameIDs:
                        logimg = wandb.Image(inputs[("color", frame_id, s)][j].data.permute(1,2,0).cpu().numpy())
                        wandb.log({"{}/color_{}_{}/{}".format(mode, frame_id, s, j): logimg}, step=self.step)
                        if mode == "train":
                            logimg = wandb.Image(inputs[("color", frame_id, s)][j].data.permute(1,2,0).cpu().numpy())
                            wandb.log({"{}/color_{}_{}/{}".format(mode, frame_id, s, j): logimg}, step=self.step)
                        if s == 0 and frame_id != 0 and mode == 'train':
                            logimg = wandb.Image(outputs[("color", frame_id, s)][j].data.permute(1,2,0).cpu().numpy())
                            wandb.log({"{}/color_pred_{}_{}/{}".format(mode, frame_id, s, j): logimg}, step=self.step)
                    disp = wandb.Image(colormap(outputs[("disp", s)][j, 0]).transpose(1,2,0))
                    wandb.log({"{}/disp_multi_{}/{}".format(mode, s, j): logimg}, step=self.step)

    def save_opts(self):
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, name=None, save_step=False):
        if name is not None:
            save_folder = os.path.join(self.log_path, "models", "weights_{}".format(name))
        else:
            if save_step:
                save_folder = os.path.join(self.log_path, "models", "weights_{}_{}".format(self.epoch,
                                                                                        self.step))
            else:
                save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)
        print(f'saved model to {save_path}')

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location=self.device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path) and (not self.opt.use_bright) and self.opt.scales == [0,1,2,3]:
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path, map_location=self.device)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    def eval_depth(self, disp, idx):
        min_depth = 1e-3
        max_depth = 80
        depth_gt = torch.Tensor(self.gt_depths_corr[idx]).unsqueeze(0).unsqueeze(0).to(self.device)

        b, _, h, w = depth_gt.shape
        if self.opt.SQL or self.opt.SQL_L:
            depth_pred = disp
        else:
            _, depth_pred = disp_to_depth(disp, min_depth, max_depth)

        depth_pred = torch.clamp(F.interpolate(depth_pred, [h, w], mode="bilinear", align_corners=False), min_depth, max_depth)
        depth_pred = depth_pred.detach()
        mask = depth_gt > 0
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask
        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)
        depth_pred = torch.clamp(depth_pred, min=min_depth, max=max_depth)
        depth_errors = compute_depth_errors(depth_gt, depth_pred)
        return depth_errors
    
    def adding_to_hol(self, hol, copyed_hol):
        for el in copyed_hol:
            if self.opt.trimin:
                if el > 0:
                    if el-1 > 0:
                        hol.append(el-1)
                        hol.append(-el+1)
                    if el-2 > 0:
                        hol.append(el-2)
                        hol.append(-el+2)
        return hol

    def custom_collate(self, batch):
        input = {}
        max_frames = [torch.max(frame['frames']).item() for frame in batch]
        input['ordering'] = [[0, "s"] if frame == 0 else [0, frame, -frame] for frame in max_frames]
        max_of_frames = max(max_frames)
        if max_of_frames == 0:
            frame_ids = [0, 's']
        else:
            frame_ids = [value for value in range(-max_of_frames, max_of_frames+1)]
            if any(x in max_frames for x in [0, 1, 2]):
                frame_ids.append('s')
        keys_of_interest = [("color_aug", i, 0) for i in frame_ids if i != 's']
        keys_of_interest.extend([("color", i, s) if i == 0 else ("color", i, 0) for i in frame_ids for s in self.opt.scales])
        keys_of_interest.extend([('K', 0), ('inv_K', 0), "stereo_T"])
        for key in keys_of_interest:
            input[key] = torch.stack([ba[key] for ba in batch if key in ba], dim=0)
        input['frames'] = frame_ids
        input['cutt'] = batch[0]["cutt_off"]
        input['to_use'] = batch[0]["to_use"]
        return input

    def valid_frames_trimin(self, inputs):
        self.valid_mask_dict = {}
        self.valid_mask = {}

        for sets, frames in enumerate(self.opt.frame_ids[1:]):

            if 's' in self.valid_frames and sets == 0:
                self.valid_mask_dict['s'] = [f_i[1] == 's' for f_i in inputs['ordering'] ]
            elif 's' not in self.valid_frames:
                self.valid_mask_dict['s'] = [False for f_i in inputs['ordering']]

            if frames != 's' and frames > 0:
                self.valid_mask_dict[frames] = [f_i[1] == frames if f_i[1] != 's' else False for f_i in inputs['ordering']]
                self.valid_mask[frames] = [f_i[1] == frames for f_i in inputs['ordering'] if f_i[1] != 's' and f_i[1] >= frames]                            

        if self.opt.trimin:
            # This is super ugly
            self.valid_tri_mask = {}
            self.valid_tri_mask_dict = {} 
            self.valid_tri_mask_reverse = {}
            for sets, frames in enumerate(self.opt.frame_ids[1:]):
                if sets == 0:
                    if self.opt.trimin: 
                        if ('s' in self.valid_frames or 1 in self.valid_frames or 2 in self.valid_frames):
                            self.valid_tri_mask_dict['s'] = [f_i[1] == 's' or f_i[1] == 1 or f_i[1] == 2 for f_i in inputs['ordering'] ]
                        else:
                            self.valid_tri_mask_dict['s'] = [False for f_i in inputs['ordering']]
                
                if frames != 's' and frames > 0:
                    if self.opt.trimin: 
                        self.valid_tri_mask_dict[frames] = [f_i[1] == frames or f_i[1] == frames+1 or f_i[1] == frames+2 if f_i[1] != 's' else False 
                                                            for f_i in inputs['ordering']]
                        self.valid_tri_mask[frames] = [f_i[1] == frames or f_i[1] == frames+1 or f_i[1] == frames+2 
                                                            for f_i in inputs['ordering'] if f_i[1] != 's' and f_i[1] >= frames]  
                if frames == 's' or frames > 0:
                    if self.opt.trimin: 
                        if frames != 's' and frames < 6:
                            if frames in self.valid_frames:
                                self.valid_tri_mask_reverse[f' {frames}'] = [f_i[1] == frames for f_i in inputs['ordering'] if f_i[1] != 's' and f_i[1] >= frames and f_i[1] <= frames+2]
                            if frames+1 in self.valid_frames:
                                self.valid_tri_mask_reverse[f' {frames}+{frames+1}'] = [f_i[1] == frames+1 for f_i in inputs['ordering'] if f_i[1] != 's' and f_i[1] >= frames and f_i[1] <= frames+2]
                            if frames+2 in self.valid_frames:
                                self.valid_tri_mask_reverse[f' {frames}+{frames+2}'] = [f_i[1] == frames+2 for f_i in inputs['ordering'] if f_i[1] != 's' and f_i[1] >= frames and f_i[1] <= frames+2]
                            
                            if frames in self.valid_frames:
                                self.valid_tri_mask_reverse[f'{-frames}'] = self.valid_tri_mask_reverse[f' {frames}']
                            if frames+1 in self.valid_frames:
                                self.valid_tri_mask_reverse[f'{-frames}+{frames+1}'] = self.valid_tri_mask_reverse[f' {frames}+{frames+1}']
                            if frames+2 in self.valid_frames:
                                self.valid_tri_mask_reverse[f'{-frames}+{frames+2}'] = self.valid_tri_mask_reverse[f' {frames}+{frames+2}']

                        elif frames != 's' and frames == 6:
                            if frames in self.valid_frames:
                                self.valid_tri_mask_reverse[f' {frames}'] = [f_i[1] == frames for f_i in inputs['ordering'] if f_i[1] != 's' and f_i[1] >= frames]
                            if frames+1 in self.valid_frames:
                                self.valid_tri_mask_reverse[f' {frames}+{frames+1}'] = [f_i[1] == frames+1 for f_i in inputs['ordering'] if f_i[1] != 's' and f_i[1] >= frames]

                            if frames in self.valid_frames:
                                self.valid_tri_mask_reverse[f'{-frames}'] = self.valid_tri_mask_reverse[f' {frames}']
                            if frames+1 in self.valid_frames:
                                self.valid_tri_mask_reverse[f'{-frames}+{frames+1}'] = self.valid_tri_mask_reverse[f' {frames}+{frames+1}']
                        elif frames != 's' and frames == 7:
                            self.valid_tri_mask_reverse[f' {frames}'] = [True for f_i in inputs['ordering'] if f_i[1] == frames]
                            self.valid_tri_mask_reverse[f'{-frames}'] = self.valid_tri_mask_reverse[f' {frames}']

                        elif frames == 's':
                            if 's' in self.valid_frames:
                                self.valid_tri_mask_reverse[f'{frames}'] = [f_i[1] == 's' for f_i in inputs['ordering'] if f_i[1] =='s' or f_i[1] <= 2]
                            if 1 in self.valid_frames:
                                self.valid_tri_mask_reverse[frames+'+1'] = [f_i[1] == 1 for f_i in inputs['ordering'] if f_i[1] =='s' or f_i[1] <= 2]
                            if 2 in self.valid_frames:
                                self.valid_tri_mask_reverse[frames+'+2'] = [f_i[1] == 2 for f_i in inputs['ordering'] if f_i[1] =='s' or f_i[1] <= 2]

            if self.opt.trimin: 
                if (1 in self.valid_frames or 2 in self.valid_frames) and ('s' not in self.valid_frames):
                    self.valid_frames.append('s')
                if (2 in self.valid_frames or 3 in self.valid_frames) and (1 not in self.valid_frames):
                    self.valid_frames.append(1)
                    self.valid_frames.append(-1)
                if (3 in self.valid_frames or 4 in self.valid_frames) and (2 not in self.valid_frames):
                    self.valid_frames.append(2)
                    self.valid_frames.append(-2)
                if (4 in self.valid_frames or 5 in self.valid_frames) and (3 not in self.valid_frames):
                    self.valid_frames.append(3)
                    self.valid_frames.append(-3)
                if (5 in self.valid_frames or 6 in self.valid_frames) and (4 not in self.valid_frames):
                    self.valid_frames.append(4)
                    self.valid_frames.append(-4)
                if (6 in self.valid_frames or 7 in self.valid_frames) and (5 not in self.valid_frames):
                    self.valid_frames.append(5)
                    self.valid_frames.append(-5)
                if (7 in self.valid_frames) and (6 not in self.valid_frames):
                    self.valid_frames.append(6)
                    self.valid_frames.append(-6)

    def x_min_opt(self, temp_positive, reprojection_losses, identity_reprojection_losses, noise, reprojection_losses_D, to_cat, dictor_norm, dictor_guide):
        if self.opt.decomp:
            for frame_id in temp_positive:
                if frame_id == 's':
                    catter, ident = torch.min(torch.cat([reprojection_losses[frame_id],
                                                    identity_reprojection_losses[frame_id]+noise[frame_id]], dim=1), dim=1)
                    to_cat.append(catter)
                    dictor_norm[(frame_id, 'norm')].append(ident==0)

                elif frame_id == 1:
                    catter, ident = torch.min(torch.cat([reprojection_losses[f'{frame_id}'],
                                                    reprojection_losses[f'{-frame_id}'],
                                                    reprojection_losses[f's+{frame_id}'],
                                                    reprojection_losses_D[f'{frame_id}'],
                                                    reprojection_losses_D[f'{-frame_id}'],
                                                    identity_reprojection_losses[f'{frame_id}']+noise[frame_id],
                                                    identity_reprojection_losses[f'{-frame_id}']+noise[frame_id],
                                                    identity_reprojection_losses[f's+{frame_id}']+noise[frame_id]], dim=1), dim=1)
                    to_cat.append(catter)
                    dictor_norm[(frame_id, 'norm')].append((ident==0) + (ident==1) + (ident==2))
                    dictor_guide[(frame_id, 'guide')].append((ident==3) + (ident==4))

                elif frame_id == 2:
                    catter, ident = torch.min(torch.cat([reprojection_losses[f'{frame_id}'],
                                                    reprojection_losses[f'{-frame_id}'],
                                                    reprojection_losses[f'{frame_id-1}+{frame_id}'],
                                                    reprojection_losses[f'{-frame_id+1}+{frame_id}'],
                                                    reprojection_losses[f's+{frame_id}'],
                                                    reprojection_losses_D[f'{frame_id}'],
                                                    reprojection_losses_D[f'{-frame_id}'],
                                                    reprojection_losses_D[f'{frame_id-1}+{frame_id}'],
                                                    reprojection_losses_D[f'{-frame_id+1}+{frame_id}'],
                                                    identity_reprojection_losses[f'{frame_id}']+noise[frame_id],
                                                    identity_reprojection_losses[f'{-frame_id}']+noise[frame_id],
                                                    identity_reprojection_losses[f'{frame_id-1}+{frame_id}']+noise[frame_id],
                                                    identity_reprojection_losses[f'{-frame_id+1}+{frame_id}']+noise[frame_id],
                                                    identity_reprojection_losses[f's+{frame_id}']+noise[frame_id]], dim=1), dim=1)
                    to_cat.append(catter)
                    dictor_norm[(frame_id, 'norm')].append((ident==0) + (ident==1) + (ident==2) + (ident==3) + (ident==4))
                    dictor_guide[(frame_id, 'guide')].append((ident==5) + (ident==6) + (ident==7) + (ident==8))

                elif frame_id >= 3:
                    catter, ident = torch.min(torch.cat([reprojection_losses[f'{frame_id}'],
                                                    reprojection_losses[f'{-frame_id}'],
                                                    reprojection_losses[f'{frame_id-1}+{frame_id}'],
                                                    reprojection_losses[f'{-frame_id+1}+{frame_id}'],
                                                    reprojection_losses[f'{frame_id-2}+{frame_id}'],
                                                    reprojection_losses[f'{-frame_id+2}+{frame_id}'],
                                                    reprojection_losses_D[f'{frame_id}'],
                                                    reprojection_losses_D[f'{-frame_id}'],
                                                    reprojection_losses_D[f'{frame_id-1}+{frame_id}'],
                                                    reprojection_losses_D[f'{-frame_id+1}+{frame_id}'],
                                                    reprojection_losses_D[f'{frame_id-2}+{frame_id}'],
                                                    reprojection_losses_D[f'{-frame_id+2}+{frame_id}'],
                                                    identity_reprojection_losses[f'{frame_id}']+noise[frame_id],
                                                    identity_reprojection_losses[f'{-frame_id}']+noise[frame_id],
                                                    identity_reprojection_losses[f'{frame_id-1}+{frame_id}']+noise[frame_id],
                                                    identity_reprojection_losses[f'{-frame_id+1}+{frame_id}']+noise[frame_id],
                                                    identity_reprojection_losses[f'{frame_id-2}+{frame_id}']+noise[frame_id],
                                                    identity_reprojection_losses[f'{-frame_id+2}+{frame_id}']+noise[frame_id]], dim=1), dim=1)
                    to_cat.append(catter)
                    dictor_norm[(frame_id, 'norm')].append((ident==0) + (ident==1) + (ident==2) + (ident==3) + (ident==4)  + (ident==5))
                    dictor_guide[(frame_id, 'guide')].append((ident==6) + (ident==7) + (ident==8) + (ident==9)+ (ident==10)+ (ident==11))

            to_optimise = torch.cat(to_cat, dim=0)
            return to_optimise, (dictor_norm, dictor_guide)
        else: 
            for frame_id in temp_positive:
                if frame_id == 's':
                    catter, ident = torch.min(torch.cat([reprojection_losses[frame_id],
                                                    identity_reprojection_losses[frame_id]+noise[frame_id]], dim=1), dim=1)
                    to_cat.append(catter)
                    dictor_norm[(frame_id, 'norm')].append(ident==0)

                elif frame_id == 1:
                    catter, ident = torch.min(torch.cat([reprojection_losses[f'{frame_id}'],
                                                    reprojection_losses[f'{-frame_id}'],
                                                    reprojection_losses[f's+{frame_id}'],
                                                    identity_reprojection_losses[f'{frame_id}']+noise[frame_id],
                                                    identity_reprojection_losses[f'{-frame_id}']+noise[frame_id],
                                                    identity_reprojection_losses[f's+{frame_id}']+noise[frame_id]], dim=1), dim=1)
                    to_cat.append(catter)
                    dictor_norm[(frame_id, 'norm')].append((ident==0) + (ident==1) + (ident==2))

                elif frame_id == 2:
                    catter, ident = torch.min(torch.cat([reprojection_losses[f'{frame_id}'],
                                                    reprojection_losses[f'{-frame_id}'],
                                                    reprojection_losses[f'{frame_id-1}+{frame_id}'],
                                                    reprojection_losses[f'{-frame_id+1}+{frame_id}'],
                                                    reprojection_losses[f's+{frame_id}'],
                                                    identity_reprojection_losses[f'{frame_id}']+noise[frame_id],
                                                    identity_reprojection_losses[f'{-frame_id}']+noise[frame_id],
                                                    identity_reprojection_losses[f'{frame_id-1}+{frame_id}']+noise[frame_id],
                                                    identity_reprojection_losses[f'{-frame_id+1}+{frame_id}']+noise[frame_id],
                                                    identity_reprojection_losses[f's+{frame_id}']+noise[frame_id]], dim=1), dim=1)

                    to_cat.append(catter)
                    dictor_norm[(frame_id, 'norm')].append((ident==0) + (ident==1) + (ident==2) + (ident==3) + (ident==4))

                elif frame_id >= 3:
                    catter, ident = torch.min(torch.cat([reprojection_losses[f'{frame_id}'],
                                                    reprojection_losses[f'{-frame_id}'],
                                                    reprojection_losses[f'{frame_id-1}+{frame_id}'],
                                                    reprojection_losses[f'{-frame_id+1}+{frame_id}'],
                                                    reprojection_losses[f'{frame_id-2}+{frame_id}'],
                                                    reprojection_losses[f'{-frame_id+2}+{frame_id}'],
                                                    identity_reprojection_losses[f'{frame_id}']+noise[frame_id],
                                                    identity_reprojection_losses[f'{-frame_id}']+noise[frame_id],
                                                    identity_reprojection_losses[f'{frame_id-1}+{frame_id}']+noise[frame_id],
                                                    identity_reprojection_losses[f'{-frame_id+1}+{frame_id}']+noise[frame_id],
                                                    identity_reprojection_losses[f'{frame_id-2}+{frame_id}']+noise[frame_id],
                                                    identity_reprojection_losses[f'{-frame_id+2}+{frame_id}']+noise[frame_id]], dim=1), dim=1)

                    to_cat.append(catter)
                    dictor_norm[(frame_id, 'norm')].append((ident==0) + (ident==1) + (ident==2) + (ident==3) + (ident==4)+ (ident==5))

            to_optimise = torch.cat(to_cat, dim=0)
            return to_optimise, (dictor_norm, dictor_guide)

def colormap(inputs, color='plasma', normalize=True, torch_transpose=True):
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
        else:
            vis = _BRIGHT_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        if color == 'plasma':
            vis = _DEPTH_COLORMAP(vis)
        else:
            vis = _BRIGHT_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        if color == 'plasma':
            vis = _DEPTH_COLORMAP(vis)
        else:
            vis = _BRIGHT_COLORMAP(vis)

        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis
