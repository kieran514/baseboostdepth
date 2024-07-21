
from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="BBD options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data"))
        self.parser.add_argument("--syns_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data"))
        self.parser.add_argument("--kt_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data"))
        self.parser.add_argument("--SYNS_eval",
                                 help="if set, use front reconstrctuions",
                                 action="store_true")
        self.parser.add_argument("--SQL",
                                 help="if set, use front reconstrctuions",
                                 action="store_true")
        self.parser.add_argument("--no_adam",
                                 help="if set, use front reconstrctuions",
                                 action="store_true")
        self.parser.add_argument("--SQL_L",
                                 help="if set, use front reconstrctuions",
                                 action="store_true")
        self.parser.add_argument("--CA_depth",
                                 help="if set, use front reconstrctuions",
                                 action="store_true")
        self.parser.add_argument("--DIFFNet",
                                 help="if set, use front reconstrctuions",
                                 action="store_true")
        self.parser.add_argument("--rand",
                                 help="if set, use front reconstrctuions",
                                 action="store_true")
        self.parser.add_argument("--reg_new",
                                 help="if set, use new regularizer",
                                 action="store_true")
        self.parser.add_argument("--correct_pose",
                                 help="if set, use new regularizer",
                                 action="store_true")
        self.parser.add_argument("--chamfer",
                                 help="if set, use new regularizer",
                                 action="store_true")
        self.parser.add_argument("--use_bright",
                                 help="if set, use front reconstrctuions",
                                 action="store_true")
        self.parser.add_argument("--adaptive_pose_error",
                                 help="if set, use front reconstrctuions",
                                 action="store_true")
        self.parser.add_argument("--debug",
                                 help="if set, use front reconstrctuions",
                                 action="store_true")
        self.parser.add_argument("--stereo_guide",
                                 help="if set, use front reconstrctuions",
                                 action="store_true")
        self.parser.add_argument("--trimin",
                                 help="if set, use front reconstrctuions",
                                 action="store_true")
        self.parser.add_argument("--guidance",
                                 help="if set, use front reconstrctuions",
                                 action="store_true")
        self.parser.add_argument("--pose_consistency",
                                 help="if set, use front reconstrctuions",
                                 action="store_true")
        self.parser.add_argument("--distributed",
                                 help="if set, use front reconstrctuions",
                                 action="store_true")
        
        self.parser.add_argument("--x_min",
                                 help="if set, use front reconstrctuions",
                                 action="store_true")
        self.parser.add_argument("--x_val",
                                 default=3,
                                 type=int)

        self.parser.add_argument("--decomp",
                                 help="if set, use front reconstrctuions",
                                 action="store_true")
        self.parser.add_argument("--stereo_scaler",
                                 help="if set, use front reconstrctuions",
                                 action="store_true")
        self.parser.add_argument("--batch_size_increase",
                                 help="if set, use front reconstrctuions",
                                 action="store_true")
        self.parser.add_argument("--decomp_weight",
                                 default=0.1,
                                 type=float)
        self.parser.add_argument("--inversion",
                    help="if set, use front reconstrctuions",
                    action="store_true")
        self.parser.add_argument("--partial_skip",
                    help="if set, use front reconstrctuions",
                    action="store_true")

        self.parser.add_argument("--high_depth",
                              help="if set, use front reconstrctuions",
                              action="store_true")
        self.parser.add_argument("--pose_error",
                                 default=1,
                                 type=float)
        self.parser.add_argument("--sobel_weight",
                                 default=1,
                                 type=float)
        self.parser.add_argument("--back_pose_error",
                              help="if set, use front reconstrctuions",
                              action="store_true")
        self.parser.add_argument("--use_ident",
                              help="if set, use front reconstrctuions",
                              action="store_true")

        self.parser.add_argument("--view",
                                 type=int,
                                 help="number of cuda",
                                 default=0,
                                 choices=[36,50,90,150])

        self.parser.add_argument("--training_file",
                                 type=str,
                                 help="train_file")

        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(file_dir, "paper"))
        self.parser.add_argument("--cuda",
                                 type=int,
                                 help="number of cuda",
                                 default=0,
                                 choices=[0,1,2,3,4,5,6,7,8,9,10])
        self.parser.add_argument("--ViT",
                                 help="if set, use monovitt depth netwrok",
                                 action="store_true")
        self.parser.add_argument("--weighted",
                                 help="if set, use weighted",
                                 action="store_true")
        self.parser.add_argument("--cl",
                                 help="if set, use weighted",
                                 action="store_true")
        self.parser.add_argument("--depth_hint",
                                 help="if set, use hint depth",
                                 action="store_true")
        self.parser.add_argument("--intrin",
                                 help="learn intrinsics",
                                 action="store_true")
        self.parser.add_argument("--intrin_sup",
                                 help="learn intrinsics",
                                 action="store_true")
        self.parser.add_argument("--save_imgs",
                                 help="if set, save images in test",
                                 action="store_true")
        self.parser.add_argument("--pose_loss_addition",
                                 help="the pose is supervised by the addiion",
                                 action="store_true")
        self.parser.add_argument("--pose_addition",
                                 help="we use pose addition as out main",
                                 action="store_true")
        self.parser.add_argument("--trimean",
                                 help="we use pose addition as out main",
                                 action="store_true")
        self.parser.add_argument("--winsorized_mean",
                                 help="we use pose addition as out main",
                                 action="store_true")
        self.parser.add_argument("--occ_mean",
                                 help="we use pose addition as out main",
                                 action="store_true")
        self.parser.add_argument("--further_loss",
                                 help="we use pose addition as out main",
                                 action="store_true")
        self.parser.add_argument("--res",
                                 help="we use pose addition as out main",
                                 action="store_true")
        self.parser.add_argument("--rand_frame",
                                 help="we use pose addition as out main",
                                 action="store_true")
        self.parser.add_argument("--mean",
                                 help="we use pose addition as out main",
                                 action="store_true")
        self.parser.add_argument("--median",
                                 help="we use pose addition as out main",
                                 action="store_true")
        
        self.parser.add_argument("--skipper_new",
                                 type=int,
                                 help="how many frames we skip into the future",
                                 default=1)
        self.parser.add_argument("--extra_occ",
                                 help="we use pose addition as out main",
                                 action="store_true")

        # TRAINING options
        self.parser.add_argument("--skipping",
                                 help="skip files",
                                 action="store_true")
        self.parser.add_argument("--sandmoptim",
                                 help="skip files",
                                 action="store_true")
        self.parser.add_argument("--incremental_skip",
                                 help="skip files",
                                 action="store_true")
        self.parser.add_argument("--zigzag",
                                 help="zigzag files",
                                 action="store_true")

        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
                                 default="eigen_zhou")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--pytorch_random_seed",
                                 default=42,
                                 type=int)
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test", "mixed"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--hold",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--mix",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
        self.parser.add_argument("--no_guide",
                                 nargs="+",
                                 type=int,
                                 help="no guidance from stereo",
                                 default=[0, 1])

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        
        self.parser.add_argument("--naive_mix",
                                 help="mixing stratergy",
                                 action="store_true")
        

        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10","odom_2", "odom_0", "odom_1","odom_3", "odom_4", "odom_5","odom_6", "odom_7","odom_8" 'Cityscape', 'DSEC', 'Oxford', 'Ford', 'ApolloScape', 'SYNS'],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")
        
        ########################
        self.parser.add_argument("--kt",
                                 help="loads the specified dataset",
                                 action="store_true")
        self.parser.add_argument("--cs",
                                 help="loads the specified dataset",
                                 action="store_true")
        self.parser.add_argument("--ox",
                                 help="loads the specified dataset",
                                 action="store_true")
        self.parser.add_argument("--ad",
                                 help="loads the specified dataset",
                                 action="store_true")
        self.parser.add_argument("--fd",
                                 help="loads the specified dataset",
                                 action="store_true")
        self.parser.add_argument("--aps",
                                 help="loads the specified dataset",
                                 action="store_true")
        self.parser.add_argument("--ds",
                                 help="loads the specified dataset",
                                 action="store_true")
        self.parser.add_argument("--ddad",
                                 help="loads the specified dataset",
                                 action="store_true")
        self.parser.add_argument("--mal",
                                 help="loads the specified dataset",
                                 action="store_true")
        self.parser.add_argument("--hol",
                                 help="loads the specified dataset",
                                 action="store_true")
        self.parser.add_argument("--ms",
                                 help="loads the specified dataset",
                                 action="store_true")
        self.parser.add_argument("--fov",
                                 help="loads the specified dataset",
                                 action="store_true")
        self.parser.add_argument("--gb",
                                 help="loads the specified dataset",
                                 action="store_true")

        self.parser.add_argument("--sim",
                                 help="loads the specified dataset",
                                 action="store_true")
        self.parser.add_argument("--au",
                                 help="loads the specified dataset",
                                 action="store_true")
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
