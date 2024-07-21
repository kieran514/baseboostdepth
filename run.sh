python train.py \
--model_name Zeus \
--kt_path /media/kieran/Extreme_SSD/data/KITTI_RAW \
--syns_path /media/kieran/Extreme_SSD/data/SYNS/SYNS \
--eval_split eigen \
--split eigen_zhou \
--height 192 \
--width 640 \
--cuda 0 \
--disparity_smoothness 0.001 \
--batch_size 4 \
--log_frequency 500 \
--learning_rate 1e-4 \
--num_epochs 30 \
--num_workers 3 \
--kt \
--rand \
--debug \
--trimin \
--incremental_skip \
--naive_mix \
--partial_skip --decomp --pose_error 5.5 \
--correct_pose \
--training_file train_files_baselines \
--weights_init pretrained \
--load_weights_folder /media/kieran/Extreme_SSD/BaseBoostDepthALL/BaseBoostDepthcomplete/paper/mono+stereo_pre

# /media/kieran/Extreme_SSD/ZeusN/paper/SQL
# --partial_skip \
# 
# --scales 0 \
# --x_min \
# --x_val 2 \
# --back_pose_error \
# --use_ident \
# --batch_size_increase
# --partial_skip
# --stereo_scaler
# --inversion \
# --pose_error 1.6 \
# --rand \
# --incremental_skip \
# --res \
# --trimin
# --use_bright \
# --debug
# --zigzag \
# --cl_weight 0.5 \
# /media/kieran/Extreme_SSD/Zeus/paper/mono+stereo_best
# intrinsics wont wrok for the last epoch drop_last false
# --intrin \
# --weighted \
# --depth_hint 
# --kt \
# --hold #
# --intrin_sup \
# --naive_mix \
# --mix \
# --aps \
# --mix \
# --cs --kt --ad --fd --ox --aps \
# --frame_ids 0 -1 1 \
