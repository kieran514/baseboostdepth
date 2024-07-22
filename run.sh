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
--batch_size 12 \
--log_frequency 1700 \
--learning_rate 1e-4 \
--num_epochs 20 \
--num_workers 3 \
--kt \
--rand \
--debug \
--trimin \
--incremental_skip \
--naive_mix \
--partial_skip --decomp --pose_error 5.5 \
--training_file train_files_baselines \
--weights_init pretrained \
--load_weights_folder None

# To train with MonoViT use --ViT
# To train with SQLdepth use --SQL
# if starting from pretrained just change 
# "--load_weights_folder None" to the network to pretrain from 
