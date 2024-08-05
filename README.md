# BaseBoostDepth: Exploiting Larger Baselines for Self-Supervised Monocular Depth Estimation 

[![Website](assets/badge-website.svg)](https://kieran514.github.io/BaseBoostDepth-Project/)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/pdf/2407.20437)
[![Paper](assets/youtube.svg)](https://youtu.be/QbDem7nnbbY)


<p align="center">
  <img src="assets/teaser2.gif" alt="High edge accuracy" width="1200" />
</p>


## Installation Setup

The models were trained using CUDA 11.1, Python 3.7.4 (conda environment), PyTorch 1.8.0 and Ubuntu 22.04.

Create a conda environment with the PyTorch library:

```bash
conda env create --file environment.yml
conda activate baseboostdepth
pip install git+'https://github.com/otaheri/chamfer_distance'
```

## Training Datasets

We use the [KITTI dataset](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) and follow the downloading/preprocessing instructions set out by [Monodepth2](https://github.com/nianticlabs/monodepth2).
Download from scripts;
```
wget -i scripts/kitti_archives_to_download.txt -P data/KITTI_RAW/
```
Then unzip the downloaded files;
```
cd data/KITTI_RAW
unzip "*.zip"
cd ..
cd ..
```
Then convert all images to jpg;
```
find data/KITTI_RAW/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```

## Pretrained Models
KITTI: (Abs_Rel, RMSE, a1)

SYNS: (Edge-Acc, Edge-Comp, Point cloud F-Score, Point cloud IoU)

| Model Name | Abs_Rel | RMSE | a1 | Edge-Acc | Edge-Comp | F-Score | IoU | Model resolution | Model |
|------------|---------|------|----|----------|-----------|---------------------|-----------------|------------------|-------|
| [BaseBoostDepth](https://drive.google.com/drive/folders/1k3MbmnX3L8zjZTOpi5oki3IKoyRAKWP8?usp=sharing) | 0.106 | 4.584 | 0.883 | 2.453 | 3.810 | 0.275 | 0.174 | 640 x 192 | MD2 |
| [BaseBoostDepth (pre)](https://drive.google.com/drive/folders/1ay9yLr8R4gHBffUSVJA2C_FVulNKvpaN?usp=sharing) | 0.104 | 4.544 | 0.888 | 2.432 | 4.763 | 0.268 | 0.168 | 640 x 192 | MD2 |
| [BaseBoostDepth (pre MonoViT)](https://drive.google.com/drive/folders/1x_VnZsmFy7qI2LknkzCwCUrYMojsfkqo?usp=sharing) | 0.096 | 4.201 | 0.906 | 2.409 | 5.314 | 0.300 | 0.191 | 640 x 192 | MonoViT |
| [BaseBoostDepth (pre SQLdepth)](https://drive.google.com/drive/folders/1LpYhn4mMpJt-TGrqqt_IkYPdgGT2sGA5?usp=sharing) | 0.084 | 3.980 | 0.920 | 2.505 | 13.164 | 0.246 | 0.151 | 640 x 192 | SQLdepth |

## Training

### Prepare Validation Data
```
python export_gt_depth.py --data_path data/KITTI_RAW --split eigen_zhou
```

```
bash run.sh
```


## KITTI Ground Truth 

We must prepare ground truth files for testing/validation and training.
```
python export_gt_depth.py --data_path data/KITTI_RAW --split eigen
python export_gt_depth.py --data_path data/KITTI_RAW --split eigen_benchmark
```

## Evaluation KITTI
We provide the evaluation for the KITTI dataset. If a ViT model is used as the weights, please use --ViT when evaluating below.

#### KITTI 

```
python evaluate_depth.py --load_weights_folder {weights_directory} --eval_mono --kt_path data/KITTI_RAW --eval_split eigen
```
```
python evaluate_depth.py --load_weights_folder {weights_directoryMonoViT} --eval_mono --kt_path data/KITTI_RAW --eval_split eigen --ViT
```
```
python evaluate_depth.py --load_weights_folder {weights_directorySQL} --eval_mono --kt_path data/KITTI_RAW --eval_split eigen --SQL
```


#### KITTI Benchmark

```
python evaluate_depth.py --load_weights_folder {weights_directory} --eval_mono --kt_path data/KITTI_RAW --eval_split eigen_benchmark
```
```
python evaluate_depth.py --load_weights_folder {weights_directoryMonoViT} --eval_mono --kt_path data/KITTI_RAW --eval_split eigen_benchmark --ViT
```
```
python evaluate_depth.py --load_weights_folder {weights_directorySQL} --eval_mono --kt_path data/KITTI_RAW --eval_split eigen_benchmark --SQL
```

## SYNS Dataset Creation
Due August

## References

* [Monodepth2](https://github.com/nianticlabs/monodepth2) (ICCV 2019)
* [MonoViT](https://github.com/zxcqlf/MonoViT) 
* [SQLdepth](https://github.com/hisfog/SfMNeXt-Impl) 

