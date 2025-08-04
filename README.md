# MonoCD: Monocular 3D Object Detection with Complementary Depths

<h5 align="center">

*Longfei Yan, Pei Yan, Shengzhou Xiong, Xuanyu Xiang, Yihua Tan*

[![arXiv](https://img.shields.io/badge/arXiv-2404.03181-b31b1b.svg)](https://arxiv.org/abs/2404.03181)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/dragonfly606/MonoCD/blob/main/LICENSE)

</h5>

This repository includes an official implementation of the paper [MonoCD: Monocular 3D Object Detection with Complementary Depths](https://arxiv.org/abs/2404.03181) based on the excellent work [MonoFlex](https://github.com/zhangyp15/MonoFlex). In this work, we first point out the coupling phenomenon that the existing multi-depth predictions have the tendency of predicted depths to consistently overestimate or underestimate the true depth values, which limits the accuracy of combined depth. We propose to increase the complementarity of depths to alleviate this problem.

![](figures/core.png)

## Installation

```bash
git clone https://github.com/dragonfly606/MonoCD.git
cd MonoCD

conda create -n monocd python=3.10 -y
conda activate monocd

# Install PyTorch that matches your local CUDA version. We adopt torch 1.4.0+cu101
pip install torch==1.13 torchvision==0.14 --index-url https://download.pytorch.org/whl/cu117

# conda install cudatoolkit ninja

# Maybe comment out Inplace-ABN and make necessary 'if' block changes to remove it from files.
pip install -r requirements.txt

# Change this as well, make necessary changes.
# cd model/backbone/DCNv2
# sh make.sh
# If the DCNv2 compilation fails, you can replace it with the version from https://github.com/lbin/DCNv2 that matches your PyTorch version, and then try recompiling.
# DCNv2 is too old for your hardware architecture
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
pip install cmake==3.26.1 pillow==8.3.2

cd ../../..
python setup.py develop

```

## Data Preparation

Please download [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize the data as follows:

```
#ROOT		
  |training/
    |calib/
    |image_2/
    |label/
    |planes/
    |ImageSets/
  |testing/
    |calib/
    |image_2/
    |ImageSets/
```

The road planes for Horizon Heatmap training could be downloaded from [HERE](https://download.openmmlab.com/mmdetection3d/data/train_planes.zip). Then remember to set the `DATA_DIR = "/path/to/your/kitti/"` in the `config/paths_catalog.py` according to your data path.

## Get Started

### Train

Training with one GPU.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --batch_size 8 --config runs/monocd.yaml --output output/exp
```

### Test

The model will be evaluated periodically during training and you can also evaluate an already trained checkpoint with

```bash
CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --config runs/monocd.yaml --ckpt YOUR_CKPT  --eval
```

### Model and log

We provide the trained model on KITTI and corresponding logs.

| Models                       | AP40@Easy | AP40@Mod. | AP40@Hard |                          Logs/Ckpts                          |
| ---------------------------- | :-------: | :-------: | :-------: | :----------------------------------------------------------: |
| MonoFlex                     |   23.64   |   17.51   |   14.83   |                              -                               |
| MonoFlex + Ours (paper)      |   24.22   |   18.27   |   15.42   |                              -                               |
| MonoFlex + Ours (reproduced) |   25.99   |   19.12   |   16.03   | [log](https://drive.google.com/file/d/1oYF4HfeZPaWiJ0IOv62UjoDkCjLtK20_/view?usp=sharing)/[ckpt](https://drive.google.com/file/d/1DbMaicafWnP-MDJAQiwnUs7QI809LbSA/view?usp=sharing) |

## Citation

If you find our work useful in your research, please consider giving us a star and citing:

```latex
@inproceedings{yan2024monocd,
  title={MonoCD: Monocular 3D Object Detection with Complementary Depths},
  author={Yan, Longfei and Yan, Pei and Xiong, Shengzhou and Xiang, Xuanyu and Tan, Yihua},
  booktitle={CVPR},
  pages={10248--10257},
  year={2024}
}
```

## Acknowledgement

This project benefits from awesome works of [MonoFlex](https://github.com/zhangyp15/MonoFlex) and [MonoGround](https://github.com/cfzd/MonoGround). Please also consider citing them.

## Contact

If you have any questions about this project, please feel free to contact longfeiyan@hust.edu.cn.