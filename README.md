# DF-Net: Unsupervised Joint Learning of Depth and Flow using Cross-Task Consistency

A TensorFlow re-implementation for [DF-Net: Unsupervised Joint Learning of Depth and Flow using Cross-Task Consistency](https://arxiv.org/abs/1809.01649). There are some minor differences from the model described in the paper:

- Model in the paper uses 2-frame as input, while this code uses 5-frame as input (you might use any odd numbers of frames as input, though you would need to tune the hyper-parameters)
- FlowNet in the paper is pre-trained on SYNTHIA, while this one is pre-trained on Cityscapes

Please see the [project page](http://yuliang.vision/DF-Net/) for more details. 

<img src="misc/zou2018dfnet.gif">


## Prerequisites
This codebase was developed and tested with the following settings:
```
Python 3.6
TensorFlow 1.2.0 (this is the only supported version)
g++ 4.x (this is the only supported version)
CUDA 8.0
Unbuntu 14.04
4 Tesla K80 GPUs (w/ 12G memory each)
```

Some Python packages you might not have
```
pypng
opencv-python
```

## Installation
1. Clone this repository
```Shell
git clone git@github.com:vt-vl-lab/DF-Net.git
cd DF-Net
```

2. Prepare models and training data
```Shell
chmod +x ./misc/prepare.sh
./misc/prepare.sh
```
NOTE: Frames belonging to KITTI2012/2015 train/test scenes have been excluded in the provided training set. Add these frames back to the training set would improve the performance of DepthNet.

## Data preparation (for evaluation)
After accepting their license conditions, download [KITTI raw](http://www.cvlibs.net/datasets/kitti/raw_data.php), [KITTI flow 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow), [KITTI flow 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow).

Then you can make soft-link for them
```Shell
cd dataset
mkdir KITTI
cd KITTI

ln -s /path/to/KITTI/raw raw
ln -s /path/to/KITTI/2012 flow2012
ln -s /path/to/KITTI/2015 flow2015
```

**(Optional)** You can add those KITTI2012/2015 frames back to the training set, by commenting line81~line85 in `data/kitti/kitti_raw_loader.py`, and do
```
python data/prepare_train_data.py --dataset_name='kitti_raw_eigen' --dump_root=/path/to/save/ --num_threads=4
```

## Training
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train_df.py --dataset_dir=/path/to/your/data --checkpoint_dir=/path/to/save/your/model
```

For the first time, custom CUDA operations for FlowNet will be compiled. If you have any compliation issues, please check `core/UnFlow/src/e2eflow/ops.py` 
- Line31: specify your CUDA path
- Line32: Add `-I $CUDA_HOME/include`, where `$CUDA_HOME` is your CUDA directory
- Line38: specify your g++ version

## Testing
Test DepthNet on KITTI raw (You can use the validation set to selct the best model.)
```
python test_kitti_depth.py --dataset_dir=/path/to/your/data --output_dir=/path/to/save/your/prediction --ckpt_file=/path/to/your/ckpt --split="val or test"
python kitti_eval/eval_depth.py --pred_file=/path/to/your/prediction --split="val or test"
```

Test FlowNet on KITTI 2012 (Please use training set)
```
python test_flownet_2012.py --dataset_dir=/path/to/your/data --ckpt_file=/path/to/your/ckpt
```

Test FlowNet on KITTI 2015 (Please use training set)
```
python test_flownet_2015.py --dataset_dir=/path/to/your/data --ckpt_file=/path/to/your/ckpt
```

NOTE: For KITTI 2012/2015
- If you want to generate visualization colormap for **training set**, you can specify `output_dir`
- If you want to test on **test set** and upload it to KITTI server, you can specify `output_dir` and test on test set.

## Pre-trained model performance
You should get the following numbers if you use the pre-trained model `pretrained/dfnet`


DepthNet (KITTI raw test test)

abs rel | sq rel | rms | log rms | a1 | a2 | a3 |
---------------|------------|------------|------------|------------|------------|------------|
0.1452 | 1.2904 | 5.6115 | 0.2194 | 0.8114 | 0.9394 | 0.9767 |


FlowNet (KITTI 2012/2015 training set)

KITTI 2012 EPE | KITTI 2015 EPE | KITTI 2015 F1 | 
---------------|----------------|---------------|
3.1052 | 7.4482 | 0.2695 |


## Citation
If you find this code useful for your research, please consider citing the following paper:

    @inproceedings{zou2018dfnet,
    author    = {Zou, Yuliang and Luo, Zelun and Huang, Jia-Bin}, 
    title     = {DF-Net: Unsupervised Joint Learning of Depth and Flow using Cross-Task Consistency}, 
    booktitle = {European Conference on Computer Vision},
    year      = {2018}
    }


## Acknowledgement
Codes are heavily borrowed from several great work, including [SfMLearner](https://github.com/tinghuiz/SfMLearner), [monodepth](https://github.com/mrharicot/monodepth), and [UnFlow](https://github.com/simonmeister/UnFlow). We thank [Shih-Yang Su](https://github.com/LemonATsu) for the code review.
