#!/bin/bash

# Download pre-trained models
echo "Downloading pre-trained models"
wget https://filebox.ece.vt.edu/~ylzou/eccv2018dfnet/pretrained.tar
tar -xvf pretrained.tar

# Download training data
echo "Downloading training data"
mkdir dataset
cd dataset
wget https://filebox.ece.vt.edu/~ylzou/eccv2018dfnet/kitti_5frame_1152_320.tar
tar -xvf kitti_5frame_1152_320.tar
cd ..

