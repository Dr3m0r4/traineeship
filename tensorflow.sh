#!/bin/bash

wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh
rm Anaconda3-2019.03-Linux-x86_64.sh

conda create -n niftynet
conda activate niftynet

conda install tensorflow-gpu
conda install opencv
conda install scikit-image
pip install simpleitk
pip install niftynet
