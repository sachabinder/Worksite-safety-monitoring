#!/bin/bash
conda create -n wsm_env python=3.10 -y
conda activate wsm_env

# TO BE CHANGE WITH YOUR CONFIG
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
conda install -c conda-forge opencv -y
conda install -c conda-forge matplotlib -y
conda install -c anaconda pillow -y
pip install utils 


