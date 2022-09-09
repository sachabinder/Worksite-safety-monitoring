#!/bin/bash
conda create -n wsm_env python=3.10 -y
conda activate wsm_env

# TO BE CHANGE WITH YOUR CONFIG
pip install torch torchvision torchaudio -y
pip install opencv-python -y
pip install matplotlib -y
pip install pillow -y


