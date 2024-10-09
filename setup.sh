#!/bin/bash

conda create -n autoinst python=3.9 -y

conda activate autoinst

pip install instanseg
pip install open3d==0.17.0 opencv-python numpy==1.24.4 pykitti nptyping==2.5.0


