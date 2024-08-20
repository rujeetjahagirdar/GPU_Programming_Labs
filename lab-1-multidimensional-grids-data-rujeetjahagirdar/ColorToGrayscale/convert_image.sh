#!/bin/bash
#SBATCH --export=/usr/local/cuda/bin
#SBATCH --gres=gpu:1

make all

./build/main ../images/gpgpu.png
