#!/bin/bash
#SBATCH --export=/usr/local/cuda/bin
#SBATCH --gres=gpu:1

module load cuda/12.2

make

./build/grad_image img/gpgpu.png

module unload cuda/12.2
