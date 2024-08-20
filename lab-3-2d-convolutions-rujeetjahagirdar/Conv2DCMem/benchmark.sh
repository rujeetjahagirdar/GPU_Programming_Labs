#!/bin/bash
#SBATCH --export=/usr/local/cuda/bin
#SBATCH --gres=gpu:1

module load cuda/12.2

make test

nvprof --analysis-metrics -o conv2d_cmem.nvvp ./build/main_test -e data/18/output.raw -i data/18/input.raw,data/18/kernel.raw -t matrix

module unload cuda/12.2
