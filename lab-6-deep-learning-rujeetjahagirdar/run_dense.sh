#!/bin/bash
#SBATCH --export=/usr/local/cuda/bin
#SBATCH --gres=gpu:1

module load cuda/12.2

make

./bin/dense-neuralnet

module unload cuda/12.2
