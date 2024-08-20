#!/bin/bash
#SBATCH --export=/usr/local/cuda/bin
#SBATCH --gres=gpu:1

module load cuda/12.2

make test

# Loop through directories 0 to 20
for dir in $(seq 0 20)
do
    echo "Running test for directory $dir"
    ./build/main_test -e ../data/$dir/output.raw -i ../data/$dir/input.raw,../data/$dir/kernel.raw -t matrix
done

module unload cuda/12.2
