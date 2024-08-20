#!/bin/bash
#SBATCH --export=/usr/local/cuda/bin
#SBATCH --gres=gpu:1

# Loop through directories 0 to 8
#for dir in $(0 8)
for dir in `seq 0 8`
do
    echo "Running test for directory $dir"
    ./build/main_test -e data/$dir/output.raw -i data/$dir/input0.raw,data/$dir/input1.raw -t matrix
done
