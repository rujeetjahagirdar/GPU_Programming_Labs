#!/bin/bash
#SBATCH --export=/usr/local/cuda/bin
#SBATCH --gres=gpu:1

make datagen
./build/datagen

make test

# Loop through directories 0 to 9
#for dir in $(0..9)
for dir in `seq 0 9`
do
    echo "Running test for directory $dir"
    ./build/main_test -e data/$dir/output.pbm -i data/$dir/input.ppm -t image
done
