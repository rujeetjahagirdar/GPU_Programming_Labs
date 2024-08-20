# CSE 4373/5373 Labl 2

Covers tiling, shared memory, and benchmarking.

## Part 1: Tiled Matrix Multiplication

For part 1, you will implement a tiled matrix multiplication kernel. Tiled matrix multiplication is a technique that reduces the number of global memory accesses a kernel makes by storing a tile of the input matrices in shared memory. More information can be found [in the notes](https://ajdillhoff.github.io/notes/cuda_memory_architecture/#tiling).

### Implementation

An empty file is already provided for you in `tiled_matmul.cu`. Most of the implementation can be copied from the notes, but you will need to double check that it handles non-square matrices correctly.

### Testing

As with the previous lab, you can compile the code with the provided `Makefile` by running `make test` in the terminal. Be sure to generate the test data first. If you're running this in the lab, simply run `sbatch run_tests.sh` to generate the test data and run the tests.

## Part 2: Adding Thread Coarsening

[Thread coarsening](https://ajdillhoff.github.io/notes/gpu_performance_basics/#thread-coarsening) can improve performance for memory-bound kernels by having each thread perform more work. This can be useful for kernels that have a high arithmetic intensity, but are limited by memory bandwidth. Matrix multiplication is a kernel that should have high arithmetic intensity.

Completing this part only requires that you implement the kernel as discussed in the lecture notes and pass the provided tests. That kernel should see slightly improved performance over tiled matrix multiplication due to reduced overhead from launching threads /and/ improved memory access patterns.

You could also attempt thread coarsening by having each thread perform more arithmetic operations. That is, each thread will compute more than one element of the output matrix. This will require more changes to the kernel, but could potentially improve performance even more. We will review such an implementation in class.

### Implementation

You will need to modify the tiled matrix multiplication kernel to use coarsening. A separate folder is available for this part. Implement your kernel in `coarse_matmul.cu`. The implementation is similar to the tiled matrix multiplication kernel, but will require some additional changes to the kernel. Feel free to use the notes as a reference.

### Testing

Compile the code with the provided `Makefile` by running `make test` in the terminal. Be sure to generate the test data first. If you're running this in the lab, simply run `sbatch run_tests.sh` to generate the test data and run the tests. Once you've passed all the tests, your kernel is ready for benchmarking.

## Part 3: Benchmarking

Once all matrix multiplication kernels have been added and tested, you can benchmark the performance of each kernel using NSight Compute. The results of this benchmarking will help you to answer the questions located in `Questions.md`. A benchmarking script is provided for you in `benchmark.sh`. You can run the script by executing `sbatch benchmark.sh` in the terminal. This will generate a `.ncu-rep` file that you can open in NSight Compute to analyze the performance of your kernels. It may take a while to run as it will test each kernel using a large input size.

Once the run is complete, load the `.ncu-rep` file into NSight Compute and analyze the performance of each kernel. You will need to answer the questions in `Questions.md` based on the results of the benchmarking.