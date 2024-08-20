# GPU_Programming_Labs
## Overview
This repository contains assignments for the GPU Programming course (CSE 5373) implemented using CUDA and C++. The assignments cover various GPU programming concepts and techniques, including multidimensional grids, tiling and thread coarsening, 2D convolutions, reductions, parallel scan, and deep learning.

## Folder Structure
- **lab-1-multidimensional-grids-data**: Implements multidimensional grid processing using CUDA.
- **lab-2-tiling-and-thread-coarsening**: Demonstrates tiling and thread coarsening techniques for optimizing CUDA programs.
- **lab-3-2d-convolutions**: Implements 2D convolution operations on GPUs.
- **lab-4-reductions**: Focuses on parallel reduction techniques for efficient computation.
- **lab-5-parallel-scan**: Implements parallel scan (prefix sum) algorithms.
- **lab-6-deep-learning**: Includes deep learning models and implementations using CUDA.

## Installation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/rujeetjahagirdar/GPU_Programming_Labs.git
    cd gpu-programming-assignments
    ```

2. **Compile CUDA programs**:
    Ensure you have CUDA Toolkit installed. Navigate to each lab folder and compile the programs using `nvcc`:
    ```bash
    nvcc -o <output_executable> <source_file.cu>
    ```

## Usage
1. **Run each program**:
    Execute the compiled binaries to run the respective CUDA programs. For example:
    ```bash
    ./lab-1-multidimensional-grids-data
    ```

2. **Check documentation**:
    Refer to the individual README files in each lab folder for specific usage instructions and details.

## Contributing
Feel free to fork the repository and submit pull requests with improvements or additional assignments.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## References
1. [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
2. [C++ Programming Language Documentation](https://en.cppreference.com/w/)
