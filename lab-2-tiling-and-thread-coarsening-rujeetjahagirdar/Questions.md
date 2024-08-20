# Matrix Multiplication

The following questions are based on profiling the provided matrix multiplication code using NSight.

1. Is this code compute bound or memory bound? How do you know?

2. Using Nsight Compute, identify the biggest performance bottleneck in the code. What is the cause of this bottleneck?

# Tiled Matrix Multiplication

1. What is different about the bounds of this kernel compared to the naive matrix multiplication kernel?

2. Using Nsight Compute, identify the biggest performance bottleneck in the tiled matrix multiplication kernel. What is the cause of this bottleneck?

# Coarsened Matrix Multiplication

1. How does this kernel compare to the tiled matrix multiplication kernel in terms of performance? What is the cause of this difference?

2. Using Nsight Compute, what line of code is the biggest performance bottleneck in the coarsened matrix multiplication kernel? What is the cause of this bottleneck?