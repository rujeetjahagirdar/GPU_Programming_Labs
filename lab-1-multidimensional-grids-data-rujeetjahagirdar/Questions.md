# Part 1: Color to Grayscale

1. The kernel code is written in a way that each thread processes considers all 3 channels per pixel. How would this compare to a kernel that processes only one channel per pixel?

--> If each channel is processed differently then there would be situation where some threads would be idle and waiting for other threads to process other channels. This would cause inefficient use of GPU resource.

# Part 2: Blurring an Image

1. When checking for kernel size, does it make more sense to check this before launching the kernel or inside the kernel? Why?

--> Kernel size should be checked before launching kernel, which would avoid unnecessary launching of kernel code.

# Part 3: Matrix Multiplication

1. How many global memory access does each thread perform given two matrices of size $m \times n$ and $n \times p$, respectively?

--> Each thread will perform 2n+1(reading+writing) global memory access.
