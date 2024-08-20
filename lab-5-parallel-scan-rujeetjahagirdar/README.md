# CSE 4373/5373 Lab 5

Covers inclusive parallel scan.

# Objective

Implement a kernel to perform an inclusive parallel scan on a 1D list. The scan operator will be the addition (plus) operator. Your kernel should be able to handle input lists of arbitrary length. To simplify the lab, you can assume that the input list will be at most of length $2048 \times 65,535$ elements. This means that the computation can be performed using only one kernel launch.

The boundary condition can be handled by filling "identity value (0 for sum)" into the shared memory of the last block when the length is not a multiple of the thread block size.

# Instructions

Edit the code in the code tab to perform the following:

- allocate device memory
- copy host memory to device
- initialize thread block and kernel grid dimensions
- invoke CUDA kernel
- copy results from device to host
- deallocate device memory
- implement the work efficient scan routine
- use shared memory to reduce the number of global memory accesses, handle the boundary conditions when loading input list elements into the shared memory

Instructions about where to place each part of the code is demarcated by the `//@@` comment lines.

# Build Instructions

The data and tests can be generated and run in the same way as in previous labs.
