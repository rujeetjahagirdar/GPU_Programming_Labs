#include "conv2d_gpu.h"

__global__ void convolve2D_constant_kernel(
    const float* input, 
    const float* kernel, 
    float* output, 
    int inputWidth, int inputHeight, 
    int kernelSize, int outputWidth, int outputHeight, 
    int stride, int dilation) {

    // TODO: Implement the kernel for constant memory filtering
}

__global__ void convolve2D_shared_kernel(
    const float* input, 
    const float* kernel, 
    float* output, 
    int inputWidth, int inputHeight, 
    int kernelSize, int outputWidth, int outputHeight, 
    int stride, int dilation) {

    // TODO: Implement the kernel for shared memory filtering
}

void convolve2D(
    const float *input,
    const float *kernel,
    float *output,
    unsigned int inputHeight,
    unsigned int inputWidth,
    unsigned int kernelSize,
    unsigned int outputHeight,
    unsigned int outputWidth,
    unsigned int padding,
    unsigned int stride,
    unsigned int dilation) {

    // Allocate device memory
    float *deviceInput, *deviceKernel, *deviceOutput;

    // TODO: Complete host function.
}