#include "conv2d_gpu.h"
#define TILE_SIZE 16
#define KERNEL_SIZE 5
#define KERNEL_RADIUS (KERNEL_SIZE / 2)

__constant__ float kFilter_constant[KERNEL_SIZE * KERNEL_SIZE]; // Define constant memory for the kernel

__global__ void convolve2D_constant_kernel(
    const float* input,
    const float* kernel, // Added kernel argument
    float* output,
    int inputWidth, int inputHeight,
    int outputWidth, int outputHeight,
    int stride, int dilation) {

    __shared__ float shared_input[TILE_SIZE][TILE_SIZE];  // Shared memory for input tile with halo

    // Calculate global thread indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate output coordinates
    int outputX = x;
    int outputY = y;

    // Calculate input tile coordinates
    int tileStartX = blockIdx.x * blockDim.x;
    int tileStartY = blockIdx.y * blockDim.y;

    // Load input tile into shared memory
    int inputX = tileStartX + threadIdx.x;
    int inputY = tileStartY + threadIdx.y;

    if (inputX >= 0 && inputX < inputWidth && inputY >= 0 && inputY < inputHeight) {
        shared_input[threadIdx.y][threadIdx.x] = input[inputY * inputWidth + inputX];
    } else {
        shared_input[threadIdx.y][threadIdx.x] = 0.0f; // Padding with zeros for out-of-bounds elements
    }

    __syncthreads();


    // Perform convolution for each output pixel
    float sum = 0.0f;
    for (int m = 0; m < KERNEL_SIZE; ++m) {
        for (int n = 0; n < KERNEL_SIZE; ++n) {
            int inputTileX = threadIdx.x + n;
            int inputTileY = threadIdx.y + m;
	    if(inputTileX < TILE_SIZE && inputTileY < TILE_SIZE){
            sum += shared_input[inputTileY][inputTileX] * kFilter_constant[m * KERNEL_SIZE + n];
	    }
        }
    }

    // Write result to output if within output bounds
    if (outputX < outputWidth && outputY < outputHeight) {
        output[outputY * outputWidth + outputX] = sum;
    }
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
    cudaMalloc((void **)&deviceInput, inputHeight * inputWidth * sizeof(float));
    cudaMalloc((void **)&deviceKernel, kernelSize * kernelSize * sizeof(float));
    cudaMalloc((void **)&deviceOutput, outputHeight * outputWidth * sizeof(float));

    // Copy data to device
    cudaMemcpy(deviceInput, input, inputHeight * inputWidth * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceKernel, kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kFilter_constant, kernel, (kernelSize)*(kernelSize)*sizeof(float));
    cudaDeviceSynchronize();
    // Execute the kernel
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((outputWidth + dimBlock.x - 1) / dimBlock.x,
                 (outputHeight + dimBlock.y - 1) / dimBlock.y);
    convolve2D_constant_kernel<<<dimGrid, dimBlock>>>(deviceInput, deviceKernel, deviceOutput, inputWidth, inputHeight, outputWidth, outputHeight,stride, dilation);
    gpuErrchk(cudaDeviceSynchronize());

    // Copy data back to host
    gpuErrchk(cudaMemcpy(output, deviceOutput,
                          outputHeight * outputWidth * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Free memory
    cudaFree(deviceInput);
    cudaFree(deviceKernel);
    cudaFree(deviceOutput);
}
