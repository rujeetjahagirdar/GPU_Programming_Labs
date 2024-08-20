#include "gputk.h"
#include<iostream>
#include "color_to_grayscale_gpu.h"
using namespace std;
__global__
void colorToGrayscale_kernel(float *output, float *input, int width, int height) {
    // TODO: Implement Kernel
    // TODO: Complete CUDA Kernel for image blurring.
     int channels = 3;
     int x = blockIdx.x * blockDim.x + threadIdx.x;
     int y = blockIdx.y * blockDim.y + threadIdx.y;
     if(x>=width || y>=height) return;
     float Values[3] = {0.0f, 0.0f, 0.0f};
     for (int c=0; c<channels; ++c) {
 	int i = y * width * channels + x * channels + c;
     	Values[c] = input[i];
     }
     output[y * width + x] = 0.299f * Values[0] + 0.587f * Values[1] + 0.114f * Values[2];

}

void colorToGrayscale(float *output, float *input, int width, int height) {
    float *deviceInputImageData = nullptr;
    float *deviceOutputImageData = nullptr;
    gpuTKTime_start(GPU, "Doing GPU Computation (memory + compute)");

    gpuTKTime_start(GPU, "Doing GPU memory allocation");
    // TODO: Allocate GPU memory here
    cudaMalloc(&deviceInputImageData, 3 * width * height * sizeof(float));
    cudaMalloc(&deviceOutputImageData,  width * height * sizeof(float));
    gpuTKTime_stop(GPU, "Doing GPU memory allocation");

    gpuTKTime_start(Copy, "Copying data to the GPU");
    // TODO: Copy data to GPU here
    cudaMemcpy(deviceInputImageData, input, 3 * width * height * sizeof(float), cudaMemcpyHostToDevice);
    gpuTKTime_stop(Copy, "Copying data to the GPU");

    ///////////////////////////////////////////////////////
    gpuTKTime_start(Compute, "Doing the computation on the GPU");
    // TODO: Configure launch parameters and call kernel
   dim3 blockSize(16, 16);
   dim3 gridSize((width+blockSize.x-1)/blockSize.x, (height+blockSize.y-1)/blockSize.y);
   colorToGrayscale_kernel<<<gridSize, blockSize>>>(deviceOutputImageData, deviceInputImageData, width, height);
   cudaDeviceSynchronize(); 
   gpuTKTime_stop(Compute, "Doing the computation on the GPU");

    ///////////////////////////////////////////////////////
    gpuTKTime_start(Copy, "Copying data from the GPU");
    // TODO: Copy data from GPU here
    cudaMemcpy(output, deviceOutputImageData, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    //printf(); 
    gpuTKTime_stop(Copy, "Copying data from the GPU");

    gpuTKTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    // TODO: Free device memory
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
}
