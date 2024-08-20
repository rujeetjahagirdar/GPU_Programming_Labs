#include "conv2d_gpu.h"
#define FILTER_RADIUS 2

__device__ __constant__ float kFilter_constant[5 * 5];

__global__ void convolve2D_constant_kernel(
    const float* input, 
    const float* kernel, 
    float* output, 
    int inputWidth, int inputHeight, 
    int kernelSize, int outputWidth, int outputHeight, 
    int stride, int dilation) {
    int kernelRadius = kernelSize / 2;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Assuming input and kernel are already padded as necessary
        if(x < outputWidth && y < outputHeight) {
            float sum = 0;
            for (int m = -kernelRadius; m <= kernelRadius; ++m) {
                for (int n = -kernelRadius; n <= kernelRadius; ++n) {
                    // Calculate the position in the input, considering stride, and dilation
                    int ix = x * stride + n * dilation;
                    int iy = y * stride + m * dilation;

                    // Check bounds
                    if (ix >= 0 && ix < inputWidth && iy >= 0 && iy < inputHeight) {
                        int inputOffset =
                            iy * inputWidth + ix;
                        int kernelOffset = (m + kernelRadius) * kernelSize +
                                           (n + kernelRadius);
                        // Perform multiplication and accumulate
                        sum += input[inputOffset] * kFilter_constant[kernelOffset];
                    }
                }
            }
            // Write the computed value to the output matrix
            output[y * outputWidth + x] = sum;
        }
    }

__global__ void convolve2D_shared_kernel(
    const float* input, 
    const float* kernel, 
    float* output, 
    int inputWidth, int inputHeight, 
    int kernelSize, int outputWidth, int outputHeight, 
    int stride, int dilation) {
    
    __device__ __shared__ float kFilter_shared[5 * 5];
    
    // TODO: Implement the kernel for shared memory filtering
    int thrdId = threadIdx.y * blockDim.x + threadIdx.x;
    if(thrdId< kernelSize*kernelSize) {
	kFilter_shared[thrdId]=kernel[thrdId];
    }
    __syncthreads();
    int kernelRadius = kernelSize / 2;
      int x = blockIdx.x * blockDim.x + threadIdx.x;
      int y = blockIdx.y * blockDim.y + threadIdx.y;
      // Assuming input and kernel are already padded as necessary
          if(x < outputWidth && y < outputHeight) {
              float sum = 0;
              for (int m = -kernelRadius; m <= kernelRadius; ++m) {
                  for (int n = -kernelRadius; n <= kernelRadius; ++n) {
                      // Calculate the position in the input, considering stride, and dilation
                      int ix = x * stride + n * dilation;
                      int iy = y * stride + m * dilation;
  
                      // Check bounds
                      if (ix >= 0 && ix < inputWidth && iy >= 0 && iy < inputHeight) {
                          int inputOffset =
                              iy * inputWidth + ix;
                          int kernelOffset = (m + kernelRadius) * kernelSize +
                                             (n + kernelRadius);
                          // Perform multiplication and accumulate
                          sum += input[inputOffset] * kFilter_shared[kernelOffset];
                      }
                  }
              }
              // Write the computed value to the output matrix
              output[y * outputWidth + x] = sum;
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
    dim3 dimBlock(16, 16);
    dim3 dimGrid((outputWidth + dimBlock.x - 1) / dimBlock.x,
                 (outputHeight + dimBlock.y - 1) / dimBlock.y);
    convolve2D_constant_kernel<<<dimGrid, dimBlock>>>(deviceInput, deviceKernel, deviceOutput, inputWidth, inputHeight, kernelSize, outputWidth, outputHeight,stride, dilation);
    convolve2D_shared_kernel<<<dimGrid, dimBlock>>>(deviceInput, deviceKernel, deviceOutput, inputWidth, inputHeight, kernelSize, outputWidth, outputHeight,stride, dilation);
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
