#include "scan.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define SECTION_SIZE BLOCK_SIZE

__global__ void segmented_first_kernel(float *X, float *Y, float *S, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        XY[threadIdx.x] = X[i];
    } else {
        XY[threadIdx.x] = 0;
    }
    // Perform parallel scan (inclusive)
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp;
        if(threadIdx.x >= stride)
            temp = XY[threadIdx.x] + XY[threadIdx.x-stride];
        __syncthreads();
        if(threadIdx.x >= stride)
            XY[threadIdx.x] = temp;
    }

    if (i < N) {
        Y[i] = XY[threadIdx.x];
        __syncthreads();
        if (threadIdx.x == blockDim.x - 1) {
                S[blockIdx.x] = XY[SECTION_SIZE -1];
        }
    }
}

__global__ void segmented_second_kernel(float *S, unsigned int N) {
    for (unsigned int stride = 1; stride < N; stride *= 2) {
        __syncthreads();
        float temp;
        if(threadIdx.x >= stride)
            temp = S[threadIdx.x] + S[threadIdx.x-stride];
        __syncthreads();
        if(threadIdx.x >= stride)
            S[threadIdx.x] = temp;
    }
}

__global__ void segmented_third_kernel(float *S, float *Y, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float sum = 0;
        for (int j = 0; j < blockIdx.x; ++j) {
            sum += S[j];
        }
        Y[i] += sum;
    }
}


void scan(float *input, float *output, int len) {
    float *d_input, *d_output, *d_sum;
    int num_blocks = (len + SECTION_SIZE - 1) / SECTION_SIZE;
    int num_segments = (num_blocks + SECTION_SIZE - 1) / SECTION_SIZE;

    cudaMalloc((void**)&d_input, len * sizeof(float));
    cudaMalloc((void**)&d_output, len * sizeof(float));
    cudaMalloc((void**)&d_sum, num_segments * sizeof(float));

    // printf("Input array:\n");
    // for (int i = 0; i < len; ++i) {
    //     printf("%.2f ", input[i]);
    // }
    // printf("\n");

    cudaMemcpy(d_input, input, len * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim_init(SECTION_SIZE, 1, 1);
    dim3 gridDim_init(num_blocks, 1, 1);

    segmented_first_kernel<<<gridDim_init, blockDim_init>>>(d_input, d_output, d_sum, len);
    cudaDeviceSynchronize();

    dim3 blockDim_single(SECTION_SIZE, 1, 1);
    dim3 gridDim_single(1, 1, 1);

    segmented_second_kernel<<<gridDim_single, blockDim_single>>>(d_sum, num_segments);
    cudaDeviceSynchronize();

    dim3 blockDim_seg(SECTION_SIZE, 1, 1);
    dim3 gridDim_seg(num_blocks, 1, 1);

    segmented_third_kernel<<<gridDim_seg, blockDim_seg>>>(d_sum, d_output, len);
    cudaDeviceSynchronize();
    
    cudaMemcpy(output, d_output, len * sizeof(float), cudaMemcpyDeviceToHost);

    // printf("Output array:\n");
    // for (int i = 0; i < len; ++i) {
    //     printf("%.2f ", output[i]);
    // }
    // printf("\n");

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_sum);
}
