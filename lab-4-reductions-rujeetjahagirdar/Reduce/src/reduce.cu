// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];
#include <stdio.h>
#include "reduce.h"
#define BLOCK_SIZE 512

__global__ void reduceKernel(float *input, float *output, int len) {
  //@@ Load a segment of the input vector into shared memory

  //@@ Traverse the reduction tree

  //@@ Write the computed sum of the block to the output vector at the
  //@@ correct index
  __shared__ float input_s[BLOCK_SIZE];
  unsigned int i = threadIdx.x;
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid<len) {
    input_s[i] = input[tid];
  }
  else {
    input_s[i] = 0;
  }

  __syncthreads();

  // input_s[i] = input[i] + input[i + BLOCK_DIM];

  for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
      if (i < stride) {
          input_s[i] = input_s[i] + input_s[i + stride];
      }
      __syncthreads();
  }

  if (i == 0) {
      // *output = input_s[0];
      output[blockIdx.x] = input_s[0];
  }
}

void reduce(float *input, float *output, int len) {
    //@@ Allocate GPU memory here
  // printf("Inside reduce function\n\n");
  float *d_input, *d_output;
  int numBlocks = (len + BLOCK_SIZE - 1)/BLOCK_SIZE;
  cudaMalloc((void**)&d_input, len * sizeof(float));
  cudaMalloc((void**)&d_output, numBlocks * sizeof(float));

    //@@ Copy memory to the GPU here
  cudaMemcpy(d_input, input, len * sizeof(float), cudaMemcpyHostToDevice);

    //@@ Initialize the grid and block dimensions here
  dim3 blockSize(512, 1, 1);
  dim3 gridSize(numBlocks, 1, 1);


    //@@ Launch the GPU Kernel here
  reduceKernel<<<gridSize, blockSize>>>(d_input, d_output, len);

    //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(output, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);  
    //@@ Free the GPU memory here
  cudaFree(d_input);
  cudaFree(d_output);

  float sum = 0.0f;
  for (int i = 0; i < numBlocks; ++i) {
      // printf("%f\t",output[i]);
      sum = sum + output[i];
  }
  *output = sum;
  // printf("Solution= %f\n",sum);
  printf("Output = %f\n",*output);
}
