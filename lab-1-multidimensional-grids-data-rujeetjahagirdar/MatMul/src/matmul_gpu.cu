#include "gputk.h"

#include "matmul_gpu.h"

__global__
void sgemm_kernel(float *A, float *B, float *C, int numARows,
                  int numACols, int numBRows, int numBCols) {
    // TODO: Insert code to implement matrix multiplication here
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

     if(x>=numBCols || y>=numARows)
	return;
     float sum=0.0f;
     for(int k=0; k< numBRows;++k) {
	sum = sum + A[y * numACols + k] * B[numBCols * k + x];
     }
     C[y * numBCols + x] = sum;
}

int sgemm(float *A_h, float *B_h, float *C_h, int numARows, int numACols,
           int numBRows, int numBCols) {
    float *A_d, *B_d, *C_d;

    gpuTKTime_start(GPU, "Allocating GPU memory.");
    // TODO: Allocate GPU memory here
    // Don't forget to wrap the function calls with gpuTKCheck() macro
    gpuTKCheck(cudaMalloc(&A_d, numARows * numACols * sizeof(float)));
    gpuTKCheck(cudaMalloc(&B_d, numBRows * numBCols * sizeof(float)));
    gpuTKCheck(cudaMalloc(&C_d, numARows * numBCols * sizeof(float)));
//    cudaMalloc(&A_d, numARows * numACols * sizeof(float));
//    cudaMalloc(&B_d, numBRows * numBCols * sizeof(float));
//    cudaMalloc(&C_d, numARows * numBCols * sizeof(float));

    gpuTKTime_stop(GPU, "Allocating GPU memory.");

    gpuTKTime_start(GPU, "Copying input memory to the GPU.");
    // TODO: Copy memory to the GPU here
    gpuTKCheck(cudaMemcpy(A_d, A_h, numARows * numACols * sizeof(float), cudaMemcpyHostToDevice));
    gpuTKCheck(cudaMemcpy(B_d, B_h, numBRows * numBCols * sizeof(float), cudaMemcpyHostToDevice));
    gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

    // TODO: Initialize the grid and block dimensions here
//    dim3 blocksperGrid(ceil(numARows/32.0), ceil(numBCols/32.0),1);
    dim3 blocksperGrid(ceil(numBCols/32.0), ceil(numARows/32.0), 1);
    dim3 threadsperBlock(32,32,1);

//    gpuTKLog(TRACE, "The block dimensions are ", blockDim.x, " x ", blockDim.y);
//    gpuTKLog(TRACE, "The grid dimensions are ", gridDim.x, " x ", gridDim.y);

    gpuTKTime_start(Compute, "Performing CUDA computation");
    // TODO: Launch the GPU Kernel here
    sgemm_kernel<<<blocksperGrid, threadsperBlock>>>(A_d, B_d, C_d, numARows, numACols, numBRows, numBCols);

    cudaDeviceSynchronize();
    gpuTKTime_stop(Compute, "Performing CUDA computation");

    gpuTKTime_start(Copy, "Copying output memory to the CPU");
    // TODO: Copy the GPU memory back to the CPU here
    gpuTKCheck(cudaMemcpy(C_h, C_d, numARows * numBCols * sizeof(float), cudaMemcpyDeviceToHost));
    gpuTKTime_stop(Copy, "Copying output memory to the CPU");

    gpuTKTime_start(GPU, "Freeing GPU Memory");
    // TODO: Free the GPU memory here
    gpuTKCheck(cudaFree(A_d));
    gpuTKCheck(cudaFree(B_d));
    gpuTKCheck(cudaFree(C_d));
    gpuTKTime_stop(GPU, "Freeing GPU Memory");

    return 0;
}
