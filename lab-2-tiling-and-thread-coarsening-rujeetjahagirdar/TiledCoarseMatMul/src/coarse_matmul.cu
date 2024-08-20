// TODO: Insert code to implement tiled matrix multiplication

#define TILE_WIDTH 2
#define COARSE_FACTOR 3
__global__ void matMulCoarse(float *M, float *N, float *P, int width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Identify the row and column of the element to work on
    int row = by * TILE_WIDTH + ty;
    int colStart = bx * TILE_WIDTH * COARSE_FACTOR;

    // Initialize Pvalue
    float Pvalue[COARSE_FACTOR];
    for (int i = 0; i < COARSE_FACTOR; i++) {
        Pvalue[i] = 0.0f;
    }

    // Loop over the tiles required to compute the current output value
    for (int ph = 0; ph < width / TILE_WIDTH; ph++) {
        Mds[ty][tx] = M[row * width + ph * TILE_WIDTH + tx];

        for (int c = 0; c < COARSE_FACTOR; c++) {
            int col = colStart + c * TILE_WIDTH;

            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + col];
            __syncthreads();

            for (int k = 0; k < TILE_WIDTH; k++) {
                Pvalue[c] += Mds[ty][k] * Nds[k][tx];
            }
            __syncthreads();
        }
    }

    for (int c = 0; c < COARSE_FACTOR; c++) {
        int col = colStart + c * TILE_WIDTH;
        P[row * width + col] = Pvalue[c];
    }
}

int tiledmm(float *A_h, float *B_h, float *C_h, int numARows, int numACols,
           int numBRows, int numBCols) {
    float *A_d, *B_d, *C_d;
//    gpuTKTime_start(GPU, "Allocating GPU memory.");
    // TODO: Allocate GPU memory here
    // Don't forget to wrap the function calls with gpuTKCheck() macro
    //gpuTKCheck(cudaMalloc(&A_d, numARows * numACols * sizeof(float)));
    //gpuTKCheck(cudaMalloc(&B_d, numBRows * numBCols * sizeof(float)));
    //gpuTKCheck(cudaMalloc(&C_d, numARows * numBCols * sizeof(float)));
    cudaMalloc(&A_d, numARows * numACols * sizeof(float));
    cudaMalloc(&B_d, numBRows * numBCols * sizeof(float));
    cudaMalloc(&C_d, numARows * numBCols * sizeof(float));

//    gpuTKTime_stop(GPU, "Allocating GPU memory.");

  //  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
    // TODO: Copy memory to the GPU here
//    gpuTKCheck(cudaMemcpy(A_d, A_h, numARows * numACols * sizeof(float), cudaMemcpyHostToDevice));
  //  gpuTKCheck(cudaMemcpy(B_d, B_h, numBRows * numBCols * sizeof(float), cudaMemcpyHostToDevice));
    cudaMemcpy(A_d, A_h, numARows * numACols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, numBRows * numBCols * sizeof(float), cudaMemcpyHostToDevice); 
//gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

    // TODO: Initialize the grid and block dimensions here
//    dim3 blocksperGrid(ceil(numARows/32.0), ceil(numBCols/32.0),1);
    dim3 blocksperGrid(ceil(numBCols/TILE_WIDTH), ceil(numARows/TILE_WIDTH), 1);
    dim3 threadsperBlock(TILE_WIDTH, TILE_WIDTH, 1);

//    gpuTKLog(TRACE, "The block dimensions are ", blockDim.x, " x ", blockDim.y);
//    gpuTKLog(TRACE, "The grid dimensions are ", gridDim.x, " x ", gridDim.y);

    //gpuTKTime_start(Compute, "Performing CUDA computation");
    // TODO: Launch the GPU Kernel here
    MatMulKernel<<<blocksperGrid, threadsperBlock>>>(A_d, B_d, C_d, numACols);

    cudaDeviceSynchronize();
    //gpuTKTime_stop(Compute, "Performing CUDA computation");

    //gpuTKTime_start(Copy, "Copying output memory to the CPU");
    // TODO: Copy the GPU memory back to the CPU here
//    gpuTKCheck(cudaMemcpy(C_h, C_d, numARows * numBCols * sizeof(float), cudaMemcpyDeviceToHost));
    cudaMemcpy(C_h, C_d, numARows * numBCols * sizeof(float), cudaMemcpyDeviceToHost);
    //gpuTKTime_stop(Copy, "Copying output memory to the CPU");

    //gpuTKTime_start(GPU, "Freeing GPU Memory");
    // TODO: Free the GPU memory here
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    //gpuTKTime_stop(GPU, "Freeing GPU Memory");

    return 0;
}
