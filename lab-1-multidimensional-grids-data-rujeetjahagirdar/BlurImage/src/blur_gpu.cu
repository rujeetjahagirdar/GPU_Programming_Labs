#include "blur_gpu.h"

#include "gputk.h"

__global__
void blurKernel(float *out, float *in, int size, int width, int height, int channels) {
    // TODO: Complete CUDA Kernel for image blurring.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
//    printf("inside kernel= %d\n",channels);
    if(x>=width || y>=height) return;
//    float sum = 0.0f;
//    int numPixels = 0;
//    printf("size = %d\n",size);
    for (int c=0; c<channels; ++c) {
	float sum = 0.0f;
	int numPixels = 0;
    	for(int ky=-size; ky<=size; ++ky) {
//		printf("ky= %g\n",ky);
    		for(int kx=-size; kx<=size; ++kx) {
//        		printf("ky= %d\tkx= %d\n",ky,kx);
			if(y + ky < 0 || y + ky >=height) continue;
			if(x + kx < 0 || x + kx >=width) continue;
			int i = (y+ky) * width * channels + (x+kx) * channels + c;
			//printf("i= %d\n",i);
			sum = sum + in[i];
	//		printf("in[i]= %f\t",in[i]);
	//		printf("sum = %f\n",sum);
			numPixels = numPixels + 1;
	//		printf("NumPixels =%d\n", numPixels);
		}
    	}
    	out[y * width * channels + x * channels + c] = sum/numPixels; 
//    	printf("%f\n",sum/numPixels); 
    }
}

void blur(float *out_h, float *in_h, int size, int width, int height, int channels) {
    float *deviceInputImageData = nullptr;
    float *deviceOutputImageData = nullptr;

    gpuTKTime_start(GPU, "Doing GPU Computation (memory + compute)");
//    printf("channels= %d\n",channels);
    gpuTKTime_start(GPU, "Doing GPU memory allocation");
    // TODO: Allocate device memory
    cudaMalloc(&deviceInputImageData, channels * width * height * sizeof(float));
    cudaMalloc(&deviceOutputImageData, channels * width * height * sizeof(float));
    gpuTKTime_stop(GPU, "Doing GPU memory allocation");

    gpuTKTime_start(Copy, "Copying data to the GPU");
    // TODO: Copy data to device
    cudaMemcpy(deviceInputImageData, in_h, channels * width * height * sizeof(float), cudaMemcpyHostToDevice);
    gpuTKTime_stop(Copy, "Copying data to the GPU");

    // TODO: Set up block and grid sizes
    dim3 blocksperGrid(ceil(width/32.0), ceil(height/32.0),1);
    dim3 threadsperBlock(32,32,1);

    gpuTKTime_start(Compute, "Doing the computation on the GPU");
    // TODO: Launch kernel
    blurKernel<<<blocksperGrid, threadsperBlock>>>(deviceOutputImageData, deviceInputImageData, size, width, height, channels);
    gpuTKTime_stop(Compute, "Doing the computation on the GPU");

    gpuTKTime_start(Copy, "Copying data from the GPU");
    // TODO: Copy data back to host
    cudaMemcpy(out_h, deviceOutputImageData, channels * width * height * sizeof(float), cudaMemcpyDeviceToHost);
    gpuTKTime_stop(Copy, "Copying data from the GPU");

    // TODO: Free device memory
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
}
