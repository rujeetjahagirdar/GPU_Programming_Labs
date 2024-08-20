#include <stdio.h>
#include "gputk.h"
#include "reduce.h"

int main(int argc, char **argv) {
    int ii;
    gpuTKArg_t args;
    float *hostInput;  // The input 1D list
    float *hostOutput; // The output list
    float *deviceInput;
    float *deviceOutput;
    int numInputElements;  // number of elements in the input list
    int numOutputElements; // number of elements in the output list

    args = gpuTKArg_read(argc, argv);
    // printf("Input Arguments:\n");
    // for (ii = 0; ii < argc; ii++) {
    //     printf("argv[%d]: %s\n", ii, argv[ii]);
    // }
    hostInput =
        (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numInputElements);
    // printf("hostInput:\n");
    // for (ii = 0; ii < numInputElements; ii++) {
    //     printf("%f ", hostInput[ii]);
    // }
    // printf("\n");
    numOutputElements = numInputElements / (BLOCK_SIZE << 1);
    if (numInputElements % (BLOCK_SIZE << 1)) {
        numOutputElements++;
    }
    hostOutput = (float *)malloc(numOutputElements * sizeof(float));

    //@@ Allocate GPU memory here

    //@@ Copy memory to the GPU here

    //@@ Initialize the grid and block dimensions here

    //@@ Launch the GPU Kernel here

    //@@ Copy the GPU memory back to the CPU here

    /********************************************************************
     * Reduce output vector on the host
     ********************************************************************/

    //@@ Free the GPU memory here
    reduce(hostInput, hostOutput,numInputElements);
    // printf("Host Output = %f\n",*hostOutput);
    // for (ii = 0; ii < numInputElements; ii++) {
    //     printf("%f ", hostInput[ii]);
    // }
    // printf("endsdlksldkl1\n");
    gpuTKSolution(args, hostOutput, 1);
    

    free(hostInput);
    free(hostOutput);

    return 0;
}