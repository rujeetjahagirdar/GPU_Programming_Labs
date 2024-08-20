# CSE 4373/5373 Lab 1

This lab explores multidimensional data and grids. After completing this, you should have a much better understanding of how to design kernels and launch configurations for 2D and 3D data.

## Part 1: Converting an Image to Grayscale

To start off the lab, implement a kernel that converts a color image to grayscale. As discussed in the lectures, we will need to account for multiple channels when using flat indexing. Feel free to copy the implementation from the notes for this one.

### Implementation

In the `ColorToGrayscale` folder, the CPU version is already provided for you. Implement a GPU version and call the kernel in `main.cu`. It is OK if you define the kernel in this file as well.

Make sure you use the same formula to convert each pixel in both the CPU and GPU versions. In the notes we used the following formula:

$$
Y = 0.299 R + 0.587 G + 0.114 B
$$

### Testing

To test your implementation, you can run `bash run_tests.sh` from the `ColorToGrayscale` folder. This requires that you have already generated the test data. To generate the data, run `make datagen` from the `ColorToGrayscale` folder and then run `./build/datagen`.

This will test your implementation on a variety of image sizes. The test script will print out the results of each test. If your implementation is correct, all tests should pass.

### Qualitative Testing

You can test your kernel on a real image by building the main application via `make all`. As long as your kernel was implemented correctly, this will successfully compile and build a `main` executable in the `build` folder. You can then run this executable with the following command:

```bash
./build/main <image_path>
```

On the lab machines, you can compile and run the provided image with the following commands:

```bash
sbatch convert_image.sh
```

If you want to use a different image than the one provided, you can modify the `convert_image.sh` script to use your image instead.

This will produce a new image named `output.png`. The input image must be a PNG file and RGB already. If you want to use a different image format, you will need to modify the code in `main.cu` to use the appropriate library. The code uses `opencv` to read and write images. If you're running this on your own machine, you will need to install `opencv` first.

### Questions

Answer the questions related to this part in the `Questions.md` file.

## Part 2: Blurring an Image

Modify the blurring kernel introduced in the lectures to use a dynamic kernel size. The kernel size should be passed in as a parameter to the kernel. The kernel size should be odd and square.

### Implementation

In the `BlurImage` folder, the CPU version is already provided for you. Implement a GPU version and call the kernel in `main.cu`. Your GPU version should be able to handle any odd and square size. Additionally, it should work with both grayscale and color images. You can do this by either implementing two kernels or by using a conditional statement in the kernel.

### Testing

To test your implementation, you can run `bash run_tests.sh` from the `BlurImage` folder. This requires that you have already generated the test data. To generate the data, run `make datagen` from the `BlurImage` folder and then run `./build/datagen`.

This will test your implementation on a variety of kernel sizes and image sizes. The test script will print out the results of each test. If your implementation is correct, all tests should pass.

### Qualitative Testing

You can test your kernel on a real image by building the main application via `make all`. As long as your kernel was implemented correctly, this will successfully compile and build a `main` executable in the `build` folder. You can then run this executable with the following command:

```bash
./build/main <image_path> <kernel_size>
```

On the lab machines, you can compile and run the provided image with the following commands:

```bash
sbatch blur_image.sh
```

If you want to use a different image than the one provided, you can modify the `blur_image.sh` script to use your image instead.


### Questions

Answer the questions related to this part in the `Questions.md` file.

## Part 3: Matrix Multiplication

Implement a kernel that performs matrix multiplication. The kernel should be able to handle matrices of any size. You can assume that the two matrices are compatible for multiplication.

### Implementation

In the `MatMul` folder, the CPU version is already provided for you. Implement a GPU version and call the kernel in `matmul_gpu.cu`. Your GPU version should be able to handle any matrix sizes.

Wrap any calls to CUDA functions with the error checking macro defined in `matmul_gpu.h`. This will automatically check the return value of the CUDA function and print an error message if it fails.

To compile the code, run `make test` from the `MatMul` folder. This will compile the GPU version and the main program that tests it.

### Testing

To test your implementation, you can run `bash run_tests.sh` from the `MatMul` folder. This requires that you have already generated the test data. To generate the data, run `make datagen` from the `MatMul` folder and then run `./build/datagen`.

This will test your implementation on a variety of matrix sizes. The test script will print out the results of each test. If your implementation is correct, all tests should pass.

### Questions

Answer the questions related to this part in the `Questions.md` file.