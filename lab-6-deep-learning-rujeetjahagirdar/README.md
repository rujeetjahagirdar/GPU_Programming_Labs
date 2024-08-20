# CSE 4373/5373 Lab 6

Covers deep learning with cuDNN.

## Getting Started

The first part of this lab is to familiarize yourself with the structure of the code. The library used for this lab was created by Aadyot Bhatnagar who has given us permission to reuse this work for educational purposes.

The library is structured as follows:
- `main.cpp`: The entry point of the program. This file is responsible for parsing command line arguments, loading data, setting up the model, training it, and evaluating it.
- `model.cpp`: The `Model` class contains high level functions for adding layers, training, predicting, and evaluating input.
- `layer.cpp`: The `Layer` class is an abstract class that defines the interface for all layers. This file also contains implementations for each type of layer used in this lab. Most of the layers have already been implemented following the description in the class notes. We also went over this during the lecture. **You will need to complete the ~TODO~ items listed in this file.**

The accompanying quiz on Canvas will help to familiarize your with the structure of this code and deep learning pipelines in general.

## Running the Code

Before building the project, make sure you have the folders `bin` and `obj` created. The data is already on `cpe-gpu`, so you do not need to download or specify an installation.

To run the code, you will need to have a machine with a CUDA compatible GPU. You will also need to have the cuDNN library installed. The code has been tested with cuDNN and CUDA 11. A script is included to run this within the lab in ERB 125.

If you are building this at home, you can use the following commands to build the code:

```bash
make
./build/dense-neuralnet
./build/conv-neuralnet
```

All code required to run the dense neural network has already been provided. You can use that run as a benchmark to compare the performance of the convolutional neural network.
