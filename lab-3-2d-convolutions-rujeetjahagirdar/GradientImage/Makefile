SRC_PATH=src
BUILD_DIR=build

LIB_PATH=$(CURDIR)/../lib
GPUTK_LIB_PATH=$(LIB_PATH)/libgputk/lib
GPUTK_SRC_PATH=$(LIB_PATH)/libgputk
NVCC=nvcc
CXX=g++

CUDA_FLAGS=-I $(GPUTK_SRC_PATH)
CXX_FLAGS=-std=c++11 -I $(GPUTK_SRC_PATH) -I/usr/include/opencv4
LIBS=-lm -L $(GPUTK_LIB_PATH) -lgputk
OPENCV_LIBS=-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

SOURCES := $(wildcard $(SRC_PATH)/*cpu.cpp) $(SRC_PATH)/utils.cpp
OBJECTS := $(SOURCES:%=$(BUILD_DIR)/%.o)
CUDA_SOURCES := $(wildcard $(SRC_PATH)/*.cu)
CUDA_OBJECTS := $(CUDA_SOURCES:%=$(BUILD_DIR)/%.o)

EXE=$(BUILD_DIR)/grad_image

# Testing Main
all: $(BUILD_DIR) $(EXE)

$(EXE): $(BUILD_DIR)/grad_image.o $(OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) $(CUDA_FLAGS) -o $@ $^ $(LIBS) $(OPENCV_LIBS)

# Testing object file
$(BUILD_DIR)/grad_image.o: $(SRC_PATH)/grad_image.cpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# C++ files
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(CUDA_FLAGS) -g -c $< -o $@

# CUDA files
$(BUILD_DIR)/%.cu.o: %.cu
	$(NVCC) $(CUDA_FLAGS) -lineinfo -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)/src

clean:
	rm -rf $(BUILD_DIR)

.SUFFIXES: .c .cu .o
.PHONY: all benchmark test clean