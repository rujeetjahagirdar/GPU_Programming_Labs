SRC_PATH=src
BUILD_DIR=build

LIB_PATH=$(CURDIR)/../lib
GPUTK_LIB_PATH=$(LIB_PATH)/libgputk/lib
GPUTK_SRC_PATH=$(LIB_PATH)/libgputk
NVCC=nvcc
CXX=g++

CUDA_FLAGS=-I$(GPUTK_SRC_PATH)
CXX_FLAGS=-std=c++11 -I$(GPUTK_SRC_PATH) -I/usr/include/opencv4
LIBS=-lm -L$(GPUTK_LIB_PATH) -lgputk
OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

SOURCES := $(wildcard $(SRC_PATH)/*cpu.cpp)
OBJECTS := $(SOURCES:%=$(BUILD_DIR)/%.o)
CUDA_SOURCES := $(wildcard $(SRC_PATH)/*.cu)
CUDA_OBJECTS := $(CUDA_SOURCES:%=$(BUILD_DIR)/%.o)
DATAGEN_SOURCES := $(SRC_PATH)/dataset_generator.cpp $(SRC_PATH)/blur_cpu.cpp
DATAGEN_OBJECTS := $(DATAGEN_SOURCES:%=$(BUILD_DIR)/%.o)

EXE=$(BUILD_DIR)/main
TEST_EXE=$(BUILD_DIR)/main_test

all: $(BUILD_DIR) $(EXE)

# Main program
$(EXE): $(BUILD_DIR)/main.o $(OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) $(CUDA_FLAGS) -o $@ $^ $(LIBS) $(OPENCV_LIBS)

# Main object file
$(BUILD_DIR)/main.o: $(SRC_PATH)/main.cpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Dataset Generator Main
datagen: $(BUILD_DIR) $(BUILD_DIR)/datagen

$(BUILD_DIR)/datagen: $(DATAGEN_OBJECTS)
	$(CXX) $(CXX_FLAGS) -o $(BUILD_DIR)/datagen $^ $(LIBS)

# Testing Main
test: $(BUILD_DIR) $(TEST_EXE)

$(TEST_EXE): $(BUILD_DIR)/main_test.o $(OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) $(CUDA_FLAGS) -o $@ $^ $(LIBS)

# Testing object file
$(BUILD_DIR)/main_test.o: $(SRC_PATH)/main_test.cpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# C++ files
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(CUDA_FLAGS) -g -c $< -o $@

# CUDA files
$(BUILD_DIR)/%.cu.o: %.cu
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)/src

clean:
	rm -rf $(BUILD_DIR)

.SUFFIXES: .c .cu .o
.PHONY: all test clean