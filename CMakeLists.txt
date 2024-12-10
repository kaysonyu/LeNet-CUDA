cmake_minimum_required(VERSION 3.15.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES 52;89)

project(LeNet LANGUAGES CXX CUDA)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(lenet src/lenet.cu src/utils.cpp)