cmake_minimum_required(VERSION 3.10)
project(tiled_convolution)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(../../common)

add_executable(tiled_convolution tiled_convolution.cu)

set_target_properties(tiled_convolution PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "61;70;75;80"
)

target_link_libraries(tiled_convolution ${CUDA_LIBRARIES})
