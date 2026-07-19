cmake_minimum_required(VERSION 3.18)
project(Speed-Demon-Cuda LANGUAGES CXX CUDA)

# C++标准设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA标准设置
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 包含公共头文件目录
include_directories(${CMAKE_SOURCE_DIR}/common)

# CUDA架构设置
set(CMAKE_CUDA_ARCHITECTURES 75 86)

# 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# 查找所有子目录中的CUDA源文件
file(GLOB_RECURSE CUDA_SOURCES
    02_practice/*.cu
    03_optimization/*.cu
    04_memory/*.cu
    05_concurrent/*.cu
    06_full_case/*.cu
    08_benchmarks/*.cu
)

# 为每个CUDA源文件创建可执行文件
foreach(source ${CUDA_SOURCES})
    get_filename_component(filename ${source} NAME_WE)
    get_filename_component(dirpath ${source} DIRECTORY)
    get_filename_component(dirname ${dirpath} NAME)
    set(target_name "${dirname}_${filename}")
    add_executable(${target_name} ${source})
    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY ${dirpath}
    )
endforeach()

message(STATUS "Speed-Demon-Cuda 项目配置完成 🚀")
