cmake_minimum_required(VERSION 3.18)
project(cuda_kernels_demo VERSION 1.0.0 DESCRIPTION "高频基础 Kernel 库演示（Softmax/LayerNorm/GELU/Dropout）" LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================
# 头文件目录：
#   - COMMON_INCLUDE   : common/include（cuda_utils.h / timer.h / math_utils.h / platform.h）
#   - KERNEL_INCLUDE   : 当前目录（softmax_kernel.cuh 等 .cuh）
# ============================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../include)
    set(COMMON_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
else()
    set(COMMON_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/common/include)
endif()
include_directories(${COMMON_INCLUDE})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ============================================================
# CUDA 架构选择（版本感知，确保跨 CUDA 版本都能编译）
#   - CUDA 13+：移除了 sm_50~sm_70，仅支持 sm_75 及以上（含 Blackwell sm_120）
#   - CUDA 12.x：支持 sm_50 到 sm_90
#   - 更早版本：使用传统架构列表
# ============================================================
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0")
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90 100 120)
elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.0")
    set(CMAKE_CUDA_ARCHITECTURES 50 61 70 75 80 86 89 90)
else()
    set(CMAKE_CUDA_ARCHITECTURES 50 52 61 70 75)
endif()

# Windows/MSVC：源码为 UTF-8（含中文注释），用 /utf-8 消除 C4819 代码页警告
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()

add_executable(cuda_kernels_demo demo.cu)

# Dropout 使用 cuRAND 设备端 API（curand_kernel.h）。
# 该 API 为 header-only，通常无需显式链接 curand 库；
# 但部分工具链下链接 curand 更稳妥，这里若能找到则链接。
find_library(CURAND_LIB curand HINTS ${CUDAToolkit_LIBRARY_DIR})
if(CURAND_LIB)
    target_link_libraries(cuda_kernels_demo ${CURAND_LIB})
endif()

if(WIN32)
    set_target_properties(cuda_kernels_demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>)
endif()
