cmake_minimum_required(VERSION 3.18)
project(day40_pytorch_cuda_extension VERSION 1.0.0
        DESCRIPTION "Day 40: PyTorch CUDA Extension 完整实战" LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 复用项目公共头文件（cuda_utils.h / math_utils.h 等）
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../common/include)
    set(COMMON_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../common/include)
else()
    set(COMMON_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/common/include)
endif()
include_directories(${COMMON_INCLUDE})
# verify_main.cu 会 #include src 下的 .cu 文件，需要把 src 加入搜索路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

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

# ============================================================
# 备用 CMake 方案：只编译不依赖 PyTorch 的验证程序 verify_main.cu。
# 注意：这里【不定义】WITH_TORCH 宏，因此各 .cu 文件中的 PyTorch 绑定层
# 不会被编译，nvcc 无需任何 PyTorch 头文件即可完成构建。
# PyTorch 扩展本身请用 setup.py 编译（见 README）。
# ============================================================
add_executable(verify_kernels src/verify_main.cu)
if(WIN32)
    set_target_properties(verify_kernels PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>)
endif()
