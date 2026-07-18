###############################################################################
# Day 1: Hello CUDA — 独立 CMake 配置
# @brief   该项目演示最基础的 CUDA 程序，可在本目录独立构建
# @note    不依赖上级目录的 CMakeLists.txt，完全自包含
###############################################################################

cmake_minimum_required(VERSION 3.18)

project(day01_hello_cuda
    VERSION 1.0.0
    DESCRIPTION "Day 1: 环境搭建与第一个 CUDA 程序"
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================
# 自动向上查找 common 目录（支持独立构建和子目录构建两种模式）
# ============================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../common/include)
    set(COMMON_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../common/include)
else()
    set(COMMON_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/common/include)
endif()

include_directories(${COMMON_INCLUDE})

# ============================================================
# 设置 CUDA 架构（支持 Maxwell 到 Blackwell）
# ============================================================
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
# 编译目标
# ============================================================
add_executable(hello_cuda src/main.cu)

# 设置目标属性
target_compile_options(hello_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Windows 下设置输出目录
if(WIN32)
    set_target_properties(hello_cuda PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>
    )
endif()
