cmake_minimum_required(VERSION 3.18)
project(day38_cython_cuda VERSION 1.0.0 DESCRIPTION "Day 38: Cython + CUDA 混合编程" LANGUAGES CXX CUDA)

# ============================================================
# 说明:
#   Cython 扩展的主推荐编译方式是 python/setup.py (见 README)。
#   本 CMakeLists.txt 作为备用方案，仅编译 src/main.cu 生成
#   独立可执行文件 cython_cuda_demo，用于验证纯 CUDA Kernel 正确性
#   (不依赖 Python 环境)。
# ============================================================

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 公共头文件路径(兼容从项目根或子目录构建)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../common/include)
    set(COMMON_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../common/include)
else()
    set(COMMON_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/common/include)
endif()
include_directories(${COMMON_INCLUDE})

# ============================================================
# CUDA 架构选择(版本感知，确保跨 CUDA 版本都能编译)
#   - CUDA 13+：移除了 sm_50~sm_70，仅支持 sm_75 及以上(含 Blackwell sm_120)
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

# Windows/MSVC：源码为 UTF-8(含中文注释)，用 /utf-8 消除 C4819 代码页警告
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()

add_executable(cython_cuda_demo src/main.cu)
if(WIN32)
    set_target_properties(cython_cuda_demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>)
endif()
