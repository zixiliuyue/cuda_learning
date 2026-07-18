cmake_minimum_required(VERSION 3.18)
project(day37_ctypes_cuda VERSION 1.0.0
        DESCRIPTION "Day 37: ctypes 调用 CUDA 动态库"
        LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

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

# 公共头文件目录（与其他 Day 项目保持一致，兼容两种布局）
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../common/include)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common/include)
else()
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common/include)
endif()

# ============================================================
# 编译为共享库（SHARED）
#   - Windows -> cuda_ctypes.dll (+ cuda_ctypes.lib 导入库)
#   - Linux   -> libcuda_ctypes.so
#   - macOS   -> libcuda_ctypes.dylib
# ============================================================
add_library(cuda_ctypes SHARED src/main.cu)

# 允许在共享库里使用可重定位设备代码（可选，增强兼容性）
set_target_properties(cuda_ctypes PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

# Windows/MSVC：源码为 UTF-8（含中文注释），用 /utf-8 消除 C4819 代码页警告
if(MSVC)
    target_compile_options(cuda_ctypes PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    target_compile_options(cuda_ctypes PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()

# 统一输出目录：把 .dll/.so 直接放到 build/ 根目录，方便 Python 定位
set_target_properties(cuda_ctypes PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
