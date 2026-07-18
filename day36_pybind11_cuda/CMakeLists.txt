# ============================================================
#  Day 36: pybind11 CUDA 封装调用 - CMake 构建脚本
#
#  本脚本会：
#    1. 通过 FetchContent 自动从 GitHub 拉取 pybind11（无需手动安装）
#    2. 根据 CUDA 版本自动选择合适的 GPU 架构
#    3. 使用 pybind11_add_module 把 src/main.cu 编译成
#       Python 可 import 的扩展模块（Windows: .pyd，Linux: .so）
#    4. 把生成的模块输出到 build/ 目录，方便 Python 直接 import
# ============================================================

cmake_minimum_required(VERSION 3.18)
project(day36_pybind11_cuda LANGUAGES CXX CUDA)

# C++ / CUDA 标准（pybind11 v2.13 需要 C++14 及以上）
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ------------------------------------------------------------
# 使用 FetchContent 自动下载 pybind11
#   - 用户无需手动 pip install pybind11 或系统安装
#   - 首次配置时会联网克隆，之后会缓存到 build/_deps 目录
#   - 如果本机已装 pybind11，也可改用：
#       find_package(pybind11 CONFIG REQUIRED)
# ------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.6)
FetchContent_MakeAvailable(pybind11)

# ------------------------------------------------------------
# CUDA 架构选择（版本感知，确保跨 CUDA 版本都能编译）
#   - CUDA 13+：移除了 sm_50~sm_70，仅支持 sm_75 及以上（含 Blackwell sm_120）
#   - CUDA 12.x：支持 sm_50 到 sm_90
#   - 更早版本：使用传统架构列表
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 编译 Python 扩展模块
#   pybind11_add_module 会自动处理 Python 头文件、库、导出符号等
#   目标名 cuda_pybind 必须与 main.cu 中 PYBIND11_MODULE(cuda_pybind, m) 一致
# ------------------------------------------------------------
pybind11_add_module(cuda_pybind src/main.cu)

# 引入 common/include 中的通用工具头（如需使用 timer.h 等）
target_include_directories(cuda_pybind PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include)

# 允许在 .cu 中使用 CUDA 分离编译与扩展的 lambda 等特性
set_target_properties(cuda_pybind PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

# ------------------------------------------------------------
# 设置输出目录到 build/ 根目录，方便 Python import
#   注意：MSVC 多配置生成器会默认加上 Release/Debug 子目录，
#   这里对各配置统一强制输出到 ${CMAKE_BINARY_DIR}
# ------------------------------------------------------------
set_target_properties(cuda_pybind PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

foreach(cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(cuda_pybind PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${cfg} ${CMAKE_BINARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY_${cfg} ${CMAKE_BINARY_DIR})
endforeach()
