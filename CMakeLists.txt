###############################################################################
# CUDA 30天学习项目 — 全局 CMake 配置
# @brief   根 CMakeLists.txt，用于一次性构建所有 30 天的学习项目
# @note    每个 dayXX 目录也可以独立构建，不依赖此根文件
###############################################################################

cmake_minimum_required(VERSION 3.18)

# 项目名称和版本
project(CUDA_30Days_Learning
    VERSION 1.0.0
    DESCRIPTION "CUDA 30天学习项目 — 从入门到实战"
    LANGUAGES CXX C
)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# CUDA 检测与配置
# ============================================================

# 尝试查找 CUDA 工具包
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "[CUDA 30天] 检测到 CUDA 编译器: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "[CUDA 30天] CUDA 版本: ${CMAKE_CUDA_COMPILER_VERSION}")

    # 设置 CUDA 标准
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # 允许设备代码链接（用于 separable compilation）
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    # 根据 CUDA 版本设置 architectures
    # 这里支持从 Maxwell (5.0) 到 Blackwell (12.0) 的架构
    # RTX 5060 Ti 属于 Blackwell 架构，Compute Capability 12.x
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.0")
        set(CMAKE_CUDA_ARCHITECTURES 50 52 61 70 75 80 86 89 90 100 120)
        message(STATUS "[CUDA 30天] 目标 GPU 架构: Maxwell 到 Blackwell (包括 RTX 5060 Ti)")
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.0")
        set(CMAKE_CUDA_ARCHITECTURES 50 52 61 70 75 80 86)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 50 52 61 70 75)
    endif()

    # Windows 平台额外配置
    if(WIN32)
        # 避免 Windows 下 MSVC 与 nvcc 的警告冲突
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/W3")
    else()
        # Linux/macOS 下的优化和警告选项
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Wall")
    endif()

    set(CUDA_AVAILABLE TRUE)
else()
    message(WARNING "[CUDA 30天] 未检测到 CUDA 编译器！将只构建非 CUDA 项目。")
    message(STATUS "[CUDA 30天] Windows 用户请安装 CUDA Toolkit 并确保 nvcc 在 PATH 中")
    message(STATUS "[CUDA 30天] macOS 用户请使用各项目的 Python 版本或通过 Google Colab 运行")
    set(CUDA_AVAILABLE FALSE)
endif()

# ============================================================
# 公共头文件路径
# ============================================================
set(COMMON_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/common/include)
include_directories(${COMMON_INCLUDE_DIR})

# ============================================================
# 添加所有 Day 子项目
# ============================================================
# 每个 dayXX 目录都是独立项目，可以单独进入目录构建
# 也可以从此根目录一次性构建全部

set(DAY_PROJECTS
    day01_hello_cuda
    day02_vector_add
    day03_matrix_add
    day04_matrix_transpose
    day05_device_query
    day06_tiled_matmul
    day07_coalesced_access
    day08_occupancy
    day09_reduction
    day10_streams
    day11_constant_texture
    day12_dynamic_parallel
    day13_cublas
    day14_cufft
    day15_image_filter
    day16_sort
    day17_scan
    day18_graph_bfs
    day19_monte_carlo
    day20_performance_tuning
    day21_pytorch_extension
    day22_numba
    day23_opencv_cuda
    day24_tensorrt
    day25_multi_gpu
    day26_amp_training
    day27_nbody
    day28_raytracing
    day29_inference_engine
    day30_summary
)

foreach(day_proj ${DAY_PROJECTS})
    if(EXISTS ${CMAKE_SOURCE_DIR}/${day_proj}/CMakeLists.txt)
        add_subdirectory(${day_proj})
    else()
        message(STATUS "[CUDA 30天] 跳过 ${day_proj} (CMakeLists.txt 尚未创建)")
    endif()
endforeach()

# ============================================================
# 构建摘要
# ============================================================
message(STATUS "")
message(STATUS "=============================================================")
message(STATUS "          CUDA 30天学习项目 — 构建配置摘要                ")
message(STATUS "=============================================================")
message(STATUS "  构建类型        : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA 可用       : ${CUDA_AVAILABLE}")
if(CUDA_AVAILABLE)
    message(STATUS "  CUDA 编译器     : ${CMAKE_CUDA_COMPILER}")
    message(STATUS "  CUDA 版本       : ${CMAKE_CUDA_COMPILER_VERSION}")
    message(STATUS "  GPU 架构列表    : ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  C++ 编译器      : ${CMAKE_CXX_COMPILER}")
message(STATUS "  平台            : ${CMAKE_SYSTEM_NAME}")
message(STATUS "=============================================================")
message(STATUS "")
message(STATUS "提示: 可以进入任意 dayXX 目录独立构建:")
message(STATUS "      cd day01_hello_cuda && mkdir build && cd build && cmake .. && cmake --build .")
message(STATUS "")
