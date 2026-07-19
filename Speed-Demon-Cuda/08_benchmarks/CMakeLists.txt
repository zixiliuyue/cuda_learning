cmake_minimum_required(VERSION 3.18)
project(cuda_benchmarks LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找CUDA
find_package(CUDA REQUIRED)

# 包含目录
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)

# 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")

# 可执行文件
add_executable(run_all_benchmarks run_all_benchmarks.cu)

# 设置GPU架构（根据你的GPU调整）
set_target_properties(run_all_benchmarks PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86"
)

# 链接库
target_link_libraries(run_all_benchmarks ${CUDA_LIBRARIES})

message(STATUS "CUDA版本: ${CUDA_VERSION}")
message(STATUS "CUDA编译器: ${CMAKE_CUDA_COMPILER}")
