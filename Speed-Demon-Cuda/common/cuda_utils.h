#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA 错误在 " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }

class CudaTimer {
private:
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    bool running;

public:
    CudaTimer() : running(false) {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }

    ~CudaTimer() {
        CUDA_CHECK(cudaEventDestroy(start_event));
        CUDA_CHECK(cudaEventDestroy(stop_event));
    }

    void start() {
        CUDA_CHECK(cudaEventRecord(start_event));
        running = true;
    }

    float stop() {
        if (!running) {
            std::cerr << "警告：计时器未启动！" << std::endl;
            return 0.0f;
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
        running = false;
        return milliseconds;
    }

    float stop_and_print(const char* name = "操作") {
        float ms = stop();
        printf("%s 耗时: %.3f ms\n", name, ms);
        return ms;
    }
};

class CpuTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    bool running;

public:
    CpuTimer() : running(false) {}

    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        running = true;
    }

    float stop() {
        if (!running) {
            std::cerr << "警告：计时器未启动！" << std::endl;
            return 0.0f;
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<float, std::milli>(end_time - start_time);
        running = false;
        return duration.count();
    }

    float stop_and_print(const char* name = "操作") {
        float ms = stop();
        printf("%s 耗时: %.3f ms\n", name, ms);
        return ms;
    }
};

inline void print_device_info() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    printf("=== CUDA 设备信息 ===\n");
    printf("检测到 %d 个 CUDA 设备\n\n", device_count);

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        printf("设备 %d: %s\n", i, prop.name);
        printf("  计算能力: %d.%d\n", prop.major, prop.minor);
        printf("  全局内存: %.2f GB\n", static_cast<float>(prop.totalGlobalMem) / (1024 * 1024 * 1024));
        printf("  共享内存/块: %.1f KB\n", static_cast<float>(prop.sharedMemPerBlock) / 1024);
        printf("  常量内存: %.1f KB\n", static_cast<float>(prop.totalConstMem) / 1024);
        printf("  Warp 大小: %d\n", prop.warpSize);
        printf("  多处理器数量: %d\n", prop.multiProcessorCount);
        printf("  最大线程数/块: %d\n", prop.maxThreadsPerBlock);
        printf("  最大网格维度: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  最大块维度: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("\n");
    }
    printf("=====================\n");
}

template <typename T>
void print_array(const T* array, int n, const char* name = "数组") {
    printf("%s: [", name);
    for (int i = 0; i < n; ++i) {
        if (i > 0) printf(", ");
        std::cout << array[i];
    }
    printf("]\n");
}

template <typename T>
bool verify_results(const T* expected, const T* actual, int n, float epsilon = 1e-5f) {
    for (int i = 0; i < n; ++i) {
        if (std::abs(expected[i] - actual[i]) > epsilon) {
            printf("验证失败！索引 %d: 预期 %f, 实际 %f\n", i, static_cast<float>(expected[i]), static_cast<float>(actual[i]));
            return false;
        }
    }
    printf("验证通过！所有 %d 个元素匹配 ✓\n", n);
    return true;
}
