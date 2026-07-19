#pragma once

#include "cuda_utils.h"
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <cmath>

struct BenchmarkResult {
    std::string name;
    float avg_time_ms;
    float min_time_ms;
    float max_time_ms;
    float std_dev_ms;
    float throughput_gbps;
    float gflops;
};

class Benchmark {
private:
    std::vector<BenchmarkResult> results;
    int warmup_iterations;
    int benchmark_iterations;

    float calculate_std_dev(const std::vector<float>& times, float avg) {
        if (times.size() <= 1) return 0.0f;
        float sum_sq = 0.0f;
        for (float t : times) {
            float diff = t - avg;
            sum_sq += diff * diff;
        }
        return std::sqrt(sum_sq / (times.size() - 1));
    }

public:
    Benchmark(int warmup = 2, int iterations = 10)
        : warmup_iterations(warmup), benchmark_iterations(iterations) {}

    template <typename Func>
    BenchmarkResult run(const std::string& name, Func&& func,
                       size_t data_size_bytes = 0, size_t flops = 0) {
        CudaTimer timer;

        printf("\n=== 基准测试: %s ===\n", name.c_str());
        printf("预热 %d 次...\n", warmup_iterations);

        for (int i = 0; i < warmup_iterations; ++i) {
            func();
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        printf("测试 %d 次...\n", benchmark_iterations);
        std::vector<float> times;
        times.reserve(benchmark_iterations);

        for (int i = 0; i < benchmark_iterations; ++i) {
            timer.start();
            func();
            float elapsed = timer.stop();
            times.push_back(elapsed);
        }

        float min_time = *std::min_element(times.begin(), times.end());
        float max_time = *std::max_element(times.begin(), times.end());
        float avg_time = 0.0f;
        for (float t : times) avg_time += t;
        avg_time /= times.size();
        float std_dev = calculate_std_dev(times, avg_time);

        float throughput = 0.0f;
        if (data_size_bytes > 0 && avg_time > 0) {
            throughput = (data_size_bytes * 2.0f) / (avg_time * 1e-3f) / (1024.0f * 1024.0f * 1024.0f);
        }

        float gflops_val = 0.0f;
        if (flops > 0 && avg_time > 0) {
            gflops_val = (flops * benchmark_iterations) / (avg_time * 1e-3f * benchmark_iterations) / 1e9f;
        }

        BenchmarkResult result{
            name, avg_time, min_time, max_time, std_dev, throughput, gflops_val
        };
        results.push_back(result);

        printf("结果:\n");
        printf("  平均: %.3f ms\n", avg_time);
        printf("  最小: %.3f ms\n", min_time);
        printf("  最大: %.3f ms\n", max_time);
        printf("  标准差: %.3f ms\n", std_dev);
        if (throughput > 0) {
            printf("  吞吐量: %.2f GB/s\n", throughput);
        }
        if (gflops_val > 0) {
            printf("  GFLOPS: %.2f\n", gflops_val);
        }

        return result;
    }

    void print_summary() {
        printf("\n\n========================================\n");
        printf("       基准测试汇总报告\n");
        printf("========================================\n\n");

        printf("%-25s %10s %10s %10s %10s %12s %10s\n",
               "测试名称", "平均(ms)", "最小(ms)", "最大(ms)",
               "标准差(ms)", "吞吐量(GB/s)", "GFLOPS");
        printf("--------------------------------------------------------------------------------\n");

        for (const auto& r : results) {
            printf("%-25s %10.3f %10.3f %10.3f %10.3f %12.2f %10.2f\n",
                   r.name.c_str(), r.avg_time_ms, r.min_time_ms, r.max_time_ms,
                   r.std_dev_ms, r.throughput_gbps, r.gflops);
        }

        printf("================================================================================\n\n");
    }

    const std::vector<BenchmarkResult>& get_results() const {
        return results;
    }

    void clear() {
        results.clear();
    }
};

template <typename T>
void initialize_array(T* array, size_t n, T min_val = 0, T max_val = 1) {
    for (size_t i = 0; i < n; ++i) {
        array[i] = min_val + static_cast<T>(rand()) / RAND_MAX * (max_val - min_val);
    }
}

template <typename T>
void cpu_vector_add(const T* a, const T* b, T* c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

template <typename T>
void cpu_matrix_multiply(const T* a, const T* b, T* c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T sum = 0;
            for (size_t k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}
