/******************************************************************************
 * @file    math_utils.h
 * @brief   数学辅助工具函数
 * @details 提供数据初始化、结果验证、精度比较等常用数学操作。
 *          所有函数都设计为可在主机端（CPU）执行，用于验证 GPU 计算结果的正确性。
 *
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  用随机浮点数初始化数组
 * @param  array   目标数组指针（主机内存）
 * @param  n       数组元素个数
 * @param  min_val 随机数最小值
 * @param  max_val 随机数最大值
 * @return void
 *
 * @details 使用 C 标准库的 rand() 函数生成随机数。
 *          对于需要可重复结果的场景，建议在调用前先用 srand() 设置随机种子。
 */
static inline void init_random_float(float* array, int n, float min_val, float max_val)
{
    if (array == NULL || n <= 0) return;

    // 计算随机数范围
    float range = max_val - min_val;

    for (int i = 0; i < n; i++)
    {
        // rand() 返回 [0, RAND_MAX] 范围内的整数
        // 先归一化到 [0, 1]，再缩放到 [min_val, max_val]
        float normalized = (float)rand() / (float)RAND_MAX;
        array[i] = min_val + normalized * range;
    }
}

/**
 * @brief  用固定模式初始化数组（用于调试和验证）
 * @param  array   目标数组指针
 * @param  n       数组元素个数
 * @param  pattern 初始化模式：0=全0, 1=全1, 2=顺序(0,1,2...), 3=逆序
 * @return void
 */
static inline void init_pattern_float(float* array, int n, int pattern)
{
    if (array == NULL || n <= 0) return;

    for (int i = 0; i < n; i++)
    {
        switch (pattern)
        {
            case 0:  array[i] = 0.0f; break;
            case 1:  array[i] = 1.0f; break;
            case 2:  array[i] = (float)i; break;
            case 3:  array[i] = (float)(n - 1 - i); break;
            default: array[i] = (float)i; break;
        }
    }
}

/**
 * @brief  比较两个浮点数组是否相等（允许误差）
 * @param  a         第一个数组
 * @param  b         第二个数组
 * @param  n         数组元素个数
 * @param  tolerance 允许的相对误差阈值（如 1e-5）
 * @return int       1=在误差范围内相等, 0=不相等
 *
 * @details 使用相对误差进行比较，避免大数和小数的精度问题。
 *          相对误差公式：|a - b| / max(|a|, |b|, 1.0) < tolerance
 */
static inline int compare_float_arrays(const float* a, const float* b, int n, float tolerance)
{
    if (a == NULL || b == NULL || n <= 0) return 0;

    int mismatch_count = 0;
    const int max_print = 5;  // 最多打印 5 个不匹配的元素

    for (int i = 0; i < n; i++)
    {
        float diff = fabsf(a[i] - b[i]);
        float denom = fmaxf(fmaxf(fabsf(a[i]), fabsf(b[i])), 1.0f);
        float relative_error = diff / denom;

        if (relative_error > tolerance)
        {
            if (mismatch_count < max_print)
            {
                printf("  [不匹配] 索引 %d: GPU=%.8f, CPU=%.8f, 相对误差=%.8e\n",
                       i, a[i], b[i], relative_error);
            }
            mismatch_count++;
        }
    }

    if (mismatch_count > 0)
    {
        printf("[验证失败] 共有 %d / %d 个元素不匹配（容差: %.2e）\n", mismatch_count, n, tolerance);
        return 0;
    }

    printf("[验证通过] 所有 %d 个元素均在容差 %.2e 范围内匹配。\n", n, tolerance);
    return 1;
}

/**
 * @brief  计算数组元素的平均值
 * @param  array  数组指针
 * @param  n      数组元素个数
 * @return float  平均值
 */
static inline float array_mean(const float* array, int n)
{
    if (array == NULL || n <= 0) return 0.0f;

    double sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        sum += (double)array[i];
    }
    return (float)(sum / (double)n);
}

/**
 * @brief  计算数组元素的最大值
 * @param  array  数组指针
 * @param  n      数组元素个数
 * @return float  最大值
 */
static inline float array_max(const float* array, int n)
{
    if (array == NULL || n <= 0) return 0.0f;

    float max_val = array[0];
    for (int i = 1; i < n; i++)
    {
        if (array[i] > max_val) max_val = array[i];
    }
    return max_val;
}

/**
 * @brief  计算数组元素的最小值
 * @param  array  数组指针
 * @param  n      数组元素个数
 * @return float  最小值
 */
static inline float array_min(const float* array, int n)
{
    if (array == NULL || n <= 0) return 0.0f;

    float min_val = array[0];
    for (int i = 1; i < n; i++)
    {
        if (array[i] < min_val) min_val = array[i];
    }
    return min_val;
}

/**
 * @brief  打印数组的前 N 个元素（用于调试）
 * @param  array   数组指针
 * @param  n       要打印的元素个数
 * @param  label   打印前缀标签
 * @return void
 */
static inline void print_array_preview(const float* array, int n, const char* label)
{
    if (array == NULL || n <= 0) return;

    printf("%s [前 %d 个元素]: ", label, n);
    int print_count = (n < 10) ? n : 10;
    for (int i = 0; i < print_count; i++)
    {
        printf("%.4f ", array[i]);
    }
    if (n > print_count)
    {
        printf("... (共 %d 个元素)", n);
    }
    printf("\n");
}

#ifdef __cplusplus
}
#endif

#endif /* MATH_UTILS_H */
