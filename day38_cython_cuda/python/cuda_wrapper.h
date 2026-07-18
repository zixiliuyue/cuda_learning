/******************************************************************************
 * @file    cuda_wrapper.h
 * @brief   Day 38: 供 Cython 声明使用的 C 接口头文件
 * @details Cython 的 .pyx 通过 `cdef extern from "cuda_wrapper.h"` 引入这些
 *          函数声明。这里必须与 src/main.cu 中 extern "C" 定义的签名完全一致。
 *
 *          为什么需要这个头文件？
 *          - Cython 生成的是 C/C++ 代码，编译时需要看到函数原型才能正确调用；
 *          - 使用 extern "C" 确保 C++ 编译器不修饰函数名，链接期符号可匹配。
 *
 * @author  CUDA 30天学习项目
 * @date     2026-07-18
 ******************************************************************************/

#ifndef CUDA_WRAPPER_H
#define CUDA_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

/* 向量加法：out = a + b，返回 0=成功, -1=错误 */
int vector_add_wrapper(const float* a, const float* b, float* out, int n);

/* 方阵乘法：c = a * b（width x width），返回 0=成功, -1=错误 */
int matmul_wrapper(const float* a, const float* b, float* c, int width);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_WRAPPER_H */
