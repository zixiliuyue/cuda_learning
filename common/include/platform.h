/******************************************************************************
 * @file    platform.h
 * @brief   跨平台检测宏定义
 * @details 自动检测当前编译平台（Windows / macOS / Linux），
 *          提供统一的平台抽象宏，方便编写跨平台代码。
 *
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#ifndef PLATFORM_H
#define PLATFORM_H

/* ============================================================
 *  平台自动检测
 * ============================================================ */

// Windows 平台检测
// _WIN32 在 32位和64位 Windows 上都定义
#if defined(_WIN32) || defined(_WIN64)
    #define PLATFORM_WINDOWS 1
    #define PLATFORM_NAME "Windows"

// macOS / iOS 平台检测
// __APPLE__ 在所有 Apple 平台上都定义
// 进一步用 TARGET_OS_MAC 区分 macOS 和 iOS
#elif defined(__APPLE__)
    #define PLATFORM_MACOS 1
    #define PLATFORM_NAME "macOS"

// Linux 平台检测
// __linux__ 在 Linux 内核系统上定义（包括 Android）
#elif defined(__linux__)
    #define PLATFORM_LINUX 1
    #define PLATFORM_NAME "Linux"

#else
    #define PLATFORM_UNKNOWN 1
    #define PLATFORM_NAME "Unknown"
#endif

/* ============================================================
 *  编译器检测
 * ============================================================ */

#if defined(_MSC_VER)
    #define COMPILER_MSVC 1
    #define COMPILER_NAME "MSVC"
#elif defined(__clang__)
    #define COMPILER_CLANG 1
    #define COMPILER_NAME "Clang"
#elif defined(__GNUC__)
    #define COMPILER_GCC 1
    #define COMPILER_NAME "GCC"
#else
    #define COMPILER_NAME "Unknown"
#endif

/* ============================================================
 *  架构检测
 * ============================================================ */

#if defined(_M_X64) || defined(__x86_64__)
    #define ARCH_X86_64 1
    #define ARCH_NAME "x86_64"
#elif defined(_M_IX86) || defined(__i386__)
    #define ARCH_X86 1
    #define ARCH_NAME "x86"
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define ARCH_ARM64 1
    #define ARCH_NAME "ARM64"
#else
    #define ARCH_NAME "Unknown"
#endif

/* ============================================================
 *  便捷宏函数
 * ============================================================ */

/**
 * @brief  打印当前平台信息的便捷宏
 * @usage  PRINT_PLATFORM_INFO();
 */
#define PRINT_PLATFORM_INFO()                                               \
    do {                                                                    \
        printf("[平台信息] 操作系统: %s | 编译器: %s | 架构: %s\n",        \
               PLATFORM_NAME, COMPILER_NAME, ARCH_NAME);                    \
    } while (0)

/**
 * @brief  判断当前平台是否支持原生 CUDA 运行
 * @details Windows 和 Linux 通常支持 CUDA（需 NVIDIA GPU）
 *          macOS 自 10.14 后不再支持 NVIDIA GPU，需要通过其他方式运行
 */
#if defined(PLATFORM_WINDOWS) || defined(PLATFORM_LINUX)
    #define CUDA_NATIVE_SUPPORTED 1
#else
    #define CUDA_NATIVE_SUPPORTED 0
#endif

#endif /* PLATFORM_H */
