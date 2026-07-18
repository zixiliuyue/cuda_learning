/******************************************************************************
 * @file    main.cu
 * @brief   Day 28: GPU 光线追踪（简易版）
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "timer.h"

#define IMAGE_WIDTH 512
#define IMAGE_HEIGHT 512
#define BLOCK_SIZE 16

struct Sphere
{
    float x, y, z, r;
};

__device__ float sphere_intersect(const Sphere& s, float ox, float oy, float* n)
{
    float dx = ox - s.x;
    float dy = oy - s.y;
    float rad = s.r;
    if (dx * dx + dy * dy < rad * rad)
    {
        float dz = sqrtf(rad * rad - dx * dx - dy * dy);
        *n = dz / rad;
        return dz + s.z;
    }
    return -1.0f;
}

__global__ void raytrace_kernel(unsigned char* d_output, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;

    float x = (float)col / (float)width * 2.0f - 1.0f;
    float y = (float)row / (float)height * 2.0f - 1.0f;

    Sphere spheres[] = {
        {0.0f, 0.0f, -2.0f, 0.5f},
        {-0.6f, 0.3f, -1.5f, 0.3f},
        {0.5f, -0.2f, -2.5f, 0.4f}
    };

    float t = -1.0f;
    float n = 0.0f;
    int hit = -1;

    for (int i = 0; i < 3; i++)
    {
        float tn;
        float tz = sphere_intersect(spheres[i], x, y, &tn);
        if (tz > 0.0f && (t < 0.0f || tz < t))
        {
            t = tz;
            n = tn;
            hit = i;
        }
    }

    int idx = (row * width + col) * 3;
    if (hit >= 0)
    {
        unsigned char shade = (unsigned char)(255.0f * fmaxf(n, 0.0f));
        d_output[idx] = shade;
        d_output[idx + 1] = shade;
        d_output[idx + 2] = shade;
    }
    else
    {
        d_output[idx] = 50;
        d_output[idx + 1] = 100;
        d_output[idx + 2] = 150;
    }
}

int main(void)
{
    printf("\n=============================================================\n");
    printf("     Day 28: GPU 光线追踪（简易版）                           \n");
    printf("=============================================================\n\n");

    print_device_info();

    int width = IMAGE_WIDTH;
    int height = IMAGE_HEIGHT;
    printf("[主机] 渲染尺寸: %d x %d\n", width, height);

    unsigned char* h_output = (unsigned char*)malloc(width * height * 3);
    unsigned char* d_output = NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_output, width * height * 3));

    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("\n[主机] 渲染场景 ...\n");
    GpuTimer timer;
    gpu_timer_init(&timer);
    gpu_timer_start(&timer);
    raytrace_kernel<<<grid_dim, block_dim>>>(d_output, width, height);
    cudaDeviceSynchronize();
    gpu_timer_stop(&timer);
    gpu_timer_print(&timer, "光线追踪渲染 ");

    CUDA_SAFE_CALL(cudaMemcpy(h_output, d_output, width * height * 3, cudaMemcpyDeviceToHost));

    printf("\n=============================================================\n");
    printf("     光线追踪优化方向                                         \n");
    printf("=============================================================\n");
    printf("  1. BVH: 加速结构减少求交次数\n");
    printf("  2. RT Core: NVIDIA 硬件光线追踪单元\n");
    printf("  3. 路径追踪: 全局光照效果\n");
    printf("  4. OptiX: NVIDIA 官方光线追踪 SDK\n");
    printf("=============================================================\n\n");

    cudaFree(d_output);
    free(h_output);
    gpu_timer_destroy(&timer);

    printf("[主机] 程序正常退出。\n\n");
    return 0;
}
