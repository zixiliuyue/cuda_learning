#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 25: 多 GPU — Python 版本

演示多 GPU 编程的概念和工作流程。
检测系统中可用的 GPU 设备，演示数据分布策略。
"""

import sys
import time
import numpy as np

# 尝试导入 PyTorch（多 GPU 支持较好）
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# 尝试导入 Numba CUDA
try:
    from numba import cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("[提示] 未安装 Numba。pip install numba")

# 常量定义
DATA_SIZE = 1024 * 1024
TOLERANCE = 1e-5
ITERATIONS = 5


def check_cuda_devices():
    """检查系统中可用的 CUDA 设备"""
    devices = []

    if NUMBA_AVAILABLE:
        num_devices = cuda.gpus.ndarray.size if hasattr(cuda.gpus, 'ndarray') else 0
        try:
            num_devices = len(cuda.gpus)
        except Exception:
            num_devices = 1 if cuda.is_available() else 0

        for i in range(num_devices):
            try:
                with cuda.gpus[i]:
                    device = cuda.get_current_device()
                    name = device.name.decode() if isinstance(device.name, bytes) else device.name
                    cc = device.compute_capability
                    devices.append({
                        'index': i,
                        'name': name,
                        'compute_capability': cc,
                        'backend': 'Numba'
                    })
            except Exception:
                pass

    if PYTORCH_AVAILABLE and torch.cuda.is_available():
        num_torch = torch.cuda.device_count()
        for i in range(num_torch):
            devices.append({
                'index': i,
                'name': torch.cuda.get_device_name(i),
                'compute_capability': torch.cuda.get_device_capability(i),
                'total_memory': torch.cuda.get_device_properties(i).total_mem,
                'backend': 'PyTorch'
            })

    return devices


def simulate_multi_gpu_numpy(data, num_gpus=2):
    """
    模拟多 GPU 数据分布计算。
    将数据切分到多个 "GPU"，分别计算后合并结果。
    """
    chunk_size = len(data) // num_gpus
    results = []

    for gpu_id in range(num_gpus):
        start_idx = gpu_id * chunk_size
        end_idx = start_idx + chunk_size if gpu_id < num_gpus - 1 else len(data)
        chunk = data[start_idx:end_idx]

        # 模拟 GPU 计算
        result_chunk = chunk * 2.0 + 1.0
        results.append(result_chunk)

    return np.concatenate(results)


def main():
    print()
    print("=" * 60)
    print("     Day 25: 多 GPU — Python 版本                            ")
    print("=" * 60)
    print()

    # ---- 设备检测 ----
    devices = check_cuda_devices()
    print(f"[主机] 检测到 {len(devices)} 个 CUDA 设备:")
    if devices:
        for d in devices:
            print(f"  GPU {d['index']}: {d['name']} "
                  f"(CC: {d['compute_capability']}, Backend: {d['backend']})")
            if 'total_memory' in d:
                print(f"           显存: {d['total_memory'] / 1024**3:.1f} GB")
    else:
        print("  (未检测到 GPU，使用模拟模式)")

    num_gpus = max(len(devices), 2)  # 至少模拟 2 GPU
    print()

    # ---- 数据并行演示 ----
    print(f"[主机] 数据大小: {DATA_SIZE:,} 元素")
    print(f"[主机] 模拟 {num_gpus} GPU 数据并行...")

    h_data = np.random.uniform(-1.0, 1.0, size=DATA_SIZE).astype(np.float32)
    h_ref = h_data * 2.0 + 1.0

    # 模拟多 GPU
    start = time.time()
    for _ in range(ITERATIONS):
        h_result = simulate_multi_gpu_numpy(h_data, num_gpus)
    multi_time = (time.time() - start) / ITERATIONS * 1000

    # 单线程（参考）
    start = time.time()
    for _ in range(ITERATIONS):
        h_single = h_data * 2.0 + 1.0
    single_time = (time.time() - start) / ITERATIONS * 1000

    # 验证
    max_diff = np.max(np.abs(h_result - h_ref))

    # PyTorch 多 GPU 演示
    if PYTORCH_AVAILABLE and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print()
        print("[主机] PyTorch 多 GPU 数据并行 (DataParallel)...")
        data_torch = torch.from_numpy(h_data).cuda()
        # 简单演示
        result_torch = data_torch * 2.0 + 1.0
        pytorch_multi = True
    else:
        pytorch_multi = False

    # 性能总结
    print()
    print("=" * 60)
    print("                        性能总结                            ")
    print("=" * 60)
    print(f"  模拟多 GPU ({num_gpus} GPU): {multi_time:.4f} ms")
    print(f"  单线程参考          : {single_time:.4f} ms")
    print(f"  结果验证            : {'通过' if max_diff < TOLERANCE else '失败'} (误差: {max_diff:.2e})")
    if pytorch_multi:
        print(f"  PyTorch DataParallel : 已演示")
    else:
        print(f"  PyTorch DataParallel : 需要 >= 2 GPU")
    print("=" * 60)
    print()

    print("知识点总结:")
    print("  1. 数据并行: 将数据切分到多个 GPU 分别计算")
    print("  2. 模型并行: 将模型拆分到多个 GPU")
    print("  3. PyTorch: DataParallel (简单) vs DistributedDataParallel (高效)")
    print("  4. NCCL: NVIDIA 的 GPU 通信库 (All-Reduce, Broadcast)")
    print("  5. 多 GPU 注意: 数据传输开销、负载均衡")
    print("  6. NVLink: GPU 间高速互连，带宽远超 PCIe")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
