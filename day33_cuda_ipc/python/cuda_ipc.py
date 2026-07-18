#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 33: CUDA IPC 进程间通信 — Python 版本

CUDA IPC 允许不同进程共享同一块 GPU 显存。由于 GPU 显存的跨进程共享
在纯 Python 中较难直接演示，这里用 multiprocessing 的共享内存
(shared_memory) 来类比 CUDA IPC 的工作原理：

    - 主进程分配一块共享内存并写入数据       ↔  cudaMalloc + cudaIpcGetMemHandle
    - 将共享内存的 "名字"（句柄）传给子进程   ↔  通过 socket/管道传递 IPC 句柄
    - 子进程用同一个名字打开共享内存并读写     ↔  cudaIpcOpenMemHandle
    - 子进程关闭映射                          ↔  cudaIpcCloseMemHandle

若 multiprocessing.shared_memory 不可用（极老版本 Python），
则退回到 numpy 单进程模拟。
"""

import sys
import time
import numpy as np

# multiprocessing.shared_memory 自 Python 3.8 起提供
try:
    from multiprocessing import shared_memory, Process
    SHM_AVAILABLE = True
except ImportError:
    SHM_AVAILABLE = False
    print("[提示] 当前 Python 不支持 shared_memory，将使用 numpy 单进程模拟。")

ARRAY_SIZE = 1024 * 1024
MAGIC_VALUE = np.float32(3.14159)


def consumer_process(shm_name, n):
    """子进程（消费者）：通过共享内存的名字打开同一块内存并做 *2 变换。

    这一步对应 CUDA 的 cudaIpcOpenMemHandle -> 修改 -> cudaIpcCloseMemHandle。
    """
    existing = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray((n,), dtype=np.float32, buffer=existing.buf)
    # 原地变换：每个元素 * 2（消费者的写入对主进程可见）
    arr[:] = arr[:] * 2.0
    existing.close()  # 关闭映射（对应 cudaIpcCloseMemHandle）


def demo_shared_memory():
    """使用 multiprocessing.shared_memory 演示跨进程共享内存读写。"""
    n = ARRAY_SIZE
    print("-" * 60)
    print("[演示 1] 跨进程共享内存 (类比 CUDA IPC 显存共享)")
    print("-" * 60)

    # 主进程（生产者）：创建共享内存并写入数据
    nbytes = n * np.dtype(np.float32).itemsize
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    arr = np.ndarray((n,), dtype=np.float32, buffer=shm.buf)
    arr[:] = MAGIC_VALUE + np.arange(n, dtype=np.float32)
    print(f"  [生产者] 已创建共享内存 '{shm.name}' 并写入 {n} 个元素 "
          f"({nbytes / 1024 / 1024:.2f} MB)")
    print(f"  [传递] 共享内存名字（句柄）传给子进程: {shm.name}")

    # 启动子进程（消费者），传入共享内存的名字（相当于 IPC 句柄）
    p = Process(target=consumer_process, args=(shm.name, n))
    p.start()
    p.join()
    print("  [消费者] 子进程已在共享内存上执行 *2 变换")

    # 校验：消费者的修改对主进程可见
    expected = (MAGIC_VALUE + np.arange(n, dtype=np.float32)) * 2.0
    max_diff = float(np.max(np.abs(arr - expected)))
    ok = max_diff < 1e-2
    print(f"  [校验] 消费者写入对生产者可见: {'通过' if ok else '失败'} "
          f"(最大误差 {max_diff:.2e})")

    # 释放共享内存
    shm.close()
    shm.unlink()
    return ok


def demo_numpy_fallback():
    """numpy 单进程模拟（当 shared_memory 不可用时）。"""
    n = ARRAY_SIZE
    print("-" * 60)
    print("[演示 1] numpy 单进程模拟 IPC 数据共享")
    print("-" * 60)
    arr = MAGIC_VALUE + np.arange(n, dtype=np.float32)
    arr[:] = arr[:] * 2.0
    expected = (MAGIC_VALUE + np.arange(n, dtype=np.float32)) * 2.0
    ok = bool(np.allclose(arr, expected))
    print(f"  [校验] 数据一致性: {'通过' if ok else '失败'}")
    return ok


def main():
    print()
    print("=" * 60)
    print("     Day 33: CUDA IPC 进程间通信 — Python 版本             ")
    print("=" * 60)
    print()

    print("[主机] CUDA IPC 工作原理:")
    print("  cudaIpcGetMemHandle  -> 导出显存句柄（生产者进程）")
    print("  [通过 socket/共享内存传递句柄]")
    print("  cudaIpcOpenMemHandle -> 映射到同一块显存（消费者进程）")
    print("  cudaIpcCloseMemHandle-> 关闭映射")
    print()

    start = time.time()
    if SHM_AVAILABLE:
        ok = demo_shared_memory()
    else:
        ok = demo_numpy_fallback()
    elapsed = (time.time() - start) * 1000

    print()
    print("=" * 60)
    print("                        总结                                ")
    print("=" * 60)
    print(f"  {'演示项':<28} | {'状态':<10}")
    print(f"  {'-'*28} | {'-'*10}")
    print(f"  {'跨进程共享内存读写校验':<28} | {'通过' if ok else '失败':<10}")
    print(f"  {'耗时(ms)':<28} | {elapsed:>8.4f}")
    print("=" * 60)
    print()

    print("知识点总结:")
    print("  1. CUDA IPC 让不同进程共享同一块 GPU 显存，避免设备-主机-设备拷贝")
    print("  2. IPC 句柄是定长字节序列，可经 socket/共享内存/管道 传递")
    print("  3. 需要 IPC 事件 (cudaIpcGetEventHandle) 做跨进程 GPU 同步")
    print("  4. Python 类比: multiprocessing.shared_memory")
    print("  5. 应用场景: 分布式训练、多进程推理服务、零拷贝数据管道")
    print("  6. 注意: IPC 仅支持同一台机器；跨机器需 NCCL/MPI/RDMA")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
