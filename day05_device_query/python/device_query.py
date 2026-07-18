#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 5: GPU 设备查询 — Python/Numba 版本"""

import sys

try:
    from numba import cuda
except ImportError:
    print("[错误] 未安装 Numba。请运行: pip install numba")
    sys.exit(1)


def main():
    print()
    print("=" * 60)
    print("     Day 5: GPU 设备查询 — Python/Numba 版本              ")
    print("=" * 60)
    print()

    if not cuda.is_available():
        print("[警告] 未检测到 NVIDIA GPU。")
        print("[提示] 可以通过 Google Colab 运行此代码获得免费 GPU。")
        return 0

    device = cuda.get_current_device()
    print(f"设备名称: {device.name.decode() if isinstance(device.name, bytes) else device.name}")
    print(f"计算能力: {device.compute_capability}")
    print(f"多处理器数量: {getattr(device, 'MULTIPROCESSOR_COUNT', '未知')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
