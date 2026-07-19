#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 40: PyTorch CUDA Extension 编译脚本

用法（在 day40_pytorch_cuda_extension 目录下执行）：
    python setup.py install        # 编译并安装到当前 Python 环境
    python setup.py build_ext --inplace   # 只编译，产物留在当前目录

编译后即可在 Python 中：
    import cuda_custom_ops
    cuda_custom_ops.relu_forward(x)

说明：
  - CUDAExtension 会自动帮我们找到 PyTorch 的头文件与 CUDA 工具链。
  - BuildExtension 是 PyTorch 提供的自定义 build_ext，负责把 .cu 交给 nvcc。
  - 我们通过 extra_compile_args 给 nvcc 和主机编译器传入 -DWITH_TORCH，
    这样各 .cu 文件里的 PyTorch 绑定层（#ifdef WITH_TORCH 段）才会被编译。
"""

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# 传给 C++/CUDA 编译器的额外参数：
#   - cxx  : 主机端（MSVC / g++）编译参数
#   - nvcc : 设备端（nvcc）编译参数
# 定义 WITH_TORCH 宏，开启 .cu 文件中的 PyTorch 绑定与算子注册。
extra_compile_args = {
    'cxx':  ['-DWITH_TORCH'],
    'nvcc': ['-DWITH_TORCH'],
}

setup(
    name='cuda_custom_ops',
    version='1.0.0',
    description='Day 40: 自定义 CUDA 算子（ReLU / LayerNorm）PyTorch 扩展',
    ext_modules=[
        CUDAExtension(
            name='cuda_custom_ops',
            sources=[
                'src/relu_forward.cu',
                'src/relu_backward.cu',
                'src/layer_norm_forward.cu',
                'src/layer_norm_backward.cu',
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
