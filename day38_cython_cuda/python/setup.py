# -*- coding: utf-8 -*-
"""Day 38: Cython + CUDA 扩展编译脚本

用法（在 python/ 目录下执行）:
    python setup.py build_ext --inplace

难点说明:
    标准 setuptools 只会用系统 C/C++ 编译器编译源文件，并不认识 .cu 文件，
    也不会调用 nvcc。因此本脚本自定义了 build_ext，使得:
      - .cu 文件 -> 用 nvcc 编译成目标文件 (.obj / .o)
      - .cpp/.c (Cython 生成的) -> 用默认编译器编译
      - 最后统一链接生成 .pyd (Windows) / .so (Linux)

    这样即可把纯 CUDA 源码与 Cython 封装编译进同一个 Python 扩展模块。
"""

import os
import sys
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy as np

IS_WINDOWS = sys.platform == "win32"

# -----------------------------------------------------------------------------
# CUDA 架构选择:
#   - Windows 上示例默认使用 sm_120 (Blackwell)，如你的显卡较旧可自行修改
#   - 其它平台默认 sm_75 (Turing)，兼容大多数较新显卡
# -----------------------------------------------------------------------------
NVCC_ARCH = "-arch=sm_120" if IS_WINDOWS else "-arch=sm_75"

# 源文件: Cython 封装 + 纯 CUDA 实现
SOURCES = ["cython_wrapper.pyx", os.path.join("..", "src", "main.cu")]

# 头文件搜索路径: NumPy C API、common 公共头、以及当前目录(cuda_wrapper.h)
INCLUDE_DIRS = [np.get_include(), os.path.join("..", "common", "include"), "."]


def find_cuda_home():
    """定位 CUDA 安装目录，用于查找 nvcc 与 cudart 库。"""
    home = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if home and os.path.isdir(home):
        return home
    # 尝试从 PATH 中的 nvcc 反推
    from shutil import which
    nvcc = which("nvcc")
    if nvcc:
        return os.path.dirname(os.path.dirname(nvcc))
    return None


CUDA_HOME = find_cuda_home()


class CudaBuildExt(build_ext):
    """自定义 build_ext: 拦截 .cu 文件用 nvcc 编译。"""

    def build_extensions(self):
        # 保存原始的 compile 方法，稍后对非 .cu 文件回退调用
        original_compile = self.compiler.compile

        def custom_compile(sources, output_dir=None, macros=None,
                           include_dirs=None, debug=0, extra_preargs=None,
                           extra_postargs=None, depends=None):
            cu_sources = [s for s in sources if s.endswith(".cu")]
            other_sources = [s for s in sources if not s.endswith(".cu")]
            objects = []

            # 1) 先用 nvcc 编译所有 .cu 文件
            for src in cu_sources:
                obj = self._nvcc_compile(src, output_dir, include_dirs, macros)
                objects.append(obj)

            # 2) 其余源文件交给默认编译器
            if other_sources:
                objects += original_compile(
                    other_sources, output_dir, macros, include_dirs,
                    debug, extra_preargs, extra_postargs, depends)
            return objects

        # 替换 compile 方法
        self.compiler.compile = custom_compile
        super().build_extensions()

    def _nvcc_compile(self, src, output_dir, include_dirs, macros):
        """调用 nvcc 编译单个 .cu 文件，返回目标文件路径。"""
        if CUDA_HOME is None:
            raise RuntimeError("未找到 CUDA 安装 (请设置 CUDA_PATH 环境变量)")

        nvcc = os.path.join(CUDA_HOME, "bin",
                            "nvcc.exe" if IS_WINDOWS else "nvcc")
        ext = ".obj" if IS_WINDOWS else ".o"
        obj = os.path.join(output_dir, os.path.splitext(src)[0].replace("..", "_up") + ext)
        os.makedirs(os.path.dirname(obj), exist_ok=True)

        cmd = [nvcc, "-c", src, "-o", obj, NVCC_ARCH,
               "-DCYTHON_BUILD",             # 跳过 main.cu 中的 main()
               "--compiler-options", "-fPIC" if not IS_WINDOWS else "/MD"]
        for inc in (include_dirs or []):
            cmd += ["-I", inc]
        for name, val in (macros or []):
            cmd += ["-D%s" % name if val is None else "-D%s=%s" % (name, val)]
        if IS_WINDOWS:
            # 中文注释源码使用 UTF-8，避免 MSVC 代码页警告 C4819
            cmd += ["-Xcompiler", "/utf-8"]

        print("[nvcc] " + " ".join(cmd))
        subprocess.check_call(cmd)
        return obj


# -----------------------------------------------------------------------------
# 定义扩展模块
# -----------------------------------------------------------------------------
extra_link_args = []
libraries = ["cudart"]
library_dirs = []
if CUDA_HOME:
    lib_sub = "x64" if IS_WINDOWS else "lib64"
    library_dirs.append(os.path.join(CUDA_HOME, "lib", lib_sub)
                        if IS_WINDOWS else os.path.join(CUDA_HOME, lib_sub))

ext = Extension(
    "cuda_cython",
    sources=SOURCES,
    include_dirs=INCLUDE_DIRS,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_link_args=extra_link_args,
    language="c++",
)

setup(
    name="cuda_cython",
    version="1.0.0",
    description="Day 38: Cython + CUDA 混合编程扩展",
    ext_modules=cythonize([ext], language_level="3"),
    cmdclass={"build_ext": CudaBuildExt},
)
