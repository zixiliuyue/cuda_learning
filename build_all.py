# -*- coding: utf-8 -*-
"""
============================================================================
build_all.py  —  CUDA 40+天项目 一键编译验证脚本（无需 CMake）
============================================================================

功能：
    自动定位 MSVC(Windows) 主机编译器环境，调用 nvcc 直接编译各天
    的 main.cu，并汇总每个项目的编译成功/失败情况。
    这是本项目的"验收工具"——验收标准为：所有可独立编译的项目编译通过。

用法：
    python build_all.py                 # 默认架构 sm_120 (RTX 5060 Ti / Blackwell)
    python build_all.py sm_89           # 指定其它架构（如 RTX 40 系 Ada）

环境要求：
    - 已安装 CUDA Toolkit（nvcc 在 PATH 中）；本项目在 CUDA 13.1 上验证通过
    - Windows：安装 Visual Studio 2022（含 MSVC），脚本会自动定位 cl.exe
    - Linux/macOS：需有 gcc/clang，脚本直接调用 nvcc

CUDA 13 重要变化（本项目已全部适配）：
    - nvcc 不再支持 sm_50 ~ sm_70，最低 sm_75
    - cudaDeviceProp 的 memoryClockRate / memoryBusWidth /
      singleToDoublePrecisionPerfRatio 字段被移除，改用 cudaDeviceGetAttribute
============================================================================
"""
import os
import sys
import glob
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
COMMON_INC = os.path.join(ROOT, "common", "include")
OUT_DIR = os.path.join(os.environ.get("TEMP", os.path.join(ROOT, "_tmp")), "cuda40_build")
os.makedirs(OUT_DIR, exist_ok=True)

# 需额外链接库/选项的特殊项目
EXTRA_ARGS = {
    "day12_dynamic_parallel": ["-rdc=true", "-lcudadevrt"],  # 动态并行：可重定位设备代码
    "day13_cublas": ["-lcublas"],                            # cuBLAS 库
    "day14_cufft": ["-lcufft"],                              # cuFFT 库
    "day32_cusparse": ["-lcusparse"],                        # cuSPARSE 稀疏矩阵库
}

# 不需要独立编译的 day（依赖外部库或没有 main() 入口）
SKIP_DAYS = {
    "day36_pybind11_cuda",      # 需要 pybind11 头文件，只能通过 CMake 编译
    "day37_ctypes_cuda",        # 共享库（无 main()），只能通过 CMake 编译
    "day40_pytorch_cuda_extension",  # 需要 PyTorch 头文件，只能通过 setup.py 编译
}

# 需要特殊编译选项的 day
SPECIAL_OPTIONS = {
    "day31_flash_attention": ["-O1"],  # LLVM 后端在 -O3 下内存不足，降为 -O1
}

# 额外需要编译的路径（非 day*/src/main.cu 格式）
EXTRA_SOURCES = [
    {
        "name": "common_cuda_kernels_demo",
        "src": os.path.join(ROOT, "common", "cuda_kernels", "demo.cu"),
        "inc": os.path.join(ROOT, "common", "cuda_kernels"),
    },
]


def find_msvc_bin():
    """在 Windows 上定位 MSVC 的 cl.exe 所在目录，返回 None 表示未找到。"""
    patterns = [
        r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64",
        r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64",
    ]
    for p in patterns:
        hits = glob.glob(os.path.join(p, "cl.exe"))
        if hits:
            return os.path.dirname(sorted(hits)[-1])  # 取版本最高的
    return None


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else "sm_120"
    env = os.environ.copy()

    # Windows 下把 MSVC 加入 PATH，供 nvcc 调用主机编译器
    if os.name == "nt":
        msvc = find_msvc_bin()
        if msvc:
            env["PATH"] = msvc + os.pathsep + env["PATH"]
            print("[环境] MSVC:", msvc)
        else:
            print("[警告] 未找到 MSVC，请确认已安装 Visual Studio 2022")

    print("[环境] 目标架构:", arch)
    print("=" * 60)

    days = sorted(glob.glob(os.path.join(ROOT, "day*")))
    ok, fail, skip = [], [], []
    for day in days:
        name = os.path.basename(day)
        src = os.path.join(day, "src", "main.cu")
        if not os.path.exists(src):
            continue
        if name in SKIP_DAYS:
            print("  [SKIP] %s (需要 CMake 编译)" % name)
            skip.append(name)
            continue
        out = os.path.join(OUT_DIR, name + (".exe" if os.name == "nt" else ""))
        cmd = ["nvcc", "-arch=" + arch, "-I", COMMON_INC, src, "-o", out]
        # Windows/MSVC：用 /utf-8 消除中文源码的 C4819 警告
        if os.name == "nt":
            cmd += ["-Xcompiler", "/utf-8"]
        # Day 39: NCCL 在 Windows 上不可用，定义 HAS_NCCL=0
        if name == "day39_nccl_allreduce":
            cmd += ["-DHAS_NCCL=0"]
        cmd += EXTRA_ARGS.get(name, [])
        # 特殊编译选项（如降低优化等级避免 LLVM OOM）
        cmd += SPECIAL_OPTIONS.get(name, [])

        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode == 0 and os.path.exists(out):
            print("  [OK]   %s" % name)
            ok.append(name)
        else:
            print("  [FAIL] %s" % name)
            for line in (r.stderr or "").splitlines():
                if "error" in line.lower():
                    print("         " + line.strip())
            fail.append(name)

    # 编译额外源文件（如 common/cuda_kernels/demo.cu）
    for extra in EXTRA_SOURCES:
        if not os.path.exists(extra["src"]):
            continue
        out = os.path.join(OUT_DIR, extra["name"] + (".exe" if os.name == "nt" else ""))
        cmd = ["nvcc", "-arch=" + arch, "-I", COMMON_INC,
               "-I", extra["inc"], extra["src"], "-o", out]
        if os.name == "nt":
            cmd += ["-Xcompiler", "/utf-8"]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode == 0 and os.path.exists(out):
            print("  [OK]   %s" % extra["name"])
            ok.append(extra["name"])
        else:
            print("  [FAIL] %s" % extra["name"])
            for line in (r.stderr or "").splitlines():
                if "error" in line.lower():
                    print("         " + line.strip())
            fail.append(extra["name"])

    print("=" * 60)
    print("编译成功: %d / %d (跳过: %d)" % (len(ok), len(ok) + len(fail), len(skip)))
    if fail:
        print("失败项目:", ", ".join(fail))
        sys.exit(1)
    print("全部编译通过！")


if __name__ == "__main__":
    main()