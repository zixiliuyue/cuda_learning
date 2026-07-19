#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 89: 国产 AI 芯片适配概览 — 昇腾 / 寒武纪 / 壁仞

CUDA 生态是 NVIDIA 的护城河，但 AI Infra 工程师越来越常遇到国产算力适配的
需求。本天以「原理为主」梳理三家主流国产 AI 芯片的架构、编程模型，以及从
CUDA 生态迁移过来的典型路径。

本天不依赖任何国产芯片 SDK（多为 Linux + 特定硬件），全部是**可在任意环境
运行的知识梳理 + 对比表 + 迁移思路打印**，帮你建立整体认知地图：

  1. 三家芯片的架构与编程模型对比（对标 CUDA 的各层）；
  2. CUDA 生态各组件 → 国产对应物的映射表（cuda/cudnn/cublas/nccl...）；
  3. 一段「CUDA kernel → 昇腾 CANN / 寒武纪 BANG」的迁移思路示意；
  4. 迁移策略选择：算子级迁移 vs 图级迁移 vs 框架后端接入。
"""

# ------------------------------------------------------------
# 环境探测（仅用于打印信息；本天为知识梳理，任意环境可运行）
# ------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# 探测是否装了国产框架（几乎总是 False，仅作友好提示）
TORCH_NPU_AVAILABLE = False
try:
    import torch_npu  # noqa: F401  昇腾 PyTorch 适配层
    TORCH_NPU_AVAILABLE = True
except Exception:
    TORCH_NPU_AVAILABLE = False

SEP = "=" * 64


# ============================================================
#  1. 三家芯片架构与编程模型
# ============================================================
def print_chip_overview():
    print(SEP)
    print("国产 AI 芯片概览：昇腾 / 寒武纪 / 壁仞")
    print(SEP)
    print("""
  厂商/芯片        华为昇腾(Ascend)      寒武纪(Cambricon)     壁仞(Biren)
  ------------------------------------------------------------------------
  代表产品        Ascend 910/310        MLU 370/590           BR100/BR104
  架构            达芬奇(Da Vinci)       MLUarch(私有)         通用 GPGPU 架构
                  Cube 矩阵计算单元      张量/向量/标量单元      SIMT + 张量核
  软件栈          CANN                   Cambricon Neuware     壁仞软件栈(类CUDA)
  编程语言        AscendC / TBE(DSL)     BANG C(类CUDA C)      类 CUDA C
  运行时/驱动     AscendCL(ACL)          CNRT / CNRT-Runtime   BIRENSUPA
  框架适配        torch_npu / MindSpore  Cambricon PyTorch     壁仞 PyTorch
  编程范式        算子下沉+图编译为主     核函数(kernel)为主      SIMT, 迁移成本低
  生态定位        自研全栈, 训推兼顾      推理为主, 训练扩展      对标 NVGPU, 兼容优先
""")
    print("  一句话记忆：")
    print("   - 昇腾：达芬奇架构 + CANN 全栈，编程偏『算子 + 图编译』，自成体系；")
    print("   - 寒武纪：BANG C 语法接近 CUDA C，写 kernel 的手感最像 CUDA；")
    print("   - 壁仞：通用 GPGPU + 类 CUDA 编程模型，主打『迁移成本低』。")


# ============================================================
#  2. CUDA 生态 → 国产对应物映射
# ============================================================
def print_ecosystem_mapping():
    print()
    print(SEP)
    print("CUDA 生态组件 → 国产芯片对应物（迁移查表）")
    print(SEP)
    rows = [
        ("编程语言", "CUDA C/C++", "AscendC", "BANG C", "类 CUDA C"),
        ("运行时 API", "CUDA Runtime", "AscendCL(ACL)", "CNRT", "BIRENSUPA RT"),
        ("BLAS 库", "cuBLAS", "CANN 内置算子", "CNML/CNNL", "壁仞 BLAS"),
        ("DNN 库", "cuDNN", "CANN 算子库", "CNNL", "壁仞 DNN"),
        ("集合通信", "NCCL", "HCCL", "CNCL", "壁仞 CCL"),
        ("图/编译", "TensorRT", "CANN 图引擎(GE)", "MagicMind", "壁仞图编译"),
        ("框架后端", "torch.cuda", "torch_npu", "Cambricon PyTorch", "壁仞 PyTorch"),
    ]
    print(f"  {'类别':<12}{'NVIDIA/CUDA':<16}{'昇腾':<18}{'寒武纪':<18}{'壁仞'}")
    print("  " + "-" * 74)
    for cat, cuda, asc, camb, biren in rows:
        print(f"  {cat:<12}{cuda:<16}{asc:<18}{camb:<18}{biren}")
    print()
    print("  → 迁移时先在此表定位『你依赖的 CUDA 组件』，再找对应替代物。")
    print("    通信(NCCL→HCCL/CNCL)与编译(TensorRT→GE/MagicMind)是重灾区。")


# ============================================================
#  3. CUDA kernel → 国产迁移思路示意（伪代码对照）
# ============================================================
def print_migration_example():
    print()
    print(SEP)
    print("迁移示意：一个 element-wise add，从 CUDA 到昇腾/寒武纪")
    print(SEP)
    print("""
  【CUDA C】每线程处理一个元素
    __global__ void add(const float* a, const float* b, float* c, int n){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) c[i] = a[i] + b[i];
    }
    add<<<grid, block>>>(a, b, c, n);

  【寒武纪 BANG C】语法最接近 CUDA，仍是写 kernel + 手动搬运
    __mlu_global__ void add(float* a, float* b, float* c, int n){
        // 用 __memcpy 把数据从 GDRAM 搬到 NRAM(片上), 再向量化计算
        __memcpy(nram_a, a, n*sizeof(float), GDRAM2NRAM);
        __bang_add(nram_c, nram_a, nram_b, n);   // 向量加法内建函数
        __memcpy(c, nram_c, n*sizeof(float), NRAM2GDRAM);
    }
    // 差异：显式片上内存(NRAM)管理 + __bang_* 向量内建, 思路仍是 SIMD/SIMT

  【昇腾 AscendC】偏『算子 + 数据流水』，用 Tiling + Queue 组织
    class AddKernel {
        void Process(){
            CopyIn();     // GM(全局) -> 片上 Local buffer, 通过 Que 流水
            Compute();    // Add(zLocal, xLocal, yLocal, tileLen)
            CopyOut();    // 片上 -> GM
        }
    };
    // 差异：编程模型从『线程』上升到『数据分块(Tiling)+多级缓冲流水』

  迁移要点：
   - 计算逻辑(加法)不变，变的是『内存层次管理』与『并行组织方式』；
   - CUDA 的 shared memory ≈ 寒武纪 NRAM ≈ 昇腾 Local Buffer；
   - 越靠近『图/框架层』迁移越省力（改后端），越靠近『手写 kernel』越费力。
""")


# ============================================================
#  4. 迁移策略：三个层次
# ============================================================
def print_migration_strategy():
    print(SEP)
    print("迁移策略选择：从省力到费力的三条路")
    print(SEP)
    print("""
  层次            做法                              工作量    可迁移覆盖率
  --------------------------------------------------------------------------
  ① 框架后端级    换设备后端(torch.cuda->torch_npu)  最小      高(标准算子)
                  模型代码几乎不改, .to('npu')
  ② 图/编译级     导出 ONNX -> 国产图编译器           中        中高
                  (CANN GE / MagicMind) 生成引擎
  ③ 算子级        自定义/非标准算子逐个重写           大        取决于算子数
                  (BANG C / AscendC 手写 kernel)

  实战建议：
   - 先跑『框架后端级』，让 90% 标准算子直接跑起来；
   - 剩下的自定义算子(如你手写的 Triton/CUDA 融合算子)才需要算子级重写；
   - 通信(NCCL->HCCL/CNCL)与混合精度策略往往需要额外适配与调优；
   - 精度对齐(算子实现差异/累加顺序)是迁移后必做的验证环节。
""")
    print("  与本课程的连接：")
    print("   - Day87 讲的 dispatcher 正是『框架后端级』迁移的技术底座——")
    print("     国产后端本质是注册一套新的 dispatch key + kernel；")
    print("   - 你在 Day41-47 写的 Triton / Day40 的 CUDA 融合算子，")
    print("     属于『算子级』迁移中最需要人工重写的部分。")


# ============================================================
#  主流程
# ============================================================
def main():
    print_chip_overview()
    print_ecosystem_mapping()
    print_migration_example()
    print_migration_strategy()

    print()
    print(SEP)
    if TORCH_NPU_AVAILABLE:
        print("检测到 torch_npu（昇腾适配层），可在昇腾设备上做真实迁移实验。")
    elif TORCH_AVAILABLE:
        dev = "CUDA" if torch.cuda.is_available() else "CPU"
        print(f"环境：PyTorch 就绪（{dev}）。本天为知识梳理，任意环境均可运行。")
    else:
        print("环境：未安装 PyTorch。本天为纯知识梳理，不依赖任何 SDK。")
    print("说明：国产芯片 SDK 多为 Linux + 特定硬件，本天以原理与迁移思路为主。")
    print(SEP)


if __name__ == "__main__":
    main()
