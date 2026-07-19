#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 90: 总收官 — AI Infra 能力图谱 + 项目梳理

90 天的终点，也是新起点。本天不引入新知识，而是把 90 天（13 个阶段）的
能力系统性收束成一张「AI Infra 能力图谱」，并对照三个真实目标岗位，帮你：

  1. 回顾 13 个阶段各自的核心能力与代表产出；
  2. 对照『训练优化 / 端侧部署 / 推理服务化』三个岗位 JD，看你的技能覆盖；
  3. 拿到一份『简历级项目清单』，把散落的 Day 产出组织成可写进简历的项目；
  4. 明确后续进阶学习路线。

本天为纯 Python 打印，任意环境稳定运行、结果可复现。
"""

# ------------------------------------------------------------
# 环境探测（仅用于收尾打印，一切逻辑不依赖）
# ------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

SEP = "=" * 64


# ============================================================
#  1. 90 天全景回顾（13 个阶段）
# ============================================================
def print_roadmap():
    print(SEP)
    print("90 天全景回顾 — 13 个阶段能力图谱")
    print(SEP)
    stages = [
        ("一", "Day 1-5", "CUDA 基础", "执行/内存模型、错误处理、设备查询"),
        ("二", "Day 6-12", "内存优化", "共享内存/合并访问/占用率/归约/流"),
        ("三", "Day 13-20", "算法实战", "cuBLAS/FFT/排序/Scan/BFS/蒙特卡洛/调优"),
        ("四", "Day 21-26", "框架集成", "PyTorch扩展/Numba/OpenCV/TensorRT/多GPU/AMP"),
        ("五", "Day 27-30", "综合项目", "N-Body/光追/推理引擎/阶段总结"),
        ("六", "Day 31-35", "进阶专题", "Flash Attention/cuSPARSE/IPC/内存池/Nsight"),
        ("七", "Day 36-38", "Python 封装", "pybind11/ctypes/Cython 三种路线"),
        ("八", "Day 39-40", "扩展实战", "NCCL AllReduce / PyTorch CUDA 扩展"),
        ("九", "Day 41-47", "Triton 算子", "融合算子/自动调优/Flash Attn/接入PyTorch"),
        ("十", "Day 48-58", "LLM 推理优化", "KV Cache/PagedAttn/批处理/采样/投机/服务化"),
        ("十一", "Day 59-66", "模型压缩部署", "量化(PTQ/QAT/GPTQ)/剪枝/蒸馏/TensorRT"),
        ("十二", "Day 67-80", "分布式训练", "DDP/ZeRO/TP/PP/MoE/3D并行/FSDP/profiling"),
        ("十三", "Day 81-90", "通信·存储·工程化", "NCCL算法/RDMA/CUTLASS/Checkpoint/GDS/排查"),
    ]
    print(f"  {'阶段':<5}{'天数':<11}{'主题':<14}{'核心能力'}")
    print("  " + "-" * 72)
    for no, days, topic, cap in stages:
        print(f"  第{no:<3}{days:<11}{topic:<14}{cap}")
    print()
    print("  能力主线：能跑通 → 跑得快 → 会用库 → 融入框架 → 做项目 →")
    print("            前沿专题 → 交付Python → 分布式/推理/压缩 → 通信存储工程化。")


# ============================================================
#  2. 三个目标岗位 JD 能力对照
# ============================================================
def print_jd_mapping():
    print()
    print(SEP)
    print("三个目标岗位 JD × 你的技能覆盖")
    print(SEP)

    jds = {
        "JD1 训练优化工程师": [
            ("分布式并行(DP/TP/PP/MoE)", "Day 67-74"),
            ("显存优化(ZeRO/梯度检查点)", "Day 69/75"),
            ("通信优化(NCCL/overlap)", "Day 39/78/81"),
            ("混合精度训练(AMP)", "Day 26"),
            ("Profiling 与瓶颈定位", "Day 35/79"),
            ("自定义高性能算子(CUDA/Triton)", "Day 40-47"),
        ],
        "JD2 模型端侧部署工程师": [
            ("量化(PTQ/QAT/GPTQ/AWQ)", "Day 59-62"),
            ("剪枝 / 知识蒸馏", "Day 63/64"),
            ("TensorRT(ONNX/Engine/Plugin)", "Day 24/65/66"),
            ("推理算子融合", "Day 43-45/54"),
            ("精度对齐与验证", "Day 60/66"),
            ("端侧/国产芯片适配", "Day 89"),
        ],
        "JD3 推理服务化工程师": [
            ("KV Cache / PagedAttention", "Day 49/50"),
            ("Continuous/Dynamic Batching", "Day 51/56"),
            ("采样与解码策略", "Day 52"),
            ("投机解码", "Day 55"),
            ("Triton 融合算子", "Day 41-47"),
            ("vLLM/TensorRT-LLM 架构", "Day 57/58"),
        ],
    }
    for jd, items in jds.items():
        print(f"\n  ▶ {jd}")
        print("    " + "-" * 56)
        for skill, days in items:
            print(f"    [✓] {skill:<34}{days}")
    print()
    print("  → 三个岗位的技能几乎全部被 90 天课程覆盖，且有大量交叉复用")
    print("    (Triton 算子/Flash Attention/量化 三个岗位都吃得到)。")


# ============================================================
#  3. 简历级项目清单
# ============================================================
def print_resume_projects():
    print()
    print(SEP)
    print("简历级项目清单 — 把散落的 Day 产出组织成可写进简历的项目")
    print(SEP)
    projects = [
        ("高性能 CUDA 算子库",
         "Day 6/9/31/40",
         "手写并优化 Tiled GEMM / 归约 / Flash Attention / LayerNorm，"
         "含 shared memory、合并访问、warp-shuffle 优化，PyTorch 扩展交付。"),
        ("Triton 融合算子集",
         "Day 41-47",
         "用 Triton 实现 Fused Softmax/LayerNorm/Flash Attention/MLP，"
         "@autotune 自动调优，接入 torch.compile，对比 cuBLAS/手写 CUDA。"),
        ("迷你 LLM 推理引擎",
         "Day 48-58",
         "整合 KV Cache + PagedAttention + Continuous Batching + 多种采样 + "
         "投机解码，端到端推理引擎并产出吞吐/延迟性能报告。"),
        ("模型压缩与端侧部署链路",
         "Day 59-66",
         "PTQ/QAT 量化 + 剪枝 + 蒸馏，ONNX→TensorRT Engine(FP16/INT8)，"
         "自定义 Plugin，完整精度对齐验证。"),
        ("分布式训练框架实践",
         "Day 67-80",
         "DDP/ZeRO/TP/PP/MoE/3D 并行的最小可跑实现 + 梯度检查点 + "
         "通信计算重叠 + profiling 体系，多卡训练小模型扩展性报告。"),
        ("AI Infra 工程化工具箱",
         "Day 81-90",
         "Ring/Tree AllReduce 模拟、Checkpoint 优化、GDS IO 路径分析、"
         "NaN 钩子/OOM 排查工具、迷你 autograd 引擎。"),
    ]
    for i, (name, days, desc) in enumerate(projects, 1):
        print(f"\n  {i}. {name}  ({days})")
        print(f"     {desc}")
    print()
    print("  写简历建议：每个项目突出『量化指标』(加速比/显存节省/吞吐提升)，")
    print("             并说明你用到的具体优化手段与遇到的问题如何定位解决。")


# ============================================================
#  4. 进阶学习路线
# ============================================================
def print_next_steps():
    print()
    print(SEP)
    print("结课之后 — 进阶学习路线")
    print(SEP)
    print("""
  1) 深入源码：读 vLLM / TensorRT-LLM / Megatron-LM / DeepSpeed 核心模块，
     从『会用』到『能改』。

  2) 真实硬件实战：在多卡/多机环境跑通真正的分布式训练与 RDMA/NVLink，
     把本课程的『单机模拟』升级为『真集群』。

  3) 前沿算子：跟进 FlashAttention-3 / FP8 训练 / MoE kernel / 
     CUTLASS 3.x，持续打磨手写高性能算子的能力。

  4) 系统工程：Kubernetes + GPU 调度、弹性训练、故障恢复、
     可观测性(metrics/tracing)，向平台级 AI Infra 演进。

  5) 参与开源：给主流推理/训练框架提 PR，是最快的成长与背书方式。

  记住：AI Infra 的核心是『让算力被高效地用起来』——
        算子、并行、通信、存储、工程化，每一环都是杠杆。
""")


# ============================================================
#  主流程
# ============================================================
def main():
    print(SEP)
    print(" " * 18 + "CUDA / AI Infra 90 天 · 结课收官")
    print(SEP)
    print_roadmap()
    print_jd_mapping()
    print_resume_projects()
    print_next_steps()

    print(SEP)
    if TORCH_AVAILABLE:
        dev = "CUDA" if torch.cuda.is_available() else "CPU"
        print(f"环境：PyTorch 就绪（{dev}）。")
    else:
        print("环境：未安装 PyTorch（本天为纯梳理，不影响）。")
    print("恭喜完成 90 天 AI Infra 专精之路 —— 这是终点，更是新的起点。🎉")
    print(SEP)


if __name__ == "__main__":
    main()
