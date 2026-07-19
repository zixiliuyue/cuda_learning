#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 65: TensorRT 深入 — ONNX -> Engine -> Plugin

接续 Day24（TensorRT 概念）。本天聚焦「完整部署链路」的原理与可跑的第一环：

    PyTorch 模型
        │  torch.onnx.export（本天真实可跑）
        ▼
    ONNX 计算图（跨框架中间表示）
        │  TensorRT Builder + OnnxParser
        ▼
    TensorRT Engine（针对具体 GPU 优化的序列化推理引擎）
        │  自定义 Plugin（IPluginV2 / IPluginV2DynamicExt）
        ▼
    高性能推理（层融合 / FP16-INT8 / kernel autotuning）

设计原则（严格遵守「不报错」）：
  - torch.onnx.export 真实可跑（不依赖 GPU，也不强制需要 onnx 包）；
  - 若探测到 tensorrt：真实构建一个最小 Engine 并做一次推理；
  - 若未装 tensorrt（最常见）：以「原理 + 代码骨架」讲解
        ONNX -> Engine 构建流程 与 自定义 Plugin 机制，绝不报错。
所有分支 exit code 0。
"""

import os
import sys
import tempfile
import numpy as np

# ------------------------------------------------------------
# 环境探测三布尔量
# ------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

CUDA_AVAILABLE = TORCH_AVAILABLE and torch.cuda.is_available()

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

SEP = "=" * 64


# ============================================================
#  一个可导出的小模型（Conv-BN-ReLU + 全局池化 + 分类头）
# ============================================================
if TORCH_AVAILABLE:

    class SmallCNN(nn.Module):
        """典型 CNN 部署单元：Conv+BN+ReLU 可被 TensorRT 融合成一个 kernel。"""

        def __init__(self, in_ch=3, n_classes=10):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, n_classes)

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.pool(x).flatten(1)
            return self.fc(x)


# ============================================================
#  第 1 环（真实可跑）：PyTorch -> ONNX 导出
# ============================================================
def export_onnx(onnx_path):
    """真实执行 torch.onnx.export，返回是否成功。

    torch.onnx.export 会自己把计算图写成 ONNX 协议文件，
    基础导出不强制依赖 onnx 这个 python 包，也不需要 GPU。
    """
    model = SmallCNN().eval()
    dummy = torch.randn(1, 3, 32, 32)  # 示例输入，用于 trace 出计算图
    try:
        # dynamo=False 强制使用经典 TorchScript 导出器：直接写 ONNX protobuf，
        # 不依赖 onnxscript，兼容性最好（新版 torch 默认走 dynamo 需额外依赖）。
        export_kwargs = dict(
            input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=13,
        )
        try:
            torch.onnx.export(model, dummy, onnx_path, dynamo=False, **export_kwargs)
        except TypeError:
            # 旧版 torch 没有 dynamo 参数，直接调用即可
            torch.onnx.export(model, dummy, onnx_path, **export_kwargs)
        ok = os.path.exists(onnx_path) and os.path.getsize(onnx_path) > 0
        return ok
    except Exception as e:  # 极少数环境导出器缺依赖时，讲原理而非报错
        print(f"  [提示] torch.onnx.export 未能完成（{type(e).__name__}: {e}）")
        print("         这不影响理解链路，下面继续讲解 ONNX 的作用与结构。")
        return False


def inspect_onnx(onnx_path):
    """若安装了 onnx 包，读回图结构打印节点；否则只报告文件大小。"""
    try:
        import onnx
        m = onnx.load(onnx_path)
        onnx.checker.check_model(m)
        nodes = [n.op_type for n in m.graph.node]
        print(f"  [ONNX] 校验通过，共 {len(nodes)} 个节点：")
        print(f"         {nodes}")
        print("  [ONNX] 计算图是跨框架中间表示，TensorRT/ONNXRuntime 等都能解析。")
    except ImportError:
        size_kb = os.path.getsize(onnx_path) / 1024
        print(f"  [ONNX] 已导出文件（{size_kb:.1f} KB）。未装 onnx 包，跳过图结构解析。")
        print("         安装 `pip install onnx` 可用 onnx.checker 校验并查看节点。")
    except Exception as e:
        print(f"  [ONNX] 读取校验时出现问题（{type(e).__name__}），不影响链路讲解。")


# ============================================================
#  第 2 环：ONNX -> TensorRT Engine
# ============================================================
def build_engine_real(onnx_path):
    """探测到 tensorrt 时，真实构建一个最小 Engine（需要 GPU + TensorRT）。"""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("  [TensorRT] ONNX 解析失败：")
            for i in range(parser.num_errors):
                print("   ", parser.get_error(i))
            return None
    config = builder.create_builder_config()
    # 允许 FP16，让 TensorRT 自动为支持的层选择半精度 kernel
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    print(f"  [TensorRT] 版本 {trt.__version__}，开始构建 Engine ...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("  [TensorRT] Engine 构建失败。")
        return None
    print(f"  [TensorRT] Engine 构建成功，序列化大小 = {len(serialized)/1024:.1f} KB")
    return serialized


def explain_engine_skeleton():
    """未装 tensorrt 时，讲解 ONNX->Engine 的构建流程 + 代码骨架。"""
    print("  未检测到 tensorrt 包（最常见情况）。以下为构建 Engine 的标准骨架：")
    print("""
    import tensorrt as trt
    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser  = trt.OnnxParser(network, logger)
    with open("model.onnx", "rb") as f:
        parser.parse(f.read())                 # ONNX -> TensorRT 网络定义

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)      # 可选：开启 FP16
    # config.set_flag(trt.BuilderFlag.INT8)    # INT8 需配 IInt8Calibrator 校准
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1,3,32,32), (8,3,32,32), (32,3,32,32))  # 动态 shape
    config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    with open("model.engine", "wb") as f:      # 序列化保存，部署时反序列化直接用
        f.write(engine)
""")
    print("  构建期 TensorRT 会做：层融合(Conv+BN+ReLU→1 kernel)、精度选择、")
    print("  kernel autotuning（对当前 GPU 实测选最快 kernel）、显存复用规划。")


# ============================================================
#  第 3 环：自定义 Plugin 机制（接 Day24）
# ============================================================
def explain_plugin():
    print(SEP)
    print("第 3 环：自定义 Plugin（当 ONNX 算子 TensorRT 不支持时）")
    print(SEP)
    print("""
  为什么需要 Plugin？
    TensorRT 内置算子有限。遇到自定义/新算子（如某种特殊 Attention、
    可变形卷积、RoPE 等），需要用 Plugin 提供 GPU 实现，插入到 Engine 中。

  Plugin 类的核心接口（IPluginV2DynamicExt，对应 C++/Python）：
    - getOutputDimensions()   : 声明输出张量形状（支持动态 shape）
    - supportsFormatCombination(): 声明支持的数据类型/内存布局(FP32/FP16/INT8)
    - enqueue()               : 真正的 CUDA kernel 调用（你写的 __global__ 在这里跑）
    - getWorkspaceSize()      : 声明需要的临时显存
    - serialize()/deserialize(): 引擎存/取时保存 plugin 参数
    - clone()                 : 复制 plugin 实例

  注册与使用流程：
    1. 写 CUDA kernel（Day1-40 的功底），封装进 enqueue()
    2. 实现 IPluginV2DynamicExt + IPluginCreator
    3. REGISTER_TENSORRT_PLUGIN(MyPluginCreator) 注册到全局 registry
    4. 在 ONNX 里用同名自定义算子占位，Parser 解析时自动匹配到你的 Plugin

  与 Day24 的联系：Day24 建立 TensorRT 整体认知；本天补齐「不支持的算子
  如何用自己手写的 CUDA kernel 接进 TensorRT」这一关键工程能力。
""")


def print_pipeline_diagram():
    print(SEP)
    print("完整部署链路图")
    print(SEP)
    print("""
    ┌─────────────┐  torch.onnx.export   ┌─────────────┐
    │ PyTorch 模型 │ ───────────────────> │  ONNX 计算图 │
    └─────────────┘   (本天真实可跑)      └──────┬──────┘
                                                 │ Builder + OnnxParser
                                                 ▼
    ┌──────────────────┐   自定义 Plugin   ┌─────────────────┐
    │   高性能推理      │ <──────────────── │ TensorRT Engine │
    │ (FP16/INT8/融合)  │  (不支持的算子)   │  (.engine 文件) │
    └──────────────────┘                   └─────────────────┘

    构建期优化：层融合 · 精度校准 · kernel autotuning · 显存复用
    运行期特性：序列化 Engine 免重复构建 · 动态 shape · 多流并发
""")


# ============================================================
#  主流程
# ============================================================
def main():
    print()
    print(SEP)
    print("Day 65: TensorRT 深入 — ONNX -> Engine -> Plugin")
    print(SEP)

    print_pipeline_diagram()

    # ---- 第 1 环：ONNX 导出（真实） ----
    print(SEP)
    print("第 1 环：PyTorch -> ONNX 导出（真实执行）")
    print(SEP)
    onnx_ok = False
    onnx_path = os.path.join(tempfile.gettempdir(), "day65_smallcnn.onnx")
    if TORCH_AVAILABLE:
        print(f"  device 探测：CUDA {'可用' if CUDA_AVAILABLE else '不可用(用 CPU 导出即可)'}")
        onnx_ok = export_onnx(onnx_path)
        if onnx_ok:
            print(f"  [ONNX] 已导出到: {onnx_path}")
            inspect_onnx(onnx_path)
    else:
        print("  未安装 PyTorch，跳过真实导出。ONNX 是跨框架中间表示，")
        print("  典型导出：torch.onnx.export(model, dummy_input, 'model.onnx', ...)")

    # ---- 第 2 环：ONNX -> Engine ----
    print()
    print(SEP)
    print("第 2 环：ONNX -> TensorRT Engine")
    print(SEP)
    if TENSORRT_AVAILABLE and onnx_ok and CUDA_AVAILABLE:
        build_engine_real(onnx_path)
    else:
        reason = []
        if not TENSORRT_AVAILABLE:
            reason.append("未装 tensorrt")
        if not CUDA_AVAILABLE:
            reason.append("无 CUDA GPU")
        if not onnx_ok:
            reason.append("无 ONNX 文件")
        print(f"  当前环境（{', '.join(reason)}）以原理+骨架讲解：")
        explain_engine_skeleton()

    # ---- 第 3 环：Plugin ----
    print()
    explain_plugin()

    # ---- 清理临时文件 ----
    if onnx_ok and os.path.exists(onnx_path):
        try:
            os.remove(onnx_path)
        except OSError:
            pass

    # ---- 总结 ----
    print(SEP)
    print("知识点总结")
    print(SEP)
    print("""  1. 部署链路：PyTorch -> ONNX -> TensorRT Engine -> (Plugin) -> 推理。
  2. ONNX：跨框架中间表示，torch.onnx.export 可直接产出，支持动态 axes。
  3. Builder+OnnxParser：把 ONNX 解析成网络定义，构建期做层融合与 autotuning。
  4. 精度：FP16 一个 flag 即可；INT8 需 IInt8Calibrator 提供校准数据集。
  5. Plugin：内置算子不够时，用手写 CUDA kernel 实现 IPluginV2DynamicExt 接入。
  6. Engine 可序列化保存，部署时反序列化直接推理，免去重复构建开销。""")
    print(SEP)

    if TENSORRT_AVAILABLE:
        print(f"环境：TensorRT {trt.__version__} 已就绪。")
    elif TORCH_AVAILABLE:
        print("环境：有 PyTorch、无 TensorRT。已真实导出 ONNX + 原理讲解 Engine/Plugin。")
        print("提示：Linux + NVIDIA GPU 上 `pip install tensorrt onnx` 可跑完整链路。")
    else:
        print("环境：无 PyTorch/TensorRT。已以原理+骨架讲解完整部署链路。")
    print(SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
