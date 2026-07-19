# Day 65: TensorRT 深入 — ONNX → Engine → Plugin

> 🚀 **第十一阶段（模型压缩与部署）**。接续 [Day 24](../day24_tensorrt/) 的 TensorRT 概念，本天打通**完整部署链路**：PyTorch → ONNX → TensorRT Engine →（自定义 Plugin）→ 高性能推理，并补齐「不支持的算子如何用手写 CUDA kernel 接入」这一关键工程能力。

## 学习目标

1. **打通** PyTorch → ONNX → TensorRT Engine 的完整链路认知
2. **真实导出** ONNX（`torch.onnx.export`，含动态 batch 维）
3. **理解** Builder + OnnxParser 构建 Engine 的流程与构建期优化
4. **掌握** 自定义 Plugin（IPluginV2DynamicExt）的接口与注册流程
5. **区分** FP16（一个 flag）与 INT8（需校准器）的开启方式

---

## 前置依赖

- **Day 24 TensorRT 概念**：本天在其基础上深入到 ONNX/Engine/Plugin
- **Day 1-40 CUDA kernel**：Plugin 的 `enqueue()` 里跑的就是你手写的 `__global__`
- **Day 59-62 量化**：INT8 Engine 的校准思想与之相通

---

## 核心概念

### 完整部署链路

```
┌─────────────┐  torch.onnx.export   ┌─────────────┐
│ PyTorch 模型 │ ───────────────────> │  ONNX 计算图 │
└─────────────┘   (本天真实可跑)      └──────┬──────┘
                                             │ Builder + OnnxParser
                                             ▼
┌──────────────────┐   自定义 Plugin   ┌─────────────────┐
│   高性能推理      │ <──────────────── │ TensorRT Engine │
│ (FP16/INT8/融合)  │  (不支持的算子)   │  (.engine 文件) │
└──────────────────┘                   └─────────────────┘
```

### 第 1 环：PyTorch → ONNX

ONNX 是跨框架**中间表示**，TensorRT / ONNXRuntime / OpenVINO 等都能解析。`dynamic_axes` 声明动态维（如 batch），部署时可变 batch：

```python
torch.onnx.export(model, dummy, "model.onnx",
                  input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                  opset_version=13)
```

> 本天脚本用 `dynamo=False` 走经典 TorchScript 导出器，兼容性最好。真实导出需 `pip install onnx`。

### 第 2 环：ONNX → Engine

Builder 解析 ONNX 得到网络定义，构建期做四类优化：

- **层融合**：Conv+BN+ReLU → 1 个 kernel，减少访存与启动开销
- **精度选择**：`FP16` 一个 flag；`INT8` 需 `IInt8Calibrator` 提供校准数据
- **Kernel autotuning**：对当前 GPU 实测多种 kernel，选最快的
- **显存复用规划**：分析生命周期，复用中间张量显存

Engine 可**序列化保存**为 `.engine`，部署时反序列化直接推理，免去重复构建。

### 第 3 环：自定义 Plugin

当 ONNX 里有 TensorRT 内置算子不支持的算子（特殊 Attention、可变形卷积、RoPE…），用 Plugin 提供 GPU 实现。核心接口（`IPluginV2DynamicExt`）：

| 接口 | 作用 |
|------|------|
| `getOutputDimensions()` | 声明输出形状（支持动态 shape） |
| `supportsFormatCombination()` | 声明支持的 dtype/布局（FP32/FP16/INT8） |
| `enqueue()` | **真正调用你手写的 CUDA kernel** |
| `getWorkspaceSize()` | 声明临时显存需求 |
| `serialize()/deserialize()` | Engine 存取时保存 plugin 参数 |
| `clone()` | 复制 plugin 实例 |

注册流程：写 CUDA kernel → 实现 `IPluginV2DynamicExt` + `IPluginCreator` → `REGISTER_TENSORRT_PLUGIN` 注册 → ONNX 用同名算子占位，Parser 自动匹配。

---

## 文件结构

```
day65_tensorrt_deep/
├── python/tensorrt_deep.py   # ONNX 导出(真实) + Engine/Plugin 原理与骨架
├── build.sh                  # Linux/WSL2 运行脚本
├── build.bat                 # Windows 运行脚本
└── README.md                 # 本文件
```

> **设计**：ONNX 导出真实执行（不需 GPU）；探测到 `tensorrt` 则真实构建 Engine，否则以「原理 + 代码骨架」讲解，**绝不报错**。缺 onnx 包时导出环节会给出安装提示并继续讲解，exit code 仍为 0。

---

## 编译与运行

### Linux / WSL2（完整链路）

```bash
cd day65_tensorrt_deep
pip install onnx                 # ONNX 导出与校验
pip install tensorrt             # 需 NVIDIA GPU，可选
python python/tensorrt_deep.py
```

### Windows

```powershell
cd day65_tensorrt_deep
.\build.bat
```

> TensorRT 通常未安装：脚本会打印完整链路图、ONNX 导出、Engine/Plugin 骨架，全程 exit code 0。

---

## 知识点总结

1. **部署链路**：PyTorch → ONNX → TensorRT Engine →（Plugin）→ 推理。
2. **ONNX**：跨框架中间表示，`torch.onnx.export` 可直接产出，支持动态 axes。
3. **Builder+OnnxParser**：解析 ONNX 为网络定义，构建期做层融合与 autotuning。
4. **精度**：FP16 一个 flag；INT8 需 `IInt8Calibrator` 提供校准数据集。
5. **Plugin**：内置算子不够时，用手写 CUDA kernel 实现 `IPluginV2DynamicExt` 接入。
6. **序列化**：Engine 存为 `.engine`，部署反序列化直接推理，免重复构建。

---

## 验收标准

- 程序打印完整部署链路图，你能讲清每一环的输入输出与作用。
- ONNX 导出环节：装有 onnx 时真实产出文件并可校验节点；否则给出提示并继续。
- 你能写出 Builder + OnnxParser 构建 Engine 的骨架代码。
- 你能列出 Plugin 的核心接口，并说明 `enqueue()` 与手写 CUDA kernel 的关系。

---

## 动手挑战

1. **看图**：装 `onnx` 后，用 `onnx.load` + `netron` 可视化导出的计算图。
2. **构建 Engine**：在有 GPU + TensorRT 的机器上，用骨架代码把 ONNX 构建成 `.engine`。
3. **写一个 Plugin**：为一个自定义激活（如 GELU 变体）写 CUDA kernel 并封装成 Plugin。
4. **对比 opset**：改 `opset_version`，观察导出的算子集差异。

---

## 常见问题

- **Q：为什么导出报 “Module onnx is not installed”？**
  A：`torch.onnx.export` 落盘 protobuf 时需要 `onnx` 包。`pip install onnx` 即可，脚本在缺包时会降级为原理讲解而非报错。

- **Q：新版 PyTorch 导出为什么要额外依赖？**
  A：PyTorch 2.9+ 默认走 `dynamo` 导出器（需 `onnxscript`）。本脚本用 `dynamo=False` 走经典路径，依赖更少。

- **Q：INT8 Engine 一定比 FP16 快吗？**
  A：多数情况更快更省显存，但需校准且可能掉点；是否采用取决于精度预算，见 Day 66。

---

## 延伸阅读

- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
- [ONNX 官方规范](https://onnx.ai/)
- [TensorRT Python API — OnnxParser / Builder](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/)
- [Extending TensorRT with Custom Layers (Plugins)](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#extending)
