# Day 54: 算子融合在推理中的应用 — Fused QKV / Fused LayerNorm+Residual

> ⚡ **推理加速的关键手法**。Transformer 里每个小算子都要启动一次 GPU kernel，并把中间结果写回显存再读出。**算子融合**把多步合成一个 kernel，减少启动开销与显存往返——这是 vLLM / TensorRT-LLM 等引擎的基本功。

## 学习目标

1. **实现** Fused QKV 投影（三个线性合成一个大 GEMM）
2. **实现** Fused LayerNorm + Residual（残差相加与归一化在一步完成）
3. **对比** 融合前后的 kernel 启动次数与中间张量数量
4. **验证** 融合前后数值完全一致（融合只改执行方式，不改数学结果）

---

## 前置依赖

- **Day 41 Triton 融合 Softmax**：融合思想的开端
- **Day 31 Attention / LayerNorm**：QKV 与 LayerNorm 的来历
- 矩阵乘 `y = x W` 与列拼接

---

## 核心概念

### 融合为什么快

每个 kernel 都有两项固定成本：**启动延迟** + **中间结果的全局显存往返**。融合收益：

1. **kernel 启动次数下降** → 减少 launch overhead
2. **中间张量不落地显存** → 减少内存带宽消耗（推理常是带宽瓶颈）

### Fused QKV 投影

```
未融合:  q = x @ Wq,  k = x @ Wk,  v = x @ Wv     # 3 次 GEMM, 3 个中间张量
融合:    qkv = x @ [Wq | Wk | Wv]                 # 1 次大 GEMM
         q, k, v = qkv.split(d, dim=-1)           # 视图切分, 不额外拷贝
```

把 Wq/Wk/Wv 在**列方向拼接**成大矩阵，一次 GEMM 得到 `[.., 3d]`，再切分。大矩阵乘更利于 Tensor Core 的利用率。

### Fused LayerNorm + Residual

```
未融合:  h = x + residual        # 中间张量 h 写回显存
         mean, var = 统计(h)
         out = (h-mean)/sqrt(var+eps) * gamma + beta
融合:    在一个 kernel 内完成残差相加 + 求 mean/var + 归一化, h 不落地
```

---

## 文件结构

```
day54_fused_inference/
├── python/fused_inference.py   # Fused QKV + Fused LayerNorm+Residual, 数值一致性验证
├── build.sh                    # Linux/WSL2/macOS 运行脚本
├── build.bat                   # Windows 运行脚本
└── README.md                   # 本文件
```

> 以 **PyTorch** 为主，`device = "cuda" if torch.cuda.is_available() else "cpu"`：有 CUDA 跑 GPU、无 CUDA 跑 CPU，逻辑都真实运行；无 torch 时退化为 NumPy。真实融合需自定义 CUDA/Triton kernel，本天用等价数学表达演示语义。

---

## 编译与运行

### Linux / WSL2 / macOS

```bash
cd day54_fused_inference
pip install torch
python python/fused_inference.py
```

### Windows

```powershell
cd day54_fused_inference
.\build.bat
```

---

## 知识点总结

1. **Fused QKV** 用列拼接把 3 次 GEMM 变 1 次，减少启动并提升 GEMM 效率。
2. **split 是视图操作**，不产生额外拷贝，切分开销极小。
3. **Fused LayerNorm+Residual** 让中间张量 h 不落地全局显存，省带宽。
4. **融合不改数学结果**，只改执行方式，因此融合前后数值必须一致。
5. **带宽瓶颈**：推理（尤其 decode 阶段）常受显存带宽限制，融合收益显著。
6. 真实融合内核用 **Triton / CUDA** 手写，PyTorch 层面可用 `torch.compile` 自动融合。

---

## 验收标准

- Fused QKV 的 Q/K/V 融合前后最大误差 < 1e-4，显示「一致 ✓」。
- Fused LayerNorm+Residual 融合前后最大误差 < 1e-4，显示「一致 ✓」。
- 程序说明融合前后 kernel 启动次数与中间张量数量的变化。
- 无 GPU 时在 CPU 上完整运行，exit code 0。

---

## 动手挑战

1. **Fused Bias+GELU**：把线性层的加 bias 与 GELU 激活融合，验证数值一致。
2. **用 Triton 实现**：仿照 Day41，用 Triton 写一个真正的 Fused LayerNorm kernel。
3. **torch.compile**：用 `torch.compile` 包住未融合版本，观察它能否自动融合。

---

## 常见问题

- **Q：融合后为什么数值可能有极小差异？**
  A：浮点运算顺序变化会带来 ~1e-6 级误差，属正常；本天阈值设 1e-4 足够。

- **Q：所有算子都能融合吗？**
  A：不能。融合适合「访存密集、计算轻」或「相邻依赖紧」的算子；跨大 GEMM 的融合收益有限。

---

## 延伸阅读

- [NVIDIA — Fused kernels in Transformer inference](https://developer.nvidia.com/blog/)
- [PyTorch torch.compile 文档](https://pytorch.org/docs/stable/torch.compiler.html)
