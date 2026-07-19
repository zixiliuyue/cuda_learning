# Day 64: 知识蒸馏（Knowledge Distillation）

> 🎓 **第十一阶段（模型压缩与部署）**。量化改数值精度、剪枝删参数，而蒸馏走另一条路：让一个**小模型（student）**去模仿**大模型（teacher）**的行为，用更少的参数逼近甚至超过大模型的精度。

## 学习目标

1. **理解** 知识蒸馏的核心：teacher 软标签里的「暗知识（dark knowledge）」
2. **掌握** 温度 T 如何软化 softmax、放大类间相似性
3. **实现** KD 损失：`alpha·T²·KL(soft) + (1-alpha)·CE(hard)` 的加权组合
4. **对比** 相同 student 在「直接训练」与「蒸馏训练」下的精度差异
5. **扫描** 温度 T 的影响，形成超参调优直觉

---

## 前置依赖

- **Day 63 剪枝**、**Day 59-62 量化**：同属模型压缩，蒸馏常与它们叠加
- **交叉熵 / softmax 基础**：KD 损失建立在其上
- **KL 散度**：软标签项用 KL 衡量 student 与 teacher 分布的差异

---

## 核心概念

### 暗知识（Dark Knowledge）

teacher 的 softmax 输出不止告诉你「这是类别 1」，还告诉你「它有点像类别 2、完全不像类别 3」。这种**类间相似性**是 one-hot 硬标签丢失的信息。实验里一个真值为 1 的样本：

```
T=1   概率: [0.000 1.000 0.000 0.000]   ← 几乎 one-hot，暗知识被压平
T=4   概率: [0.023 0.900 0.072 0.005]   ← 类 2 明显比类 3 更相似
```

### 温度 T

$$\text{soft}(z)_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

T 越大，分布越「软」（熵越大），非目标类的相对概率被放大。蒸馏时 teacher 和 student 用**同一个 T** 软化；推理时 student 用 `T=1`。

### KD 损失（加权）

```python
soft_teacher = softmax(teacher_logits / T)
log_soft_student = log_softmax(student_logits / T)
kd = KL(log_soft_student || soft_teacher) * T**2   # T² 补偿梯度缩放
ce = cross_entropy(student_logits, hard_label)
loss = alpha * kd + (1 - alpha) * ce
```

- **软标签项**乘 `T²`：软化会让梯度幅值缩小约 `1/T²`，乘回来保持量级稳定。
- **alpha**：权衡「信任 teacher」与「信任真值」，常取 0.5~0.9。

---

## 文件结构

```
day64_distillation/
├── python/distillation.py   # teacher/student 蒸馏 + 温度扫描，含无 GPU 原理演示
├── build.sh                 # Linux/WSL2 运行脚本
├── build.bat                # Windows 运行脚本
└── README.md                # 本文件
```

> 有 CUDA 用 GPU，无 CUDA 用 CPU（都真实训练）；无 torch 用 NumPy 演示温度软化机制。

---

## 编译与运行

### Linux / WSL2

```bash
cd day64_distillation
python python/distillation.py
```

### Windows

```powershell
cd day64_distillation
.\build.bat
```

> 无 GPU 也能完整跑完，exit code 0。

---

## 知识点总结

1. **知识蒸馏**：让小 student 模仿大 teacher 的软标签，用更少参数逼近其精度。
2. **暗知识**：teacher softmax 里的类间相似性，比 one-hot 硬标签信息更丰富。
3. **温度 T**：软化分布放大暗知识；蒸馏项乘 `T²` 补偿梯度；推理时 T=1。
4. **加权损失**：`alpha·T²·KL(soft) + (1-alpha)·CE(hard)`，权衡 teacher 与真值。
5. **组合使用**：蒸馏常与量化/剪枝叠加，是端侧小模型常用的「涨点」手段。

---

## 验收标准

- 程序打印暗知识演示：同一样本在 T=1 与 T>1 下的概率分布，能看出软化效果。
- 对比实验中，「student 蒸馏」精度 ≥「student 直接训练」（本例 +2~6 个百分点）。
- 压缩比（teacher/student 参数量）打印为约 130x。
- 你能解释软标签项为什么要乘 `T²`。

---

## 动手挑战

1. **调 alpha**：把 alpha 从 0.7 扫到 0.1~0.9，观察软/硬标签权重对精度的影响。
2. **中间层蒸馏**：除了 logits，让 student 的隐藏层特征也去匹配 teacher（FitNets）。
3. **无标签蒸馏**：只用 teacher 软标签（alpha=1），在无真值的数据上蒸馏，看效果。
4. **自蒸馏**：teacher 与 student 结构相同，观察是否仍能涨点（Born-Again Networks）。

---

## 常见问题

- **Q：student 蒸馏后能超过 teacher 吗？**
  A：可能。软标签起到正则化作用，小模型有时泛化更好，本例中 student 蒸馏后就略超 teacher。

- **Q：温度是不是越大越好？**
  A：不是。过大分布过平会淹没主类信息，过小又退化为硬标签。常见最优在 2~5。

- **Q：KL 的方向重要吗？**
  A：标准 KD 用 `KL(student || teacher)`（前向 KL）。方向不同会带来 mode-covering / mode-seeking 的差异，是研究话题。

---

## 延伸阅读

- [Hinton et al., Distilling the Knowledge in a Neural Network (2015)](https://arxiv.org/abs/1503.02531)
- [FitNets: Hints for Thin Deep Nets (2015)](https://arxiv.org/abs/1412.6550)
- [DistilBERT (2019)](https://arxiv.org/abs/1910.01108)
- [A Survey on Knowledge Distillation (2021)](https://arxiv.org/abs/2006.05525)
