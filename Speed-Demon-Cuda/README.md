# 🚀 Speed-Demon-Cuda: CUDA加速学习手册

> 从零到精通，掌握CUDA并行编程的艺术

## 📚 项目简介

Speed-Demon-Cuda 是一个系统性的 CUDA 学习项目，旨在帮助开发者从理论到实践，逐步掌握 GPU 并行编程技术。本项目参考了 Speed-Demon 和 Speed-Demon-Python 的设计理念，专注于 NVIDIA CUDA 平台的高性能计算。

## 🎯 学习目标

- ✅ 理解 CUDA 编程模型和核心概念
- ✅ 掌握 CUDA 内存优化策略
- ✅ 学会编写高效的 CUDA 内核
- ✅ 精通性能分析和优化技术
- ✅ 实现真实场景的并行计算

## 📂 项目结构

```
Speed-Demon-Cuda/
├── 01_theory/          # 📖 理论基础
├── 02_practice/        # 💻 实践入门
├── 03_optimization/    # ⚡ 性能优化
├── 04_memory/          # 🧠 内存管理
├── 05_concurrent/      # 🔄 并发与流
├── 06_full_case/       # 🎯 完整案例
├── 07_interview/       # 🎤 面试指南
├── 08_benchmarks/      # 📊 基准测试
├── 09_anti_patterns/   # ⚠️ 反模式
├── 10_decision_tree/   # 🌳 决策树
└── common/             # 🔧 公共工具
```

## 🛠️ 学习路径

### 第一阶段：基础入门 (Week 1-2)
1. **01_theory** - 学习 CUDA 基础理论
2. **02_practice** - 编写第一个 CUDA 程序

### 第二阶段：性能优化 (Week 3-4)
3. **03_optimization** - 学习核心优化技术
4. **04_memory** - 精通内存层次结构

### 第三阶段：高级主题 (Week 5-6)
5. **05_concurrent** - 并发执行和流处理
6. **06_full_case** - 实战项目开发

### 第四阶段：进阶提升 (Week 7+)
7. **07_interview** - 面试准备
8. **08_benchmarks** - 性能基准测试
9. **09_anti_patterns** - 避免常见错误
10. **10_decision_tree** - 决策指南

## 🔧 环境要求

### 硬件要求
- NVIDIA GPU (支持 CUDA Compute Capability 7.5+)
- 建议显存 >= 4GB

### 软件要求
- CUDA Toolkit 11.0+
- CMake 3.18+
- Visual Studio 2022 (Windows) / GCC 9+ (Linux)
- Python 3.8+ (用于性能分析)

## 🚀 快速开始

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd Speed-Demon-Cuda
   ```

2. **构建项目**
   ```bash
   # Windows
   build.bat
   
   # Linux/Mac
   chmod +x build.sh
   ./build.sh
   ```

3. **运行第一个示例**
   ```bash
   cd 02_practice
   # 运行编译好的示例程序
   ```

详细的使用指南请查看 [QUICKSTART.md](QUICKSTART.md)

## 📖 学习资源

### 官方文档
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### 速查表
- [CHEATSHEET.md](CHEATSHEET.md) - CUDA 常用命令速查表

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

## 📝 许可证

MIT License

---

**🚀 让计算速度飞起来！**
