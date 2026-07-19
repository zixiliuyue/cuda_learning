# Speed-Demon 快速入门指南

## 🚀 第一步：运行环境检查

### Windows
```batch
cd Speed-Demon
build.bat
```

### Linux / macOS
```bash
cd Speed-Demon
chmod +x build.sh
./build.sh
```

## 📚 教程结构

```
Speed-Demon/
├── 01_theory/              # 理论基础
│   ├── 01_pprof_theory.md
│   ├── 02_memory_model.md
│   ├── 03_scheduler_theory.md
│   └── 04_gc_theory.md
├── 02_practice/            # 实操案例
│   ├── 01_cpu_profiling/  # CPU 性能分析
│   ├── 02_memory_profiling/  # 内存分析
│   └── 03_goroutine_leak/    # Goroutine 泄漏
├── 03_projects/           # 项目实战
│   └── 01_web_service/    # Web 服务优化
├── build.bat              # Windows 构建脚本
├── build.sh               # Linux/macOS 构建脚本
├── go.mod                 # Go 模块定义
├── README.md              # 项目说明
└── QUICKSTART.md          # 本文档
```

## 🎯 学习路径推荐

### Day 1: 基础理论
1. 阅读 `01_pprof_theory.md` - 理解 pprof 工作原理
2. 阅读 `02_memory_model.md` - 理解内存模型和逃逸分析
3. 阅读 `03_scheduler_theory.md` - 理解 Go 调度器
4. 阅读 `04_gc_theory.md` - 理解 GC 工作原理

### Day 2-3: 实操练习

#### CPU 性能分析
```bash
cd 02_practice/01_cpu_profiling
go run main.go

# 另一个终端
curl http://localhost:6060/heavy

# 采集 profile
go tool pprof http://localhost:6060/debug/pprof/profile?seconds=10
```

#### 内存分析
```bash
cd 02_practice/02_memory_profiling
go run main.go

# 另一个终端
for i in {1..100}; do curl http://localhost:6060/leak; done

# 分析内存
go tool pprof http://localhost:6060/debug/pprof/heap
```

#### Goroutine 泄漏
```bash
cd 02_practice/03_goroutine_leak
go run main.go

# 另一个终端
for i in {1..20}; do curl http://localhost:6060/leak1; done

# 观察 goroutine 数量增长
curl http://localhost:6060/stats
```

### Day 4: 项目实战
```bash
cd 03_projects/01_web_service
go run main.go

# 使用 hey 进行压力测试
hey -n 10000 -c 50 http://localhost:6060/user/bad?id=1
hey -n 10000 -c 50 http://localhost:6060/user/good?id=1

# 对比两者的性能差异
```

## 🔧 常用 pprof 命令

### 进入交互模式
```bash
go tool pprof <profile-file-or-url>
```

### 常用命令
```bash
top              # 查看热点函数
top -cum         # 按累计时间排序
list <function>  # 查看函数源码级耗时
web              # 生成火焰图（需要 graphviz）
svg              # 生成 SVG 文件
traces           # 查看所有调用栈
```

## 📊 快速诊断清单

### CPU 高
1. `go tool pprof profile`
2. `top` 看哪个函数占比高
3. `list <function>` 看具体哪一行
4. 考虑算法优化、并发优化

### 内存高
1. `go tool pprof -alloc_space heap`
2. 看是否有太多小对象分配
3. 检查是否有内存泄漏
4. 考虑 sync.Pool 或预分配

### Goroutine 增长
1. `curl http://localhost:6060/debug/pprof/goroutine?debug=2`
2. 看大部分 goroutine 卡在什么操作
3. 检查 channel 操作是否正确
4. 检查是否有未关闭的资源

## ⚠️ 常见问题

### Q: web 命令打不开？
A: 需要安装 graphviz
- Windows: 下载 https://graphviz.org/download/
- macOS: `brew install graphviz`
- Ubuntu: `sudo apt install graphviz`

### Q: profile 采集不到数据？
A: 确保采集期间有实际负载，采样时间不要太短

### Q: 编译时逃逸分析怎么看？
A: `go build -gcflags="-m -m" ./... 2>&1 | grep escape`

## 📖 延伸资源

- [Go 官方 pprof 文档](https://golang.org/pkg/runtime/pprof/)
- [Google pprof](https://github.com/google/pprof)
- [Go 语言高性能编程](https://geektutu.com/post/high-performance-go.html)

## 🎓 学习建议

1. **动手是关键** - 每个例子都要亲自运行、修改、观察
2. **对比思考** - 优化前后一定要有数据对比
3. **系统思维** - 性能问题往往是系统级的，不是单点问题
4. **建立直觉** - 看多了 profile，自然会有感觉

祝你学习愉快！🚀
