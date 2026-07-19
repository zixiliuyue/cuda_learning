# Day 51: Continuous / Dynamic Batching（连续/动态批处理）

> 🚀 **第十阶段 · LLM 推理优化**。Day48-50 解决了「单请求怎么算得快、KV 怎么省显存」。本天上升到**服务调度**层面：面对随机到达、长度各异的大量请求，如何组批才能既高吞吐又低延迟。这是 vLLM/TGI 等推理引擎相比朴素批处理的分水岭。

## 学习目标

1. **理解** 静态批处理的两大痛点：对头阻塞（短请求等长请求）与 padding 浪费
2. **实现** 静态批与连续批两种调度策略的离散事件模拟
3. **对比** 两者的平均延迟、吞吐、GPU 有效利用率
4. **掌握** 连续批「以解码步为粒度、完成即出、空位即入」的核心机制

---

## 前置依赖

- **Day 49 KV Cache / Day 50 PagedAttention**：连续批依赖分页 KV Cache 才能灵活增删请求
- **Day 48 Decoder 层**：每个解码步就是对当前 batch 做一次前向

---

## 核心概念

### 静态批 vs 连续批

```
  时间轴（每列一个解码步，█=有效计算，░=padding/空闲槽位）：

  静态批（batch=4，整批等最长者）：
    reqA(len2): ██░░░░░░   ← 生成完仍占槽陪跑
    reqB(len3): ███░░░░░
    reqC(len8): ████████
    reqD(len5): █████░░░
              └ 整批 8 步后才统一结束、才开下一批 → 大量 ░ 浪费

  连续批（完成即出、空位即入）：
    reqA(len2): ██                 ← 完成立即返回并让位
    reqB(len3): ███
    reqC(len8): ████████
    reqD(len5): █████
    新到 reqE 在 A 完成后立刻补入 A 的槽位 → 每步尽量装满
```

- **静态批**：凑齐一批一起开始、一起结束，整批被最长请求拖住（对头阻塞），已完成请求继续占槽产生 padding。
- **连续批（Continuous / In-flight Batching）**：调度粒度是**单个解码步**。每步完成的请求立即移出并返回，空出的槽位下一步就让新到请求补入，GPU 每步尽量装满有效请求。

### 为什么连续批需要 PagedAttention？

请求随时进出 batch，KV Cache 必须能灵活分配/回收——分页管理（Day50）恰好提供了这种「按块增删、随意复用」的能力，两者是绝配。

---

## 文件结构

```
day51_dynamic_batching/
├── python/dynamic_batching.py  # 静态批 vs 连续批 离散事件模拟与指标对比
├── build.sh                    # Linux/WSL2 运行脚本
├── build.bat                   # Windows 运行脚本
└── README.md                   # 本文件
```

---

## 编译与运行

```bash
cd day51_dynamic_batching
python python/dynamic_batching.py   # 纯 Python 调度模拟，有无 torch/GPU 均可
```

Windows：`.\build.bat`

---

## 知识点总结

1. **对头阻塞**：静态批里短请求必须陪最长请求跑完，延迟被放大。
2. **padding 浪费**：已完成请求继续占槽，GPU 算的是无效格子，利用率低。
3. **步级调度**：连续批以解码步为单位决策，是消除上述浪费的关键。
4. **完成即出、空位即入**：完成请求立即返回，空槽立即被新请求填满。
5. **与分页协同**：灵活进出 batch 需要 PagedAttention 的按块 KV 管理支撑。
6. **指标权衡**：连续批同时优化吞吐与平均延迟，是推理引擎的默认策略。

---

## 验收标准

- 程序打印静态批与连续批的总步数、平均延迟、吞吐、GPU 利用率。
- 连续批的平均延迟低于静态批、吞吐与利用率高于静态批，打印出提升百分比。
- 你能解释「对头阻塞」与「padding 浪费」如何被连续批消除。
- 无 GPU、无 torch 下均可运行，exit code 0。

---

## 动手挑战

1. **扫参数**：改变请求数 N、batch 大小与长度分布，观察两种策略差距如何变化。
2. **加优先级**：给连续批加入按剩余长度或 SLA 的调度优先级，观察尾延迟变化。
3. **抢占**：显存不足时抢占某请求（换出 KV），模拟 vLLM 的 preemption + 重算/换出。
4. **接真实前向**：把每一步替换成对 batch 的真实 batched decode 前向（接 Day49 的 KV Cache）。

---

## 常见问题

- **Q：连续批一定更好吗？**
  A：在变长、随机到达的真实负载下几乎总是更优。若所有请求等长且同时到达，两者退化为相同。

- **Q：本天为什么是纯 Python 模拟而不跑 GPU？**
  A：调度策略的收益来自「安排哪些请求在同一步计算」，与单步是否在 GPU 上无关。模拟能清晰隔离并量化调度本身的影响；真实系统里每一步替换成一次 batched GPU 前向即可。

- **Q：连续批和动态 batching 是一回事吗？**
  A：常混用。广义动态批指运行时按负载组批；连续批是其中最有效的一种——以解码步为粒度持续增删请求（也叫 in-flight batching）。

---

## 延伸阅读

- [Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI'22)](https://www.usenix.org/conference/osdi22/presentation/yu)
- [How Continuous Batching Enables 23x Throughput (Anyscale)](https://www.anyscale.com/blog/continuous-batching-llm-inference)
