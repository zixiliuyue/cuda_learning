# 01 - GIL 陷阱：多线程跑 CPU 密集任务反而变慢

## 问题描述

Python 的全局解释器锁（Global Interpreter Lock, GIL）确保在任何时刻只有一个线程执行 Python 字节码。这意味着：

- **IO 密集型任务**：多线程有效，因为线程在等待 IO 时会释放 GIL
- **CPU 密集型任务**：多线程反而更慢，因为线程切换会带来额外开销

## 示例代码

```python
import threading
import time

def cpu_heavy_task(n):
    result = 0
    for i in range(n):
        result += i * i
    return result

# 反模式：多线程跑 CPU 密集任务
def threaded_version():
    threads = []
    for _ in range(4):
        t = threading.Thread(target=cpu_heavy_task, args=(10_000_000,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()

# 正确做法：单线程
def single_thread_version():
    for _ in range(4):
        cpu_heavy_task(10_000_000)
```

## 性能对比

运行 `python 01_gil_trap.py` 会输出类似：

```
单线程耗时: 2.15 秒
多线程耗时: 2.87 秒
多进程耗时: 0.68 秒
```

**结论**：多线程比单线程还慢！

## 为什么会这样？

1. **GIL 的存在**：同一时刻只有一个线程执行 Python 代码
2. **线程切换开销**：操作系统需要在多个线程间切换
3. **缓存失效**：频繁的线程切换导致 CPU 缓存失效

## 修复方案

### 方案 1：使用多进程

```python
from multiprocessing import Pool

def multiprocess_version():
    with Pool(4) as pool:
        pool.map(cpu_heavy_task, [10_000_000] * 4)
```

### 方案 2：使用 C 扩展释放 GIL

NumPy、Pandas 等库的内部操作会释放 GIL，可以在多线程中高效运行。

### 方案 3：使用 concurrent.futures

```python
from concurrent.futures import ProcessPoolExecutor

def executor_version():
    with ProcessPoolExecutor(max_workers=4) as executor:
        list(executor.map(cpu_heavy_task, [10_000_000] * 4))
```

## 总结

| 任务类型 | 推荐方案 |
|---------|---------|
| CPU 密集型 | 多进程 |
| IO 密集型 | 多线程 / 异步 |
| 混合类型 | 根据瓶颈选择 |

记住：**多线程不是银弹！** 在 Python 中处理 CPU 密集型任务时，优先考虑多进程。
