# 05 - 循环内访问全局变量

## 问题描述

在 Python 中，访问全局变量比访问局部变量慢得多。如果在循环中频繁访问全局变量，会导致显著的性能下降。这是因为全局变量存储在字典中，而局部变量存储在数组中，数组访问比字典查找快得多。

## 示例代码

```python
GLOBAL_MULTIPLIER = 2

def calculate_bad(numbers):
    result = []
    for num in numbers:
        # 在循环中访问全局变量
        result.append(num * GLOBAL_MULTIPLIER)
    return result
```

## 性能对比

运行 `python 05_global_var.py` 会输出类似：

```
循环内访问全局变量耗时: 0.15 秒
将全局变量赋值给局部变量耗时: 0.09 秒
使用局部变量快 1.67x

循环内调用全局函数耗时: 0.20 秒
将全局函数赋值给局部变量耗时: 0.12 秒
使用局部函数快 1.67x
```

## 为什么会这样？

1. **局部变量访问是 O(1) 数组索引**：非常快
2. **全局变量访问是字典查找**：需要哈希计算和可能的冲突处理
3. **字节码层面的差异**：
   - `LOAD_FAST` - 加载局部变量（快）
   - `LOAD_GLOBAL` - 加载全局变量（慢）

可以使用 `dis` 模块查看字节码差异：

```python
import dis
dis.dis(calculate_bad)
```

## 修复方案

### 方案 1：将全局变量赋值给局部变量

```python
def calculate_good1(numbers):
    multiplier = GLOBAL_MULTIPLIER  # 赋值给局部变量
    result = []
    append = result.append  # 甚至可以把方法也赋值给局部变量
    
    for num in numbers:
        append(num * multiplier)
    return result
```

### 方案 2：使用函数参数传递

```python
def calculate_good2(numbers, multiplier=GLOBAL_MULTIPLIER):
    result = []
    for num in numbers:
        result.append(num * multiplier)
    return result
```

### 方案 3：使用类封装

```python
class Calculator:
    def __init__(self, multiplier):
        self.multiplier = multiplier
    
    def calculate(self, numbers):
        result = []
        multiplier = self.multiplier  # 属性赋值给局部变量
        for num in numbers:
            result.append(num * multiplier)
        return result
```

## 更极端的场景：循环内的所有查找

```python
import math

def compute_bad(data):
    result = []
    for x in data:
        # 每次循环都要查找 math.sqrt 和 len
        result.append(math.sqrt(x) + len(data))
    return result

def compute_good(data):
    sqrt = math.sqrt  # 预先绑定
    data_len = len(data)  # 只计算一次
    result = []
    append = result.append
    
    for x in data:
        append(sqrt(x) + data_len)
    return result
```

## 性能优化的层次

```python
def optimize_example(numbers):
    # 1. 预先获取所有需要的全局对象
    import math
    sqrt = math.sqrt
    pi = math.pi
    
    # 2. 预先计算不变的量
    data_len = len(numbers)
    
    # 3. 绑定方法到局部变量
    result = []
    append = result.append
    
    # 4. 循环内只使用局部变量
    for num in numbers:
        append(sqrt(num) * pi / data_len)
    
    return result
```

## 总结

| 优化技巧 | 性能提升 | 适用场景 |
|---------|---------|---------|
| 全局变量 -> 局部变量 | ~1.5-2x | 循环内频繁访问 |
| 方法绑定到局部变量 | ~1.2-1.5x | 循环内频繁调用方法 |
| 预先计算不变量 | 视情况 | 值在循环内不变 |
| 使用局部导入 | 小幅提升 | 模块在循环内使用 |

**记住**：
1. **在循环外获取全局变量** - 不要在循环内反复查找
2. **常用方法绑定到局部变量** - 如 `append = list.append`
3. **使用 `dis` 分析字节码** - 理解底层执行机制
4. **不要过度优化** - 只有在性能瓶颈时才这样做

这个优化在简单循环中可能不明显，但在嵌套循环或大数据量时差异很大。
