# 04 - 循环 append 列表低效

## 问题描述

在循环中多次调用 `list.append()` 方法时，虽然列表会自动扩容，但频繁的扩容操作（需要重新分配内存并复制数据）会导致性能下降。特别是当可以预先知道列表大小时，应该一次性分配内存。

## 示例代码

```python
def create_list_bad(n):
    result = []
    for i in range(n):
        result.append(i * i)
    return result
```

虽然 `append()` 是 O(1) 摊销复杂度，但当 n 很大时，多次扩容仍然有开销。

## 性能对比

运行 `python 04_loop_append.py` 会输出类似：

```
循环 append 耗时: 0.12 秒
列表推导式耗时: 0.07 秒
预分配列表耗时: 0.05 秒
list(range) 耗时: 0.02 秒

性能提升比:
列表推导式比循环 append 快 1.71x
预分配比循环 append 快 2.40x
```

## 为什么会这样？

1. **列表扩容机制**：列表的容量是指数增长的（0, 4, 8, 16, 25, 35, 46...）
2. **内存重新分配**：每次扩容都需要分配新内存并复制所有元素
3. **缓存友好性**：预分配的内存布局更有利于 CPU 缓存

## 修复方案

### 方案 1：使用列表推导式

```python
def create_list_good1(n):
    return [i * i for i in range(n)]
```

列表推导式是最快的方式，因为它在底层优化了内存分配。

### 方案 2：预分配列表大小

```python
def create_list_good2(n):
    result = [0] * n  # 预分配
    for i in range(n):
        result[i] = i * i
    return result
```

### 方案 3：使用 map 或生成器

```python
def create_list_good3(n):
    return list(map(lambda i: i * i, range(n)))
```

### 方案 4：使用 extend 批量添加

```python
def create_list_good4(n):
    result = []
    result.extend(i * i for i in range(n))
    return result
```

## 更极端的场景：嵌套循环

```python
# ❌ 不好
matrix = []
for i in range(1000):
    row = []
    for j in range(1000):
        row.append(i * j)
    matrix.append(row)

# ✅ 更好
matrix = [[i * j for j in range(1000)] for i in range(1000)]
```

## 特殊情况：不知道最终大小

如果不知道列表最终大小，`append()` 仍然是最佳选择：

```python
def filter_positive(numbers):
    result = []
    for num in numbers:
        if num > 0:
            result.append(num)
    return result
```

这种情况下 `append()` 是合理的，因为无法预分配。

## 总结

| 场景 | 推荐方案 |
|------|---------|
| 已知大小且逻辑简单 | 列表推导式 |
| 已知大小但逻辑复杂 | 预分配 + 索引赋值 |
| 未知大小 | append() |
| 批量添加 | extend() |

**记住**：
1. **优先使用列表推导式** - 最简洁、最快
2. **避免在循环中反复构建列表** - 考虑生成器
3. **使用 `sys.getsizeof()` 检查内存使用**

```python
import sys
lst = []
print(f"空列表: {sys.getsizeof(lst)} bytes")
lst.append(1)
print(f"1个元素: {sys.getsizeof(lst)} bytes")
lst.extend(range(100))
print(f"101个元素: {sys.getsizeof(lst)} bytes")
```
