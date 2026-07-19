# 02 - 可变默认参数导致内存泄漏

## 问题描述

Python 中函数的默认参数只在函数定义时初始化一次，而不是每次调用时重新创建。如果默认参数是可变对象（如列表、字典），多次调用函数会导致数据累积，引发内存泄漏或数据错误。

## 示例代码

```python
def add_item(item, items=[]):
    items.append(item)
    return items

# 反模式：多次调用会累积数据
result1 = add_item(1)  # [1]
result2 = add_item(2)  # [1, 2] - 不是期望的 [2]！
result3 = add_item(3)  # [1, 2, 3] - 内存持续增长
```

## 性能对比

运行 `python 02_mutable_default.py` 会输出类似：

```
第 1000 次调用后列表长度: 1000
第 2000 次调用后列表长度: 2000
第 3000 次调用后列表长度: 3000
...
第 10000 次调用后列表长度: 10000
内存使用: 85.23 KB

使用修复方案后:
每次调用列表长度都是: 1
内存使用: 0.09 KB
```

## 为什么会这样？

1. **默认参数在定义时创建**：`items=[]` 只在函数定义时创建一次
2. **可变对象被修改**：每次调用都修改同一个列表对象
3. **引用一直存在**：函数对象持有对默认参数的引用，导致无法垃圾回收

## 修复方案

### 方案 1：使用 None 作为默认值

```python
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

### 方案 2：文档化可变默认参数（不推荐）

如果确实需要共享状态，应该在文档中明确说明：

```python
def add_item(item, items=[]):
    """添加项目到列表
    
    注意：items 参数是共享的，多次调用会累积数据
    """
    items.append(item)
    return items
```

## 常见的可变默认参数场景

```python
# ❌ 错误
def process_data(data, cache={}):
    pass

# ❌ 错误
def log_message(msg, history=[]):
    pass

# ❌ 错误
def get_config(overrides={}):
    pass
```

## 总结

1. **永远不要使用可变对象作为默认参数**
2. **使用 None 作为默认值，在函数内部初始化**
3. **使用类型提示帮助发现问题**：

```python
from typing import List, Optional

def add_item(item: int, items: Optional[List[int]] = None) -> List[int]:
    if items is None:
        items = []
    items.append(item)
    return items
```

这个反模式可能导致难以调试的 bug，务必小心！
