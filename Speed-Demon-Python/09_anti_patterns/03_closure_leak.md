# 03 - 闭包捕获大对象导致无法释放

## 问题描述

闭包（Closure）是指内部函数引用了外部函数的变量。当闭包被返回或被其他对象引用时，它会捕获整个外部作用域的环境，即使只使用了其中一个变量，所有外部变量都无法被垃圾回收，导致内存泄漏。

## 示例代码

```python
def create_handler():
    big_data = [i * i for i in range(1_000_000)]  # 大对象
    
    def handler(x):
        return x * 2  # 实际上并没有使用 big_data
    
    return handler

# 反模式：handler 闭包捕获了整个外部环境，big_data 无法释放
h = create_handler()
# 即使 handler 根本没用到 big_data，big_data 仍然存在于闭包的 __closure__ 中
```

## 性能对比

运行 `python 03_closure_leak.py` 会输出类似：

```
反模式: 闭包引用的变量数量: 1
反模式: 大对象仍在内存中: True
反模式: 大对象内存占用: 8.45 MB

修复方案: 闭包引用的变量数量: 0
修复方案: 大对象已被释放: True
修复方案: 内存占用几乎为 0
```

## 为什么会这样？

1. **闭包捕获整个作用域**：Python 的闭包会捕获整个外部函数的局部变量字典
2. **引用计数增加**：即使内部函数没有使用某些变量，它们仍然被闭包引用
3. **垃圾回收受阻**：只要闭包存在，所有捕获的变量都无法被回收

## 修复方案

### 方案 1：使用类而不是闭包

```python
class Handler:
    def __init__(self, multiplier):
        self.multiplier = multiplier
    
    def __call__(self, x):
        return x * self.multiplier

def create_handler():
    big_data = [i * i for i in range(1_000_000)]
    # 只传递需要的值
    return Handler(2)
```

### 方案 2：在创建闭包前删除不需要的变量

```python
def create_handler():
    big_data = [i * i for i in range(1_000_000)]
    multiplier = 2
    
    # 删除不需要的变量
    del big_data
    
    def handler(x):
        return x * multiplier
    
    return handler
```

### 方案 3：使用 functools.partial

```python
from functools import partial

def multiply(x, multiplier):
    return x * multiplier

def create_handler():
    big_data = [i * i for i in range(1_000_000)]
    return partial(multiply, multiplier=2)
```

## 更隐蔽的场景

```python
def process_large_file(filename):
    data = load_huge_data(filename)  # 100MB 数据
    
    def validate():
        # 只检查了一些元数据，根本没用到 data
        return True
    
    def save():
        # 用到了 data
        return data.save()
    
    # 即使只返回 validate，data 也被捕获了
    return validate  # ❌ data 无法释放
```

## 总结

1. **闭包会捕获整个外部作用域**，而不仅仅是使用的变量
2. **对于大对象，优先使用类或 functools.partial 而不是闭包**
3. **在创建闭包前删除不需要的变量**
4. **使用 `__closure__` 属性检查闭包捕获的变量**：

```python
# 检查闭包捕获的变量
print(h.__closure__)
# 查看变量内容
print(h.__closure__[0].cell_contents)
```

闭包是强大的工具，但要小心使用，避免意外的内存泄漏！
