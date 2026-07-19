# Python 数据结构性能优化指南

## 目录
1. [list vs tuple vs array - 有序序列的选择](#1-list-vs-tuple-vs-array---有序序列的选择)
2. [dict vs OrderedDict vs defaultdict - 字典类型选择](#2-dict-vs-ordereddict-vs-defaultdict---字典类型选择)
3. [set 的高效操作 - 集合运算与去重](#3-set-的高效操作---集合运算与去重)
4. [deque vs list - 双端队列性能对比](#4-deque-vs-list---双端队列性能对比)
5. [heapq 与 bisect - 堆与二分查找](#5-heapq-与-bisect---堆与二分查找)
6. [数据结构选择决策树](#6-数据结构选择决策树)
7. [性能对比汇总表](#7-性能对比汇总表)

---

## 1. list vs tuple vs array - 有序序列的选择

### 核心特性对比

| 特性 | list | tuple | array |
|------|------|-------|-------|
| **可变性** | 可变 | 不可变 | 可变 |
| **元素类型** | 任意混合 | 任意混合 | 必须相同 |
| **内存占用** | 高 | 中 | 低 |
| **访问速度** | 快 | 快 | 最快 |
| **迭代性能** | 快 | 快 | 最快 |

### 性能分析

#### 创建性能
- **tuple**: 创建最快，因为不可变，Python可以优化内存分配
- **list**: 创建速度次之，需要额外的空间用于动态扩容
- **array**: 创建最慢，因为需要类型检查和连续内存分配

#### 访问性能
- **array**: 最快，因为元素是连续存储的基本类型，CPU缓存友好
- **list** 和 **tuple**: 性能相近，因为都存储对象指针

#### 迭代性能
- **array**: 迭代和数值计算最快，避免了Python对象的开销
- **list** 和 **tuple**: 性能接近

### 使用场景

**使用 list 当:**
- 需要动态修改（添加、删除、修改元素）
- 需要存储不同类型的元素
- 需要通用的序列操作

**使用 tuple 当:**
- 数据不需要修改（作为配置、常量）
- 需要作为字典的键（不可哈希的list不能）
- 需要保护数据不被意外修改
- 函数返回多个值时

**使用 array 当:**
- 存储大量相同类型的数值数据
- 需要进行数值计算
- 内存占用是关键因素
- 与C语言交互时

### 代码示例

```python
from array import array

# list - 通用可变序列
my_list = [1, 2, 3, 4, 5]
my_list.append(6)  # OK
my_list[0] = 10    # OK

# tuple - 不可变序列
my_tuple = (1, 2, 3, 4, 5)
# my_tuple.append(6)  # 错误！
# my_tuple[0] = 10    # 错误！

# array - 类型固定的数值数组
# 'i' 表示有符号整数，'f' 表示浮点数，'d' 表示双精度浮点数
my_array = array('i', [1, 2, 3, 4, 5])
my_array.append(6)  # OK，但必须是相同类型
```

---

## 2. dict vs OrderedDict vs defaultdict - 字典类型选择

### 核心特性对比

| 特性 | dict | OrderedDict | defaultdict |
|------|------|-------------|-------------|
| **插入顺序** | Python 3.7+ 保留 | 严格保留 | Python 3.7+ 保留 |
| **默认值** | 无 | 无 | 有 |
| **性能** | 最快 | 稍慢 | 稍慢 |
| **特殊功能** | 无 | 重排、移动到两端 | 默认工厂函数 |

### 性能分析

#### Python 3.7 之后的变化
从 Python 3.7 开始，普通的 `dict` 也保留插入顺序了！这大大减少了 `OrderedDict` 的使用场景。

#### 创建性能
- **dict**: 最快，最常用
- **defaultdict**: 稍慢，因为需要处理默认值逻辑
- **OrderedDict**: 最慢，维护双向链表

#### 查找性能
三者的查找性能都非常接近，因为底层都是哈希表实现。

### 使用场景

**使用 dict 当:**
- 普通的键值对存储需求
- 需要最佳性能
- Python 3.7+ 且只需要保留插入顺序

**使用 OrderedDict 当:**
- 需要维护插入顺序，且需要顺序相关操作：
  - `move_to_end(key, last=True)` - 将键移动到两端
  - `popitem(last=True)` - 从两端弹出
  - 需要支持反向迭代
- Python 版本低于 3.7 且需要保留插入顺序

**使用 defaultdict 当:**
- 需要自动为不存在的键创建默认值
- 统计计数、分组等场景

### 代码示例

```python
from collections import OrderedDict, defaultdict

# dict - 普通字典
d = {'a': 1, 'b': 2}
print(d.get('c', 0))  # 需要手动处理默认值

# defaultdict - 带默认值的字典
dd = defaultdict(int)  # 不存在的键默认值为 0
dd['count'] += 1       # 无需检查键是否存在

# 分组示例
groups = defaultdict(list)
for item in ['apple', 'banana', 'apricot', 'blueberry']:
    groups[item[0]].append(item)
# 结果: {'a': ['apple', 'apricot'], 'b': ['banana', 'blueberry']}

# OrderedDict - 有序字典
od = OrderedDict([('a', 1), ('b', 2), ('c', 3)])
od.move_to_end('a')        # 将 'a' 移动到末尾
od.move_to_end('c', last=False)  # 将 'c' 移动到开头
first = od.popitem(last=False)   # 弹出第一个元素
```

---

## 3. set 的高效操作 - 集合运算与去重

### 核心特性

- **哈希表实现**: 成员检查 O(1) 复杂度
- **无序不重复**: 元素唯一，不保证顺序
- **数学集合操作**: 并集、交集、差集、对称差等

### 性能分析

#### 成员检查
- **set**: O(1) 平均复杂度，极快
- **list**: O(n) 线性查找，慢

**性能差距**: 对于大型数据集，set 的成员检查可以比 list 快 **几千倍**。

#### 去重性能
- **set**: 最简单、最快的去重方式
- **list + 手动去重**: 需要额外空间和判断

#### 集合运算
并集、交集、差集等操作都经过高度优化。

### 使用场景

**使用 set 当:**
- 需要快速成员检查
- 需要去重
- 需要数学集合运算（交集、并集、差集等）
- 需要消除重复元素但不关心顺序

### 代码示例

```python
# 成员检查
large_set = set(range(1000000))
large_list = list(range(1000000))

# set 检查 - O(1)
999999 in large_set  # 极快

# list 检查 - O(n)
999999 in large_list  # 很慢！

# 集合运算
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

union = set1 | set2        # 并集: {1, 2, 3, 4, 5, 6, 7, 8}
intersection = set1 & set2  # 交集: {4, 5}
difference = set1 - set2    # 差集: {1, 2, 3}
sym_diff = set1 ^ set2      # 对称差: {1, 2, 3, 6, 7, 8}

# 去重
duplicates = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
unique = set(duplicates)  # {1, 2, 3, 4}

# 多个集合的交集
common = set1 & set2 & set3  # 比多次循环快得多
```

### 性能提示

**不要这样做:**
```python
# 低效：每次检查都要遍历整个列表
for item in items:
    if item not in seen_list:
        seen_list.append(item)
        process(item)
```

**应该这样做:**
```python
# 高效：O(1) 成员检查
seen = set()
for item in items:
    if item not in seen:
        seen.add(item)
        process(item)
```

---

## 4. deque vs list - 双端队列性能对比

### 核心特性对比

| 操作 | list 复杂度 | deque 复杂度 |
|------|-------------|--------------|
| **尾部 append** | O(1) 摊销 | O(1) |
| **尾部 pop** | O(1) 摊销 | O(1) |
| **头部 append** | O(n) | O(1) |
| **头部 pop** | O(n) | O(1) |
| **中间插入** | O(n) | O(n) |
| **随机访问** | O(1) | O(n) |

### 性能分析

#### 为什么 list 在头部操作慢？
list 底层是动态数组，在头部插入或删除时，所有元素都需要向后或向前移动，导致 O(n) 复杂度。

#### deque 的优势
deque（双端队列）底层是双向链表的变种（块链表），在两端操作都是 O(1) 复杂度。

### 使用场景

**使用 deque 当:**
- 需要频繁在两端添加/删除元素
- 实现队列（FIFO）
- 实现栈（LIFO）- 虽然 list 也可以
- 需要固定大小的滑动窗口

**使用 list 当:**
- 需要随机访问元素
- 主要在尾部操作
- 需要通用的序列操作

### 代码示例

```python
from collections import deque

# deque - 双端队列
dq = deque([1, 2, 3, 4, 5])

# 两端操作都是 O(1)
dq.append(6)        # 右端添加 O(1)
dq.appendleft(0)    # 左端添加 O(1)
right = dq.pop()    # 右端弹出 O(1)
left = dq.popleft() # 左端弹出 O(1)

# 固定大小的 deque - 实现滑动窗口
window = deque(maxlen=3)
for i in range(10):
    window.append(i)
    print(list(window))
# [0], [0,1], [0,1,2], [1,2,3], [2,3,4], ...

# 队列 FIFO
queue = deque()
queue.append('task1')
queue.append('task2')
next_task = queue.popleft()  # 'task1'

# 栈 LIFO (list 也可以，deque 同样高效)
stack = deque()
stack.append('item1')
stack.append('item2')
top = stack.pop()  # 'item2'

# rotate 旋转操作
dq = deque([1, 2, 3, 4, 5])
dq.rotate(2)   # 右移2位: [4, 5, 1, 2, 3]
dq.rotate(-1)  # 左移1位: [5, 1, 2, 3, 4]
```

### 性能对比示例

```python
# ❌ 低效：用 list 作为队列
queue_list = []
for i in range(10000):
    queue_list.append(i)
while queue_list:
    item = queue_list.pop(0)  # O(n)！每次都要移动所有元素

# ✅ 高效：用 deque 作为队列
queue_deque = deque()
for i in range(10000):
    queue_deque.append(i)
while queue_deque:
    item = queue_deque.popleft()  # O(1)
```

---

## 5. heapq 与 bisect - 堆与二分查找

### 5.1 heapq - 堆队列

#### 核心特性
- 实现了最小堆（默认）
- 二叉堆数据结构
- 底层用 list 存储
- 堆顶元素总是最小的

#### 时间复杂度

| 操作 | 复杂度 |
|------|--------|
| **heappush** | O(log n) |
| **heappop** | O(log n) |
| **heapify** | O(n) |
| **获取最小值** | O(1) |

#### 使用场景
- 优先级队列
- 快速获取最小/最大的 K 个元素
- 合并多个有序序列
- 调度系统

#### 代码示例

```python
import heapq

# 创建堆
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 2)

# 弹出最小元素
smallest = heapq.heappop(heap)  # 1

# 从可迭代对象构建堆
data = [5, 3, 8, 1, 2, 9]
heapq.heapify(data)  # 原地构建，O(n)

# 获取前 K 个最小/最大元素
smallest_3 = heapq.nsmallest(3, data)  # [1, 2, 3]
largest_3 = heapq.nlargest(3, data)    # [9, 8, 5]

# 优先级队列
tasks = []
heapq.heappush(tasks, (2, 'low priority'))
heapq.heappush(tasks, (1, 'high priority'))
heapq.heappush(tasks, (3, 'medium priority'))

while tasks:
    priority, task = heapq.heappop(tasks)
    print(f"Processing: {task} (priority: {priority})")
# 按优先级顺序输出
```

### 5.2 bisect - 二分查找模块

#### 核心特性
- 在有序列表中进行二分查找
- 找到插入位置以保持列表有序
- O(log n) 查找复杂度

#### 时间复杂度

| 操作 | 复杂度 |
|------|--------|
| **bisect 查找** | O(log n) |
| **insort 插入** | O(n) |

**注意**: 虽然查找是 O(log n)，但插入仍然是 O(n) 因为是 list。

#### 使用场景
- 在已排序列表中快速查找
- 维护有序列表（插入时保持有序）
- 频率直方图
- 查找近似匹配

#### 代码示例

```python
import bisect

sorted_list = [1, 3, 5, 7, 9]

# 查找插入位置
pos = bisect.bisect_left(sorted_list, 6)  # 返回 3
pos_right = bisect.bisect_right(sorted_list, 5)  # 返回 3

# 有序插入（保持列表有序）
bisect.insort(sorted_list, 6)
# sorted_list 变为 [1, 3, 5, 6, 7, 9]

# 查找元素是否存在
def binary_search(lst, target):
    pos = bisect.bisect_left(lst, target)
    return pos < len(lst) and lst[pos] == target

# 评分等级映射
def get_grade(score):
    breakpoints = [60, 70, 80, 90]
    grades = ['F', 'D', 'C', 'B', 'A']
    idx = bisect.bisect_right(breakpoints, score)
    return grades[idx]

print(get_grade(85))  # 'B'
print(get_grade(59))  # 'F'
print(get_grade(90))  # 'A'
```

### heapq vs bisect 对比

| 场景 | heapq | bisect + sorted list |
|------|-------|----------------------|
| **构建** | O(n) heapify | O(n log n) sorted |
| **获取最小** | O(log n) pop | O(n) pop(0) |
| **任意查找** | O(n) 遍历 | O(log n) 二分 |
| **有序插入** | O(log n) push | O(n) insort |
| **获取K个最小** | O(n + k log n) | O(k) 取前k个 |

**选择建议:**
- 需要频繁获取/删除最小元素 → **heapq**
- 需要频繁查找任意元素 → **bisect + sorted list**
- 需要获取 Top K 但不需要完全排序 → **heapq.nsmallest/nlargest**

---

## 6. 数据结构选择决策树

```
开始
  │
  ├─ 需要键值对访问？
  │  ├─ 是 → 字典类型
  │  │   ├─ 需要自动默认值？
  │  │   │  ├─ 是 → defaultdict
  │  │   │  └─ 否
  │  │   │       ├─ 需要顺序操作（move_to_end等）？
  │  │   │       │  ├─ 是 → OrderedDict
  │  │   │       │  └─ 否 → dict ✅
  │  │   │      
  │  └─ 否 → 序列/集合类型
  │      ├─ 需要快速成员检查和去重？
  │      │  ├─ 是 → set ✅
  │      │  └─ 否
  │      │       ├─ 需要频繁两端操作？
  │      │       │  ├─ 是 → deque ✅
  │      │       │  └─ 否
  │      │       │       ├─ 元素不可变？
  │      │       │       │  ├─ 是 → tuple ✅
  │      │       │       │  └─ 否
  │      │       │       │       ├─ 大量数值计算且类型相同？
  │      │       │       │       │  ├─ 是 → array ✅
  │      │       │       │       │  └─ 否 → list ✅
  │
  ├─ 需要优先级队列或快速获取Top-K？
  │  └─ 是 → heapq ✅
  │
  └─ 需要在有序列表中二分查找？
     └─ 是 → bisect ✅
```

### 快速决策表

| 需求场景 | 推荐数据结构 | 不推荐 |
|----------|-------------|--------|
| 通用可变序列 | list | - |
| 常量配置、保护数据 | tuple | list |
| 大量同类型数值计算 | array | list |
| 普通键值存储 | dict | OrderedDict |
| 自动创建默认值 | defaultdict | dict + if判断 |
| 顺序操作（move_to_end等） | OrderedDict | dict |
| 快速成员检查、去重 | set | list |
| 频繁两端操作 | deque | list |
| 优先级队列、Top-K | heapq | sorted list |
| 有序列表二分查找 | bisect | 线性查找 |

---

## 7. 性能对比汇总表

### 7.1 序列类型性能对比（单位: ms）

| 结构 | 创建 (100k元素) | 随机访问 (10万次) | 迭代求和 | 内存占用估计 |
|------|----------------|------------------|----------|-------------|
| **list** | 77.56 | 3.68 | 238.96 | ~800KB + 对象开销 |
| **tuple** | 78.90 | 3.19 | 239.26 | ~720KB + 对象开销 |
| **array('i')** | 264.46 | 5.23 | 336.62 | ~400KB |

**关键发现**: tuple访问略快，但三者迭代性能接近；array内存节省约50%

### 7.2 字典类型性能对比

| 结构 | 创建 (10k键值对×100) | 查找 (11万次) | 插入+删除 (5万次) |
|------|---------------------|--------------|-----------------|
| **dict** | 77.00 | 9.88 | 21.11 |
| **OrderedDict** | 159.86 | 10.19 | 24.97 |
| **defaultdict** | 100.80 | 10.05 | 21.12 |

**关键发现**: dict性能最优；OrderedDict创建约慢1倍；三者查找性能接近

### 7.3 成员检查与去重性能

| 结构 | 成员检查 (10万次) | 去重 (100k元素×50) | 复杂度 |
|------|------------------|---------------------|--------|
| **set** | 5.63 | 177.90 | O(1) |
| **list** | 90223.73 | 348.51 | O(n) |

**关键发现**: set成员检查比list快约**16000倍**！

### 7.4 两端操作性能对比

| 操作 | deque | list | 速度倍数 |
|------|-------|------|---------|
| **左端 append** (10万次) | 0.55 ms | 2.45 ms | deque 快 **4.4x** |
| **左端 pop** (10万次) | 0.53 ms | 11.40 ms | deque 快 **21.6x** |
| **右端 append** (10万次) | 5.40 ms | 5.28 ms | 相近 |

**关键发现**: deque在左端操作有压倒性优势！

### 7.5 堆与二分查找性能

| 操作 | heapq | sorted list / 其他 |
|------|-------|-------------------|
| **构建 10k 元素** | 3.81 ms | 12.28 ms (sorted) |
| **弹出 100 个最小** | 0.88 ms | 2.93 ms (list.pop(0)) |
| **1000 次查找** | O(n) 遍历 | 25.64 ms (bisect) vs 568.00 ms (线性) |
| **100 次有序插入** | - | 3.66 ms (bisect.insort) vs 85.41 ms (append+sort) |

**关键发现**: 
- heapq构建堆比sorted快3.2倍
- bisect二分查找比线性查找快22倍
- bisect.insort比每次重新排序快23倍

---

## 最佳实践总结

### 性能优化 Checklist

1. **成员检查永远用 set**
   - `x in set` 是 O(1)，`x in list` 是 O(n)
   - 大型数据集差距可达数千倍

2. **两端操作用 deque**
   - 队列、栈、滑动窗口场景
   - `popleft()` 和 `appendleft()` 都是 O(1)

3. **数值计算考虑 array**
   - 大量相同类型数值数据
   - 内存更省，计算更快

4. **不可变数据用 tuple**
   - 作为字典键
   - 保护数据不被修改
   - 可能有微小性能提升

5. **Top-K 问题用 heapq**
   - `heapq.nlargest(k, data)` 无需完全排序
   - 优先级队列天然适合

6. **有序查找用 bisect**
   - 已排序的 list 用二分查找
   - 比线性查找快得多

### 常见反模式

❌ **错误**
```python
# 用 list 作为队列（popleft 是 O(n)）
queue = []
queue.append(item)
item = queue.pop(0)  # 慢！
```

✅ **正确**
```python
from collections import deque
queue = deque()
queue.append(item)
item = queue.popleft()  # O(1)
```

---

❌ **错误**
```python
# 线性查找成员
for item in items:
    if item not in seen_list:  # O(n) 每次！
        seen_list.append(item)
```

✅ **正确**
```python
seen = set()
for item in items:
    if item not in seen:  # O(1)
        seen.add(item)
```

---

❌ **错误**
```python
# 先排序再取前K个（O(n log n)）
top_k = sorted(data)[:k]
```

✅ **正确**
```python
import heapq
top_k = heapq.nsmallest(k, data)  # 更高效
```

---

## 运行基准测试

执行以下命令运行完整的性能基准测试：

```bash
cd Speed-Demon-Python/03_optimization
python data_structures_benchmark.py
```

测试结果将显示所有数据结构在不同操作下的实际性能对比。
