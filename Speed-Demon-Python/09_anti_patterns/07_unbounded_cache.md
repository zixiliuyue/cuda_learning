# 07 - 无界缓存内存泄漏

## 问题描述

无界缓存是指没有大小限制的缓存，它会随着时间不断增长，最终耗尽所有可用内存，导致 OOM（Out of Memory）错误。这是 Web 应用和长时间运行的服务中最常见的内存泄漏原因之一。

## 示例代码

```python
# 反模式：无界缓存
cache = {}

def get_user(user_id):
    if user_id not in cache:
        # 模拟数据库查询
        cache[user_id] = fetch_user_from_db(user_id)
    return cache[user_id]

# 当服务运行时间越长，访问的 user_id 越多，cache 字典就越大
```

## 性能对比

运行 `python 07_unbounded_cache.py` 会输出类似：

```
=== 无界缓存内存增长 ===
第 1 次循环后缓存大小: 1000 条, 内存使用: 0.37 MB
第 2 次循环后缓存大小: 2000 条, 内存使用: 0.75 MB
第 3 次循环后缓存大小: 3000 条, 内存使用: 1.12 MB
...
第 10 次循环后缓存大小: 10000 条, 内存使用: 3.75 MB

内存使用随条目数量线性增长，没有上限！

=== LRU 缓存内存使用 ===
LRU 缓存大小保持在: 1000 条
内存使用稳定在: 0.37 MB
```

## 为什么会这样？

1. **字典无限增长**：Python 字典没有自动清理机制
2. **引用永远存在**：只要服务在运行，缓存对象就不会被垃圾回收
3. **长尾效应**：即使 99% 的缓存项都不会再被访问，它们仍然占用内存
4. **内存碎片化**：频繁的内存分配和释放会导致内存碎片化

## 修复方案

### 方案 1：使用 LRU 缓存

```python
from functools import lru_cache

@lru_cache(maxsize=1000)  # 限制最大缓存数量
def get_user(user_id):
    return fetch_user_from_db(user_id)
```

### 方案 2：使用带过期时间的缓存

```python
from cachetools import TTLCache

# 最大 1000 条，过期时间 5 分钟
cache = TTLCache(maxsize=1000, ttl=300)

def get_user(user_id):
    if user_id not in cache:
        cache[user_id] = fetch_user_from_db(user_id)
    return cache[user_id]
```

### 方案 3：使用 LFU 缓存（基于访问频率）

```python
from cachetools import LFUCache

# 最大 1000 条，优先淘汰最不常访问的
cache = LFUCache(maxsize=1000)
```

### 方案 4：使用外部缓存服务

```python
import redis

r = redis.Redis(host='localhost', port=6379, db=0)

def get_user(user_id):
    key = f"user:{user_id}"
    cached = r.get(key)
    if cached:
        return json.loads(cached)
    
    user = fetch_user_from_db(user_id)
    r.setex(key, 300, json.dumps(user))  # 5 分钟过期
    return user
```

## 缓存最佳实践

### 1. 设置合理的缓存大小

```python
# ❌ 错误
cache = {}  # 无限大

# ✅ 正确
from functools import lru_cache

@lru_cache(maxsize=100)  # 小而高效的缓存
def expensive_function(x):
    pass
```

### 2. 设置过期时间

```python
from datetime import datetime, timedelta

class TimedCache:
    def __init__(self, ttl_seconds=300):
        self.cache = {}
        self.ttl = timedelta(seconds=ttl_seconds)
    
    def get(self, key):
        if key not in self.cache:
            return None
        
        value, timestamp = self.cache[key]
        if datetime.now() - timestamp > self.ttl:
            del self.cache[key]
            return None
        
        return value
    
    def set(self, key, value):
        self.cache[key] = (value, datetime.now())
```

### 3. 定期清理缓存

```python
import threading
import time

def clean_cache_periodically(cache, interval=60):
    def cleaner():
        while True:
            time.sleep(interval)
            # 清理过期条目
            now = time.time()
            expired = [k for k, (v, t) in cache.items() if now - t > 300]
            for k in expired:
                del cache[k]
    
    thread = threading.Thread(target=cleaner, daemon=True)
    thread.start()
```

### 4. 监控缓存命中率

```python
class MonitoredCache:
    def __init__(self, maxsize=1000):
        self.cache = {}
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0
```

## 不同缓存策略对比

| 策略 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| LRU | 实现简单，内存固定 | 可能淘汰很快会用到的 | 通用场景 |
| LFU | 保留高频访问数据 | 可能保留历史热门但现在冷的 | 访问频率稳定的场景 |
| TTL | 自动过期，数据新鲜 | 过期时间需要调优 | 需要数据新鲜度的场景 |
| Redis | 分布式，持久化 | 有网络开销，需要额外服务 | 分布式系统 |

## 总结

1. **永远不要使用无界缓存** - 字典做缓存是危险的
2. **选择合适的缓存策略** - LRU, LFU, TTL 各有适用场景
3. **监控缓存命中率** - 命中率太低说明缓存没用
4. **考虑外部缓存** - 对于分布式系统，使用 Redis/Memcached
5. **小心内存泄漏** - 定期检查内存使用情况

记住：**缓存不是万能的**，有时候不缓存比错误的缓存更好！
