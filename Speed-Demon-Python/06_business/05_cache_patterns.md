# 缓存设计模式详解

缓存是提升系统性能的核心技术之一。本文详细讲解6大类缓存设计模式，包含问题描述、Python代码实现、适用场景、优缺点对比。

## 目录
1. [缓存穿透](#1-缓存穿透)
2. [缓存击穿](#2-缓存击穿)
3. [缓存雪崩](#3-缓存雪崩)
4. [缓存一致性](#4-缓存一致性)
5. [本地缓存vs分布式缓存](#5-本地缓存vs分布式缓存)
6. [缓存预热和降级](#6-缓存预热和降级)

---

## 1. 缓存穿透

### 问题描述
缓存穿透是指查询一个**不存在的数据**，缓存和数据库都没有命中，导致每次查询都要打到数据库上。

**危害**：恶意攻击者可以利用不存在的key发起大量请求，直接压垮数据库。

### 解决方案一：布隆过滤器

#### 原理
布隆过滤器是一种空间效率极高的概率型数据结构，用于判断一个元素**可能存在**或**一定不存在**。

#### Python代码实现
```python
import mmh3
import math

class BloomFilter:
    def __init__(self, capacity: int, error_rate: float = 0.01):
        self.capacity = capacity
        self.error_rate = error_rate
        self.bit_size = self._get_bit_size()
        self.hash_count = self._get_hash_count()
        self.bit_array = [0] * self.bit_size
    
    def _get_bit_size(self) -> int:
        return int(-(self.capacity * math.log(self.error_rate)) / (math.log(2) ** 2))
    
    def _get_hash_count(self) -> int:
        return int((self.bit_size / self.capacity) * math.log(2))
    
    def add(self, key: str):
        for seed in range(self.hash_count):
            index = mmh3.hash(key, seed) % self.bit_size
            self.bit_array[index] = 1
    
    def might_contain(self, key: str) -> bool:
        for seed in range(self.hash_count):
            index = mmh3.hash(key, seed) % self.bit_size
            if self.bit_array[index] == 0:
                return False
        return True

# 使用示例
def get_data_with_bloom_filter(user_id: str, bloom_filter: BloomFilter, cache: dict, db: dict):
    if not bloom_filter.might_contain(user_id):
        return None
    if user_id in cache:
        return cache[user_id]
    data = db.get(user_id)
    if data:
        cache[user_id] = data
    return data
```

#### 适用场景
- 数据量巨大且key的数量已知
- 可以容忍少量误判（误判率可配置）
- 防止恶意爬虫或攻击

#### 优缺点对比
| 优点 | 缺点 |
|------|------|
| 空间效率极高（节省内存） | 存在误判率（可能把不存在的数据判断为存在） |
| 查询速度O(k)，k为哈希函数数量 | 不支持删除操作（标准布隆过滤器） |
| 适合大数据量场景 | 需要预先估计数据量 |

---

### 解决方案二：空值缓存

#### 原理
当查询的数据不存在时，在缓存中存储一个空值（如None），并设置较短的过期时间。

#### Python代码实现
```python
import time
from typing import Any, Optional

class NullCache:
    def __init__(self, null_ttl: int = 60):
        self.cache = {}
        self.null_ttl = null_ttl
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        value, expire_time = self.cache[key]
        if time.time() > expire_time:
            del self.cache[key]
            return None
        return value
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        expire_time = time.time() + ttl
        self.cache[key] = (value, expire_time)
    
    def set_null(self, key: str):
        expire_time = time.time() + self.null_ttl
        self.cache[key] = (None, expire_time)

# 使用示例
def get_data_with_null_cache(key: str, cache: NullCache, db: dict):
    cached_data = cache.get(key)
    if key in cache.cache:
        return cached_data
    db_data = db.get(key)
    if db_data:
        cache.set(key, db_data)
    else:
        cache.set_null(key)
    return db_data
```

#### 适用场景
- 简单场景，不需要复杂的布隆过滤器
- 空查询较多但不频繁的场景

#### 优缺点对比
| 优点 | 缺点 |
|------|------|
| 实现简单 | 需要额外内存存储空值 |
| 无需预先知道数据量 | 可能导致缓存中存在大量无效key |
| 可以有效防止穿透 | 空值过期后仍可能穿透 |

---

## 2. 缓存击穿

### 问题描述
缓存击穿是指一个**热点key**在缓存过期的瞬间，同时有大量并发请求打到数据库上。

**危害**：数据库瞬间压力剧增，可能导致宕机。

### 解决方案一：互斥锁

#### 原理
当缓存失效时，只允许一个请求去查询数据库并重建缓存，其他请求等待。

#### Python代码实现
```python
import threading
import time
from typing import Any, Optional

class MutexCache:
    def __init__(self):
        self.cache = {}
        self.locks = {}
        self.global_lock = threading.Lock()
    
    def _get_lock(self, key: str) -> threading.Lock:
        with self.global_lock:
            if key not in self.locks:
                self.locks[key] = threading.Lock()
            return self.locks[key]
    
    def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        self.cache[key] = (value, time.time() + ttl)
    
    def get_with_mutex(self, key: str, load_from_db, ttl: int = 3600) -> Any:
        data = self.get(key)
        if data is not None:
            value, expire_time = data
            if time.time() < expire_time:
                return value
        
        lock = self._get_lock(key)
        with lock:
            data = self.get(key)
            if data is not None:
                value, expire_time = data
                if time.time() < expire_time:
                    return value
            
            db_data = load_from_db(key)
            if db_data is not None:
                self.set(key, db_data, ttl)
            return db_data

# 使用示例
def query_database(user_id: str):
    print(f"查询数据库: {user_id}")
    return {"id": user_id, "name": f"用户{user_id}"}

cache = MutexCache()
result = cache.get_with_mutex("user_1001", query_database)
```

#### 适用场景
- 热点key较多的场景
- 数据库查询成本较高
- 可以容忍少量请求等待

#### 优缺点对比
| 优点 | 缺点 |
|------|------|
| 有效防止数据库压力骤增 | 存在死锁风险 |
| 实现相对简单 | 锁开销可能影响性能 |
| 数据一致性较好 | 热点key仍可能导致阻塞 |

---

### 解决方案二：永不过期（逻辑过期）

#### 原理
热点key不设置物理过期时间，而是在value中存储一个逻辑过期时间，通过后台异步线程更新缓存。

#### Python代码实现
```python
import threading
import time
from typing import Any, Callable

class LogicalExpireCache:
    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()
    
    def set(self, key: str, value: Any, logical_ttl: int = 3600):
        self.cache[key] = {
            "value": value,
            "expire_time": time.time() + logical_ttl,
            "updating": False
        }
    
    def get(self, key: str, load_from_db: Callable, logical_ttl: int = 3600) -> Any:
        if key not in self.cache:
            return None
        
        cache_data = self.cache[key]
        current_time = time.time()
        
        if current_time < cache_data["expire_time"]:
            return cache_data["value"]
        
        if not cache_data["updating"]:
            with self.lock:
                if not cache_data["updating"]:
                    cache_data["updating"] = True
                    
                    def async_update():
                        try:
                            new_value = load_from_db(key)
                            cache_data["value"] = new_value
                            cache_data["expire_time"] = time.time() + logical_ttl
                        finally:
                            cache_data["updating"] = False
                    
                    threading.Thread(target=async_update, daemon=True).start()
        
        return cache_data["value"]

# 使用示例
cache = LogicalExpireCache()
cache.set("hot_key", {"data": "value"}, logical_ttl=60)
result = cache.get("hot_key", query_database)
```

#### 适用场景
- 超高并发场景（秒杀、抢购）
- 绝对不能让数据库承受压力
- 可以容忍短暂的数据不一致

#### 优缺点对比
| 优点 | 缺点 |
|------|------|
| 完全避免数据库压力 | 占用内存较大（缓存永不过期） |
| 性能最好 | 存在数据不一致窗口 |
| 适合热点key场景 | 需要后台异步更新机制 |

---

## 3. 缓存雪崩

### 问题描述
缓存雪崩是指**大量key在同一时间失效**，导致所有请求都打到数据库上。

**危害**：数据库压力瞬间达到峰值，可能导致服务不可用。

### 解决方案一：过期时间打散

#### 原理
在设置缓存过期时间时，增加一个随机值，避免大量key同时过期。

#### Python代码实现
```python
import random
import time
from typing import Any

class RandomizedExpireCache:
    def __init__(self, base_ttl: int = 3600, random_range: int = 600):
        self.cache = {}
        self.base_ttl = base_ttl
        self.random_range = random_range
    
    def _get_random_ttl(self) -> int:
        return self.base_ttl + random.randint(0, self.random_range)
    
    def set(self, key: str, value: Any):
        ttl = self._get_random_ttl()
        self.cache[key] = (value, time.time() + ttl)
    
    def get(self, key: str) -> Any:
        if key not in self.cache:
            return None
        value, expire_time = self.cache[key]
        if time.time() > expire_time:
            del self.cache[key]
            return None
        return value
    
    def mset(self, items: dict):
        for key, value in items.items():
            self.set(key, value)

# 使用示例
cache = RandomizedExpireCache(base_ttl=3600, random_range=600)
items = {f"key_{i}": f"value_{i}" for i in range(1000)}
cache.mset(items)
```

#### 适用场景
- 批量设置缓存的场景
- 数据时效性要求不高
- 需要平滑数据库负载

#### 优缺点对比
| 优点 | 缺点 |
|------|------|
| 实现简单 | 部分数据过期时间不准确 |
| 有效分散过期时间 | 不能完全消除风险 |
| 性能开销小 | 对热点key效果有限 |

---

### 解决方案二：多级缓存

#### 原理
使用L1（本地缓存）+ L2（分布式缓存）多级架构，降低数据库压力。

#### Python代码实现
```python
import time
from typing import Any, Optional
from functools import lru_cache

class MultiLevelCache:
    def __init__(self, l1_maxsize: int = 1000, l1_ttl: int = 60):
        self.l1_ttl = l1_ttl
        self.l1_cache = {}
        self.l2_cache = {}
    
    @lru_cache(maxsize=1000)
    def _l1_get(self, key: str):
        return self.l1_cache.get(key)
    
    def get(self, key: str, load_from_db) -> Any:
        l1_data = self._l1_get(key)
        if l1_data:
            value, expire_time = l1_data
            if time.time() < expire_time:
                return value
        
        if key in self.l2_cache:
            value, expire_time = self.l2_cache[key]
            if time.time() < expire_time:
                self.l1_cache[key] = (value, time.time() + self.l1_ttl)
                return value
        
        db_data = load_from_db(key)
        if db_data:
            self.l2_cache[key] = (db_data, time.time() + 3600)
            self.l1_cache[key] = (db_data, time.time() + self.l1_ttl)
        return db_data

# 使用示例
ml_cache = MultiLevelCache()
result = ml_cache.get("user_1001", query_database)
```

#### 适用场景
- 高并发分布式系统
- 读多写少场景
- 需要分层保护数据库

#### 优缺点对比
| 优点 | 缺点 |
|------|------|
| 多级保护，安全性高 | 实现复杂度高 |
| L1缓存速度极快 | 数据一致性问题更复杂 |
| 可以应对不同层级的故障 | 维护成本高 |

---

## 4. 缓存一致性

### 4.1 Cache Aside Pattern（旁路缓存）

#### 原理
- 读：先读缓存，缓存未命中则读数据库，然后写入缓存
- 写：先更新数据库，再删除缓存

#### Python代码实现
```python
class CacheAsidePattern:
    def __init__(self):
        self.cache = {}
        self.db = {}
    
    def read(self, key: str):
        if key in self.cache:
            return self.cache[key]
        data = self.db.get(key)
        if data:
            self.cache[key] = data
        return data
    
    def write(self, key: str, value: any):
        self.db[key] = value
        if key in self.cache:
            del self.cache[key]
```

#### 适用场景
- 最常用的缓存模式
- 读多写少场景
- 一般业务系统

#### 优缺点对比
| 优点 | 缺点 |
|------|------|
| 实现简单 | 存在短暂数据不一致 |
| 数据一致性较好 | 并发写场景可能有问题 |
| 易于维护 | 需要处理缓存删除失败 |

---

### 4.2 Read/Write Through Pattern（读写穿透）

#### 原理
- 读：缓存未命中时，由缓存服务负责加载数据并写入缓存
- 写：先写缓存，由缓存服务同步写数据库

#### Python代码实现
```python
class ReadWriteThroughCache:
    def __init__(self, db):
        self.cache = {}
        self.db = db
    
    def read(self, key: str):
        if key in self.cache:
            return self.cache[key]
        data = self.db.get(key)
        if data:
            self.cache[key] = data
        return data
    
    def write(self, key: str, value: any):
        self.cache[key] = value
        self.db[key] = value
```

#### 适用场景
- 对数据一致性要求高
- 写操作较多的场景
- 需要缓存提供者支持

#### 优缺点对比
| 优点 | 缺点 |
|------|------|
| 数据一致性好 | 写性能较差（同步写数据库） |
| 应用层逻辑简单 | 需要缓存服务支持 |
| 避免缓存和数据库不一致 | 实现复杂度高 |

---

### 4.3 Write Back Pattern（写回）

#### 原理
- 写：只写缓存，异步批量写回数据库
- 读：缓存未命中时触发写回和加载

#### Python代码实现
```python
import threading
import time

class WriteBackCache:
    def __init__(self, db, flush_interval: int = 5):
        self.cache = {}
        self.dirty_keys = set()
        self.db = db
        self.flush_interval = flush_interval
        self._start_flush_thread()
    
    def _start_flush_thread(self):
        def flush_loop():
            while True:
                time.sleep(self.flush_interval)
                self._flush_dirty_keys()
        
        threading.Thread(target=flush_loop, daemon=True).start()
    
    def _flush_dirty_keys(self):
        for key in list(self.dirty_keys):
            if key in self.cache:
                self.db[key] = self.cache[key]
                self.dirty_keys.discard(key)
    
    def read(self, key: str):
        return self.cache.get(key) or self.db.get(key)
    
    def write(self, key: str, value: any):
        self.cache[key] = value
        self.dirty_keys.add(key)
```

#### 适用场景
- 写操作极多的场景
- 可以容忍数据丢失
- 性能要求极高的统计类业务

#### 优缺点对比
| 优点 | 缺点 |
|------|------|
| 写性能极佳 | 存在数据丢失风险 |
| I/O合并，减少数据库压力 | 数据一致性最差 |
| 适合高频写场景 | 实现最复杂 |

---

## 5. 本地缓存vs分布式缓存

### 5.1 本地缓存（lru_cache）

#### Python代码实现
```python
from functools import lru_cache
import time

@lru_cache(maxsize=1000)
def get_user_data(user_id: str):
    time.sleep(0.1)
    return {"id": user_id, "name": f"用户{user_id}"}

# 手动实现LRU缓存
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key: str):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: any):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

#### 适用场景
- 单机应用
- 数据量小
- 访问频率极高
- 不需要跨机器共享

#### 优缺点对比
| 优点 | 缺点 |
|------|------|
| 速度极快（内存访问） | 容量受限于单机内存 |
| 无网络开销 | 多机数据不一致 |
| 实现简单 | 无法水平扩展 |

---

### 5.2 分布式缓存（Redis）

#### Python代码实现
```python
import redis
import json
from typing import Any, Optional

class RedisCache:
    def __init__(self, host: str = "localhost", port: int = 6379):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
    
    def get(self, key: str) -> Optional[Any]:
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        self.client.setex(key, ttl, json.dumps(value))
    
    def delete(self, key: str):
        self.client.delete(key)
    
    def exists(self, key: str) -> bool:
        return self.client.exists(key) > 0
```

#### 适用场景
- 分布式系统
- 大数据量
- 需要跨机器共享
- 需要持久化

#### 优缺点对比
| 优点 | 缺点 |
|------|------|
| 可水平扩展 | 有网络开销 |
| 多机数据一致 | 速度比本地缓存慢 |
| 支持持久化 | 需要维护Redis集群 |
| 丰富的功能 | 有运维成本 |

---

### 5.3 对比总结

| 维度 | 本地缓存 (lru_cache) | 分布式缓存 (Redis) |
|------|---------------------|-------------------|
| **访问速度** | 极快 (~100ns) | 较快 (~1ms) |
| **容量限制** | 单机内存 | 可集群扩展 |
| **数据共享** | 仅本进程/机器 | 跨多机共享 |
| **网络开销** | 无 | 有 |
| **持久化** | 不支持 | 支持 |
| **实现复杂度** | 极低 | 中等 |
| **运维成本** | 无 | 较高 |
| **适用场景** | 单机、小数据、超高频 | 分布式、大数据、共享 |

---

## 6. 缓存预热和降级

### 6.1 缓存预热

#### 原理
系统启动或低峰期，提前将热点数据加载到缓存中。

#### Python代码实现
```python
import threading
from typing import List, Callable

class CacheWarmer:
    def __init__(self, cache):
        self.cache = cache
    
    def warm_up(self, hot_keys: List[str], load_data: Callable):
        print(f"开始预热 {len(hot_keys)} 个热点key...")
        for key in hot_keys:
            data = load_data(key)
            if data:
                self.cache[key] = data
    
    def async_warm_up(self, hot_keys: List[str], load_data: Callable):
        thread = threading.Thread(target=self.warm_up, args=(hot_keys, load_data))
        thread.daemon = True
        thread.start()

# 使用示例
def load_hot_data(key: str):
    return {"id": key, "data": "热点数据"}

cache = {}
warmer = CacheWarmer(cache)
hot_keys = [f"hot_{i}" for i in range(100)]
warmer.async_warm_up(hot_keys, load_hot_data)
```

#### 适用场景
- 系统启动时
- 促销活动开始前
- 已知热点数据的场景

#### 优缺点对比
| 优点 | 缺点 |
|------|------|
| 避免启动时缓存穿透 | 需要预测热点数据 |
| 提升用户体验 | 预热数据可能不准确 |
| 平滑数据库压力 | 预热占用启动时间 |

---

### 6.2 缓存降级

#### 原理
当缓存或数据库不可用时，提供降级方案，返回默认数据或旧数据。

#### Python代码实现
```python
import time
from typing import Any, Callable

class CacheDegradation:
    def __init__(self):
        self.backup_cache = {}
        self.fuse_count = {}
        self.fuse_threshold = 10
    
    def get_with_degradation(self, key: str, primary_get: Callable, backup_data: Any = None) -> Any:
        try:
            result = primary_get(key)
            if result is not None:
                self.backup_cache[key] = (result, time.time())
                return result
        except Exception as e:
            print(f"主缓存获取失败: {e}")
        
        if key in self.backup_cache:
            data, timestamp = self.backup_cache[key]
            print(f"使用降级数据，数据新鲜度: {time.time() - timestamp:.1f}秒")
            return data
        
        return backup_data
    
    def circuit_breaker(self, key: str, operation: Callable) -> Any:
        if self.fuse_count.get(key, 0) >= self.fuse_threshold:
            print(f"熔断器触发，跳过{key}")
            return None
        
        try:
            result = operation()
            self.fuse_count[key] = 0
            return result
        except Exception as e:
            self.fuse_count[key] = self.fuse_count.get(key, 0) + 1
            raise e

# 使用示例
def unreliable_cache_get(key: str):
    if time.time() % 5 < 1:
        raise Exception("缓存不可用")
    return {"data": "正常数据"}

degrader = CacheDegradation()
for i in range(10):
    result = degrader.get_with_degradation("key_1", unreliable_cache_get, {"data": "默认值"})
    print(result)
```

#### 适用场景
- 高可用要求的系统
- 核心业务流程
- 依赖外部服务的场景

#### 优缺点对比
| 优点 | 缺点 |
|------|------|
| 提高系统可用性 | 可能返回过期数据 |
| 防止故障扩散 | 实现复杂度高 |
| 用户体验更友好 | 需要维护备份数据 |

---

## 总结：缓存模式选择指南

| 问题场景 | 推荐方案 | 实现难度 | 性能效果 |
|---------|---------|---------|---------|
| **不存在的key查询** | 布隆过滤器 + 空值缓存 | 中 | 好 |
| **热点key失效** | 互斥锁 + 逻辑过期 | 中 | 极好 |
| **大量key同时失效** | 过期时间打散 + 多级缓存 | 中 | 好 |
| **读多写少业务** | Cache Aside | 低 | 好 |
| **数据一致性要求高** | Read/Write Through | 高 | 中 |
| **写操作极多** | Write Back | 高 | 极好 |
| **单机高频访问** | lru_cache本地缓存 | 低 | 极好 |
| **分布式共享数据** | Redis分布式缓存 | 中 | 好 |
| **系统启动/活动前** | 缓存预热 | 低 | 好 |
| **高可用要求场景** | 缓存降级 + 熔断 | 高 | 好 |

---

**最佳实践建议**：
1. 优先使用Cache Aside模式，这是最成熟的方案
2. 热点key考虑使用逻辑过期或互斥锁
3. 布隆过滤器用于防止恶意穿透
4. 多级缓存架构提供分层保护
5. 必须考虑缓存降级和熔断机制
6. 监控缓存命中率、过期key数量等指标
