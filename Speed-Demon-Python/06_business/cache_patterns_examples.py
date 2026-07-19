#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存设计模式示例代码
包含：缓存穿透、缓存击穿、缓存雪崩、缓存一致性、本地缓存vs分布式缓存、缓存预热和降级
"""

import time
import random
import threading
import math
from typing import Any, Optional, Callable, List
from functools import lru_cache
from collections import OrderedDict

# ============================================================================
# 模拟数据库查询
# ============================================================================
def query_database(user_id: str):
    """模拟数据库查询，耗时0.1秒"""
    time.sleep(0.1)
    return {"id": user_id, "name": f"用户{user_id}"}

# ============================================================================
# 1. 缓存穿透 - 布隆过滤器
# ============================================================================
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
    
    def _simple_hash(self, key: str, seed: int) -> int:
        result = seed
        for char in key:
            result = (result * 31 + ord(char)) % self.bit_size
        return result
    
    def add(self, key: str):
        for seed in range(self.hash_count):
            index = self._simple_hash(key, seed)
            self.bit_array[index] = 1
    
    def might_contain(self, key: str) -> bool:
        for seed in range(self.hash_count):
            index = self._simple_hash(key, seed)
            if self.bit_array[index] == 0:
                return False
        return True

def demo_bloom_filter():
    print("\n" + "=" * 60)
    print("演示1: 布隆过滤器 - 防止缓存穿透")
    print("=" * 60)
    
    bloom = BloomFilter(capacity=1000, error_rate=0.01)
    cache = {}
    db = {f"user_{i}": {"id": f"user_{i}"} for i in range(100)}
    
    for key in db.keys():
        bloom.add(key)
    
    test_keys = ["user_50", "user_99", "user_1000", "user_9999"]
    for key in test_keys:
        if not bloom.might_contain(key):
            print(f"key={key} 布隆过滤器判断不存在，直接返回None（防止了缓存穿透）")
        else:
            print(f"key={key} 可能存在，继续查询缓存/数据库")

# ============================================================================
# 2. 缓存穿透 - 空值缓存
# ============================================================================
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

def demo_null_cache():
    print("\n" + "=" * 60)
    print("演示2: 空值缓存 - 防止缓存穿透")
    print("=" * 60)
    
    cache = NullCache(null_ttl=5)
    db = {"user_1": {"id": "user_1"}}
    
    for i in range(3):
        key = "user_not_exists"
        result = cache.get(key)
        if result is None and key not in cache.cache:
            print(f"第{i+1}次查询: 缓存未命中，查询数据库")
            db_data = db.get(key)
            if db_data:
                cache.set(key, db_data)
            else:
                cache.set_null(key)
                print(f"  -> 数据库也不存在，设置空缓存")
        else:
            print(f"第{i+1}次查询: 缓存命中（空值），不需要查数据库")

# ============================================================================
# 3. 缓存击穿 - 互斥锁
# ============================================================================
class MutexCache:
    def __init__(self):
        self.cache = {}
        self.locks = {}
        self.global_lock = threading.Lock()
        self.db_query_count = 0
    
    def _get_lock(self, key: str) -> threading.Lock:
        with self.global_lock:
            if key not in self.locks:
                self.locks[key] = threading.Lock()
            return self.locks[key]
    
    def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        self.cache[key] = (value, time.time() + ttl)
    
    def get_with_mutex(self, key: str, load_from_db: Callable, ttl: int = 3600) -> Any:
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
            
            self.db_query_count += 1
            db_data = load_from_db(key)
            if db_data is not None:
                self.set(key, db_data, ttl)
            return db_data

def demo_mutex_cache():
    print("\n" + "=" * 60)
    print("演示3: 互斥锁 - 防止缓存击穿")
    print("=" * 60)
    
    cache = MutexCache()
    
    def worker():
        result = cache.get_with_mutex("hot_key", lambda k: query_database(k))
    
    threads = [threading.Thread(target=worker) for _ in range(10)]
    
    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    print(f"10个并发请求，数据库只被查询了 {cache.db_query_count} 次")
    print(f"总耗时: {time.time() - start:.3f}秒")

# ============================================================================
# 4. 缓存击穿 - 逻辑过期（永不过期）
# ============================================================================
class LogicalExpireCache:
    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()
        self.db_query_count = 0
    
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
                            self.db_query_count += 1
                            new_value = load_from_db(key)
                            cache_data["value"] = new_value
                            cache_data["expire_time"] = time.time() + logical_ttl
                        finally:
                            cache_data["updating"] = False
                    
                    threading.Thread(target=async_update, daemon=True).start()
        
        return cache_data["value"]

def demo_logical_expire():
    print("\n" + "=" * 60)
    print("演示4: 逻辑过期 - 热点key永不过期")
    print("=" * 60)
    
    cache = LogicalExpireCache()
    cache.set("hot_key", query_database("hot_key"), logical_ttl=1)
    
    print("缓存已设置，逻辑过期时间1秒")
    
    def worker():
        for _ in range(5):
            result = cache.get("hot_key", lambda k: query_database(k), 1)
            time.sleep(0.3)
    
    threads = [threading.Thread(target=worker) for _ in range(5)]
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    print(f"数据库被查询了 {cache.db_query_count} 次（仅异步更新时查询）")

# ============================================================================
# 5. 缓存雪崩 - 过期时间打散
# ============================================================================
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
    
    def get_expire_time(self, key: str) -> Optional[float]:
        if key in self.cache:
            return self.cache[key][1]
        return None

def demo_randomized_expire():
    print("\n" + "=" * 60)
    print("演示5: 过期时间打散 - 防止缓存雪崩")
    print("=" * 60)
    
    cache = RandomizedExpireCache(base_ttl=100, random_range=50)
    
    for i in range(10):
        cache.set(f"key_{i}", f"value_{i}")
    
    print("10个key的过期时间:")
    base_time = time.time()
    for i in range(10):
        expire_time = cache.get_expire_time(f"key_{i}")
        ttl = expire_time - base_time if expire_time else 0
        print(f"  key_{i}: {ttl:.0f}秒后过期")
    
    print("\n可以看到过期时间分散在100-150秒之间，不会同时过期")

# ============================================================================
# 6. 缓存雪崩 - 多级缓存
# ============================================================================
class MultiLevelCache:
    def __init__(self, l1_maxsize: int = 1000, l1_ttl: int = 60):
        self.l1_ttl = l1_ttl
        self.l1_cache = {}
        self.l2_cache = {}
        self.l1_hits = 0
        self.l2_hits = 0
        self.db_hits = 0
    
    def get(self, key: str, load_from_db: Callable) -> Any:
        if key in self.l1_cache:
            value, expire_time = self.l1_cache[key]
            if time.time() < expire_time:
                self.l1_hits += 1
                return value
        
        if key in self.l2_cache:
            value, expire_time = self.l2_cache[key]
            if time.time() < expire_time:
                self.l2_hits += 1
                self.l1_cache[key] = (value, time.time() + self.l1_ttl)
                return value
        
        self.db_hits += 1
        db_data = load_from_db(key)
        if db_data:
            self.l2_cache[key] = (db_data, time.time() + 3600)
            self.l1_cache[key] = (db_data, time.time() + self.l1_ttl)
        return db_data

def demo_multi_level_cache():
    print("\n" + "=" * 60)
    print("演示6: 多级缓存 - L1本地 + L2分布式")
    print("=" * 60)
    
    cache = MultiLevelCache()
    
    keys = ["user_1", "user_2", "user_3"]
    for _ in range(3):
        for key in keys:
            cache.get(key, lambda k: query_database(k))
    
    print(f"L1缓存命中: {cache.l1_hits}次")
    print(f"L2缓存命中: {cache.l2_hits}次")
    print(f"数据库命中: {cache.db_hits}次")

# ============================================================================
# 7. 缓存一致性 - Cache Aside
# ============================================================================
class CacheAsidePattern:
    def __init__(self):
        self.cache = {}
        self.db = {}
    
    def read(self, key: str):
        if key in self.cache:
            print(f"  缓存命中: {key}")
            return self.cache[key]
        print(f"  缓存未命中，查询数据库: {key}")
        data = self.db.get(key)
        if data:
            self.cache[key] = data
        return data
    
    def write(self, key: str, value: any):
        print(f"  更新数据库: {key}")
        self.db[key] = value
        if key in self.cache:
            print(f"  删除缓存: {key}")
            del self.cache[key]

def demo_cache_aside():
    print("\n" + "=" * 60)
    print("演示7: Cache Aside Pattern (旁路缓存)")
    print("=" * 60)
    
    cache = CacheAsidePattern()
    cache.db["user_1"] = {"name": "张三"}
    
    print("第一次读取:")
    result1 = cache.read("user_1")
    
    print("\n第二次读取:")
    result2 = cache.read("user_1")
    
    print("\n更新数据:")
    cache.write("user_1", {"name": "李四"})
    
    print("\n更新后读取:")
    result3 = cache.read("user_1")

# ============================================================================
# 8. 本地缓存 - LRU
# ============================================================================
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
            evicted_key, evicted_value = self.cache.popitem(last=False)
            print(f"  淘汰最久未使用: {evicted_key}")

def demo_lru_cache():
    print("\n" + "=" * 60)
    print("演示8: LRU本地缓存")
    print("=" * 60)
    
    lru = LRUCache(capacity=3)
    
    print("依次添加A, B, C:")
    for key in ["A", "B", "C"]:
        lru.put(key, f"value_{key}")
    print(f"  当前缓存顺序: {list(lru.cache.keys())}")
    
    print("\n访问A（A变为最近使用）:")
    lru.get("A")
    print(f"  当前缓存顺序: {list(lru.cache.keys())}")
    
    print("\n添加D（超过容量，淘汰最久未使用的B）:")
    lru.put("D", "value_D")
    print(f"  当前缓存顺序: {list(lru.cache.keys())}")

# ============================================================================
# 9. 缓存预热
# ============================================================================
class CacheWarmer:
    def __init__(self, cache):
        self.cache = cache
    
    def warm_up(self, hot_keys: List[str], load_data: Callable):
        print(f"开始预热 {len(hot_keys)} 个热点key...")
        for key in hot_keys:
            data = load_data(key)
            if data:
                self.cache[key] = data
        print("预热完成!")

def demo_cache_warmup():
    print("\n" + "=" * 60)
    print("演示9: 缓存预热")
    print("=" * 60)
    
    cache = {}
    warmer = CacheWarmer(cache)
    
    def load_hot_data(key: str):
        return {"id": key, "data": "热点数据"}
    
    hot_keys = [f"hot_{i}" for i in range(5)]
    warmer.warm_up(hot_keys, load_hot_data)
    
    print(f"\n预热后缓存中有 {len(cache)} 个key")
    print(f"缓存key: {list(cache.keys())}")

# ============================================================================
# 10. 缓存降级和熔断
# ============================================================================
class CacheDegradation:
    def __init__(self):
        self.backup_cache = {}
        self.fuse_count = {}
        self.fuse_threshold = 3
    
    def get_with_degradation(self, key: str, primary_get: Callable, backup_data: Any = None) -> Any:
        try:
            result = primary_get(key)
            if result is not None:
                self.backup_cache[key] = (result, time.time())
                return result
        except Exception as e:
            print(f"  主缓存获取失败: {e}")
        
        if key in self.backup_cache:
            data, timestamp = self.backup_cache[key]
            freshness = time.time() - timestamp
            print(f"  使用降级数据，新鲜度: {freshness:.1f}秒")
            return data
        
        return backup_data

def demo_degradation():
    print("\n" + "=" * 60)
    print("演示10: 缓存降级")
    print("=" * 60)
    
    degrader = CacheDegradation()
    
    call_count = [0]
    def unreliable_cache_get(key: str):
        call_count[0] += 1
        if call_count[0] <= 2:
            return {"data": "正常数据"}
        else:
            raise Exception("缓存不可用")
    
    for i in range(5):
        print(f"第{i+1}次请求:")
        result = degrader.get_with_degradation("key_1", unreliable_cache_get, {"data": "默认值"})
        print(f"  结果: {result}")

# ============================================================================
# 主函数：运行所有演示
# ============================================================================
def main():
    print("=" * 60)
    print("缓存设计模式示例代码")
    print("=" * 60)
    
    demos = [
        ("布隆过滤器", demo_bloom_filter),
        ("空值缓存", demo_null_cache),
        ("互斥锁", demo_mutex_cache),
        ("逻辑过期", demo_logical_expire),
        ("过期时间打散", demo_randomized_expire),
        ("多级缓存", demo_multi_level_cache),
        ("Cache Aside", demo_cache_aside),
        ("LRU缓存", demo_lru_cache),
        ("缓存预热", demo_cache_warmup),
        ("缓存降级", demo_degradation),
    ]
    
    print("\n可用的演示:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    
    print("\n运行所有演示...")
    
    for name, demo_func in demos:
        try:
            demo_func()
            time.sleep(0.5)
        except Exception as e:
            print(f"\n演示 '{name}' 出错: {e}")
    
    print("\n" + "=" * 60)
    print("所有演示完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
