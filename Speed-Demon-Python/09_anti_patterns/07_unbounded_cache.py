import time
import sys
from functools import lru_cache
from collections import OrderedDict

def get_memory_usage(obj):
    """计算对象的粗略内存使用（MB）"""
    return sys.getsizeof(obj) / 1024 / 1024

def simulate_unbounded_cache():
    """模拟无界缓存"""
    cache = {}
    
    print("=== 无界缓存内存增长 ===")
    for iteration in range(1, 11):
        for i in range(1000):
            key = f"user_{(iteration - 1) * 1000 + i}"
            cache[key] = {"id": i, "data": "x" * 100}
        
        mem_usage = sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in cache.items())
        mem_usage_mb = mem_usage / 1024 / 1024
        print(f"第 {iteration} 次循环后缓存大小: {len(cache)} 条, 内存使用: {mem_usage_mb:.2f} MB")
    
    return cache

class LRUCache:
    """简单的 LRU 缓存实现"""
    def __init__(self, maxsize=1000):
        self.cache = OrderedDict()
        self.maxsize = maxsize
    
    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def set(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)
        self.cache[key] = value
    
    def __len__(self):
        return len(self.cache)

def simulate_lru_cache():
    """模拟 LRU 缓存"""
    cache = LRUCache(maxsize=1000)
    
    print("\n=== LRU 缓存内存使用 ===")
    for iteration in range(1, 11):
        for i in range(1000):
            key = f"user_{(iteration - 1) * 1000 + i}"
            cache.set(key, {"id": i, "data": "x" * 100})
        
        mem_usage = sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in cache.cache.items())
        mem_usage_mb = mem_usage / 1024 / 1024
        print(f"第 {iteration} 次循环后缓存大小: {len(cache)} 条, 内存使用: {mem_usage_mb:.2f} MB")
    
    return cache

class TTLCache:
    """带过期时间的缓存"""
    def __init__(self, maxsize=1000, ttl=2):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.ttl = ttl
    
    def get(self, key):
        if key not in self.cache:
            return None
        
        value, timestamp = self.cache[key]
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            return None
        
        self.cache.move_to_end(key)
        return value
    
    def set(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)
        self.cache[key] = (value, time.time())
    
    def __len__(self):
        return len(self.cache)

def simulate_ttl_cache():
    """模拟 TTL 缓存"""
    cache = TTLCache(maxsize=1000, ttl=1)
    
    print("\n=== TTL 缓存过期演示 ===")
    for i in range(5):
        cache.set(f"key_{i}", f"value_{i}")
        print(f"添加 key_{i} 后缓存大小: {len(cache)}")
    
    print("\n等待 2 秒让缓存过期...")
    time.sleep(2)
    
    for i in range(5):
        value = cache.get(f"key_{i}")
        print(f"获取 key_{i}: {value}")
    
    print(f"过期后缓存大小: {len(cache)}")

@lru_cache(maxsize=100)
def fibonacci(n):
    """使用 functools.lru_cache 的示例"""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def compare_cache_performance():
    """比较不同缓存策略的性能"""
    print("\n=== 缓存性能对比 ===")
    
    unbounded = {}
    lru = LRUCache(maxsize=100)
    
    start = time.time()
    for i in range(10000):
        key = f"key_{i % 200}"
        if key not in unbounded:
            unbounded[key] = "x" * 100
        _ = unbounded[key]
    unbounded_time = time.time() - start
    
    start = time.time()
    for i in range(10000):
        key = f"key_{i % 200}"
        if lru.get(key) is None:
            lru.set(key, "x" * 100)
        _ = lru.get(key)
    lru_time = time.time() - start
    
    print(f"无界缓存耗时: {unbounded_time:.4f} 秒")
    print(f"LRU 缓存耗时: {lru_time:.4f} 秒")
    print(f"性能差异: {unbounded_time / lru_time:.2f}x")
    print(f"无界缓存条目数: {len(unbounded)}")
    print(f"LRU 缓存条目数: {len(lru)}")

if __name__ == "__main__":
    unbounded_cache = simulate_unbounded_cache()
    
    lru_cache = simulate_lru_cache()
    
    simulate_ttl_cache()
    
    compare_cache_performance()
    
    print("\n=== functools.lru_cache 演示 ===")
    result = fibonacci(30)
    print(f"fibonacci(30) = {result}")
    print(f"缓存命中率信息: {fibonacci.cache_info()}")
    
    print("\n=== 总结 ===")
    print("无界缓存问题: 内存无限增长 → 最终 OOM")
    print("LRU 缓存优点: 内存固定 → 可预测，安全")
    print("TTL 缓存优点: 自动过期 → 数据新鲜")
