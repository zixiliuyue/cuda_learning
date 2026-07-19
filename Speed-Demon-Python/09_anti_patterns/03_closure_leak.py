import sys
import gc
from functools import partial

def create_handler_bad():
    big_data = [i * i for i in range(100_000)]
    multiplier = 2
    
    def handler(x):
        return x * multiplier
    
    return handler

def create_handler_good1():
    big_data = [i * i for i in range(100_000)]
    multiplier = 2
    
    del big_data
    
    def handler(x):
        return x * multiplier
    
    return handler

class Handler:
    def __init__(self, multiplier):
        self.multiplier = multiplier
    
    def __call__(self, x):
        return x * self.multiplier

def create_handler_good2():
    big_data = [i * i for i in range(100_000)]
    return Handler(2)

def multiply(x, multiplier):
    return x * multiplier

def create_handler_good3():
    big_data = [i * i for i in range(100_000)]
    return partial(multiply, multiplier=2)

def get_object_size(obj):
    return sys.getsizeof(obj) + sum(sys.getsizeof(item) for item in obj) if isinstance(obj, list) else sys.getsizeof(obj)

if __name__ == "__main__":
    print("=== 反模式演示 ===")
    gc.collect()
    h_bad = create_handler_bad()
    print(f"闭包引用的变量数量: {len(h_bad.__closure__) if h_bad.__closure__ else 0}")
    
    closure_cells = [cell.cell_contents for cell in (h_bad.__closure__ or [])]
    big_data_exists = any(isinstance(cell, list) and len(cell) > 1000 for cell in closure_cells)
    print(f"大对象仍在闭包中: {big_data_exists}")
    
    total_size = sum(get_object_size(cell) for cell in closure_cells)
    print(f"闭包捕获对象总大小: {total_size / 1024 / 1024:.2f} MB")
    
    print("\n=== 修复方案 1: 删除不需要的变量 ===")
    gc.collect()
    h_good1 = create_handler_good1()
    print(f"闭包引用的变量数量: {len(h_good1.__closure__) if h_good1.__closure__ else 0}")
    
    closure_cells = [cell.cell_contents for cell in (h_good1.__closure__ or [])]
    big_data_exists = any(isinstance(cell, list) and len(cell) > 1000 for cell in closure_cells)
    print(f"大对象仍在闭包中: {big_data_exists}")
    
    print("\n=== 修复方案 2: 使用类 ===")
    gc.collect()
    h_good2 = create_handler_good2()
    print(f"对象属性数量: {len(dir(h_good2)) - len(dir(object))}")
    print(f"是否包含大列表: {'big_data' not in dir(h_good2)}")
    print(f"实例内存大小: {sys.getsizeof(h_good2)} bytes")
    
    print("\n=== 修复方案 3: 使用 functools.partial ===")
    gc.collect()
    h_good3 = create_handler_good3()
    print(f"partial 参数: {h_good3.keywords}")
    print(f"partial 内存大小: {sys.getsizeof(h_good3)} bytes")
    
    print("\n=== 功能验证 ===")
    print(f"h_bad(5) = {h_bad(5)}")
    print(f"h_good1(5) = {h_good1(5)}")
    print(f"h_good2(5) = {h_good2(5)}")
    print(f"h_good3(5) = {h_good3(5)}")
