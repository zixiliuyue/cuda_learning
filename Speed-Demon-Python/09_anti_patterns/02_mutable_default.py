import sys

def add_item_bad(item, items=[]):
    items.append(item)
    return items

def add_item_good(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items

if __name__ == "__main__":
    print("=== 反模式演示 ===")
    for i in range(1, 6):
        result = add_item_bad(i)
        print(f"第 {i} 次调用结果: {result}")
        print(f"列表对象ID: {id(result)}")
    
    print(f"\n最终列表长度: {len(result)}")
    print(f"内存使用: {sys.getsizeof(result)} bytes")
    
    print("\n=== 修复方案演示 ===")
    for i in range(1, 6):
        result = add_item_good(i)
        print(f"第 {i} 次调用结果: {result}")
        print(f"列表对象ID: {id(result)}")
    
    print(f"\n最终列表长度: {len(result)}")
    print(f"内存使用: {sys.getsizeof(result)} bytes")
    
    print("\n=== 大量调用对比 ===")
    import gc
    gc.collect()
    
    for i in range(10000):
        add_item_bad(i)
    
    print(f"10000次调用后，默认列表长度: {len(add_item_bad.__defaults__[0])}")
    print(f"内存使用: {sys.getsizeof(add_item_bad.__defaults__[0])} bytes")
