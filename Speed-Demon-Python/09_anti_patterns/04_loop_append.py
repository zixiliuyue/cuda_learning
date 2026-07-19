import time
import sys

def create_list_bad(n):
    result = []
    for i in range(n):
        result.append(i * i)
    return result

def create_list_good1(n):
    return [i * i for i in range(n)]

def create_list_good2(n):
    result = [0] * n
    for i in range(n):
        result[i] = i * i
    return result

def create_list_good3(n):
    return list(map(lambda i: i * i, range(n)))

def create_list_good4(n):
    result = []
    result.extend(i * i for i in range(n))
    return result

def measure_time(func, n, iterations=5):
    times = []
    for _ in range(iterations):
        start = time.time()
        func(n)
        times.append(time.time() - start)
    return sum(times) / len(times)

if __name__ == "__main__":
    n = 1_000_000
    iterations = 5
    
    print(f"=== 创建包含 {n} 个元素的列表 ===")
    print(f"平均 {iterations} 次运行时间\n")
    
    t_bad = measure_time(create_list_bad, n, iterations)
    t_good1 = measure_time(create_list_good1, n, iterations)
    t_good2 = measure_time(create_list_good2, n, iterations)
    t_good3 = measure_time(create_list_good3, n, iterations)
    t_good4 = measure_time(create_list_good4, n, iterations)
    
    print(f"循环 append 耗时: {t_bad:.4f} 秒")
    print(f"列表推导式耗时: {t_good1:.4f} 秒")
    print(f"预分配列表耗时: {t_good2:.4f} 秒")
    print(f"map + list 耗时: {t_good3:.4f} 秒")
    print(f"extend + 生成器耗时: {t_good4:.4f} 秒")
    
    print("\n=== 性能提升比 ===")
    print(f"列表推导式比循环 append 快: {t_bad / t_good1:.2f}x")
    print(f"预分配比循环 append 快: {t_bad / t_good2:.2f}x")
    
    print("\n=== 列表扩容演示 ===")
    lst = []
    prev_size = sys.getsizeof(lst)
    print(f"空列表: {prev_size} bytes")
    
    for i in range(1, 50):
        lst.append(i)
        current_size = sys.getsizeof(lst)
        if current_size != prev_size:
            print(f"{i} 个元素: {current_size} bytes (扩容了!)")
            prev_size = current_size
    
    print("\n=== 验证结果一致性 ===")
    r1 = create_list_bad(10)
    r2 = create_list_good1(10)
    r3 = create_list_good2(10)
    print(f"循环 append 前10个: {r1}")
    print(f"列表推导式前10个: {r2}")
    print(f"预分配前10个: {r3}")
    print(f"结果一致: {r1 == r2 == r3}")
