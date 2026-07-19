import time
import dis
import math

GLOBAL_MULTIPLIER = 2
GLOBAL_OFFSET = 10

def calculate_bad(numbers):
    result = []
    for num in numbers:
        result.append(num * GLOBAL_MULTIPLIER + GLOBAL_OFFSET)
    return result

def calculate_good(numbers):
    multiplier = GLOBAL_MULTIPLIER
    offset = GLOBAL_OFFSET
    result = []
    append = result.append
    for num in numbers:
        append(num * multiplier + offset)
    return result

def math_operation_bad(numbers):
    result = []
    for num in numbers:
        result.append(math.sqrt(num) + math.sin(num) + len(numbers))
    return result

def math_operation_good(numbers):
    sqrt = math.sqrt
    sin = math.sin
    data_len = len(numbers)
    result = []
    append = result.append
    for num in numbers:
        append(sqrt(num) + sin(num) + data_len)
    return result

def measure_time(func, arg, iterations=10):
    times = []
    for _ in range(iterations):
        start = time.time()
        func(arg)
        times.append(time.time() - start)
    return sum(times) / len(times)

if __name__ == "__main__":
    numbers = list(range(1, 100_001))
    iterations = 10
    
    print("=== 测试 1: 循环内访问全局变量 vs 局部变量 ===")
    t_bad = measure_time(calculate_bad, numbers, iterations)
    t_good = measure_time(calculate_good, numbers, iterations)
    
    print(f"循环内访问全局变量耗时: {t_bad:.4f} 秒")
    print(f"使用局部变量耗时: {t_good:.4f} 秒")
    print(f"性能提升: {t_bad / t_good:.2f}x\n")
    
    print("=== 测试 2: 循环内调用全局函数 vs 局部绑定 ===")
    t_bad2 = measure_time(math_operation_bad, numbers, iterations)
    t_good2 = measure_time(math_operation_good, numbers, iterations)
    
    print(f"循环内调用全局函数耗时: {t_bad2:.4f} 秒")
    print(f"使用局部绑定耗时: {t_good2:.4f} 秒")
    print(f"性能提升: {t_bad2 / t_good2:.2f}x\n")
    
    print("=== 字节码对比: 全局变量访问 vs 局部变量访问 ===")
    print("反模式函数字节码:")
    dis.dis(calculate_bad)
    print("\n优化后函数字节码:")
    dis.dis(calculate_good)
    
    print("\n=== 结果验证 ===")
    r1 = calculate_bad([1, 2, 3, 4, 5])
    r2 = calculate_good([1, 2, 3, 4, 5])
    print(f"反模式结果: {r1}")
    print(f"优化结果: {r2}")
    print(f"结果一致: {r1 == r2}")
    
    print("\n=== 小技巧: 查看变量查找速度 ===")
    import timeit
    global_var_time = timeit.timeit(
        'x = GLOBAL_MULTIPLIER',
        setup='from __main__ import GLOBAL_MULTIPLIER',
        number=1_000_000
    )
    local_var_time = timeit.timeit(
        'x = local_var',
        setup='local_var = 2',
        number=1_000_000
    )
    print(f"100万次全局变量访问耗时: {global_var_time:.4f} 秒")
    print(f"100万次局部变量访问耗时: {local_var_time:.4f} 秒")
    print(f"局部变量快: {global_var_time / local_var_time:.2f}x")
