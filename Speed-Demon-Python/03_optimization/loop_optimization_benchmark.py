import timeit
import itertools
import math
import sys

def benchmark(name, stmt, setup="pass", number=1000, repeat=3):
    """基准测试函数，返回平均执行时间（微秒）"""
    times = timeit.repeat(stmt=stmt, setup=setup, number=number, repeat=repeat)
    avg_time = min(times) / number * 1_000_000
    print(f"{name:40} {avg_time:8.2f} μs")
    return avg_time

def print_section(title):
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

def run_benchmarks():
    print(f"Python 循环优化基准测试")
    print(f"Python 版本: {sys.version.split()[0]}")
    print(f"测试配置: {1000} 次迭代，取 {3} 次中最快值")
    
    # 1. 列表推导 vs for 循环
    print_section("1. 列表推导 vs for 循环")
    t1 = benchmark(
        "for 循环 + append",
        """
result = []
for i in range(1000):
    result.append(i * 2)
        """
    )
    t2 = benchmark("列表推导", "result = [i * 2 for i in range(1000)]")
    print(f"提升: {(t1/t2):.2f}x ({(1-t2/t1)*100:.1f}%)")

    # 2. map/filter vs 循环
    print_section("2. map/filter vs 循环")
    t1 = benchmark(
        "for 循环过滤+转换",
        """
result = []
for i in range(1000):
    if i % 2 == 0:
        result.append(i * 3)
        """
    )
    t2 = benchmark("filter + map", "list(map(lambda x: x*3, filter(lambda x: x%2==0, range(1000))))")
    t3 = benchmark("列表推导 + 条件", "[i * 3 for i in range(1000) if i % 2 == 0]")
    print(f"map/filter 提升: {(t1/t2):.2f}x")
    print(f"列表推导 提升: {(t1/t3):.2f}x")

    # 3. 内置函数优先
    print_section("3. 内置函数优先")
    setup = "data = list(range(1000))"
    t1 = benchmark("手动循环求和", "total=0\nfor n in data:\n    total+=n", setup=setup)
    t2 = benchmark("内置 sum()", "sum(data)", setup=setup)
    t3 = benchmark("手动循环求最大值", "m=0\nfor n in data:\n    if n>m: m=n", setup=setup)
    t4 = benchmark("内置 max()", "max(data)", setup=setup)
    print(f"sum() 提升: {(t1/t2):.2f}x")
    print(f"max() 提升: {(t3/t4):.2f}x")

    # 4. 避免全局变量查找
    print_section("4. 避免全局变量查找")
    setup = "import math; data = list(range(1000))"
    t1 = benchmark("使用全局 math.sin", "r=[]\nfor x in data:\n    r.append(math.sin(x))", setup=setup)
    t2 = benchmark("使用局部 sin 引用", "sin=math.sin\nr=[]\nfor x in data:\n    r.append(sin(x))", setup=setup)
    print(f"局部化函数提升: {(t1/t2):.2f}x")

    # 5. 使用局部变量
    print_section("5. 使用局部变量（属性查找优化）")
    t1 = benchmark("循环内 append", "r=[]\nfor i in range(1000):\n    r.append(i)")
    t2 = benchmark("局部 append 引用", "r=[]\na=r.append\nfor i in range(1000):\n    a(i)")
    print(f"局部方法引用提升: {(t1/t2):.2f}x")

    # 6. 循环展开
    print_section("6. 循环展开")
    t1 = benchmark("普通循环", "t=0\nfor i in range(1000):\n    t+=i")
    t2 = benchmark("展开 4 次", "t=0\nfor i in range(0,1000,4):\n    t+=i; t+=i+1; t+=i+2; t+=i+3")
    print(f"循环展开提升: {(t1/t2):.2f}x")

    # 7. 生成器表达式
    print_section("7. 生成器表达式（内存测试）")
    import sys
    list_mem = sys.getsizeof([i * 2 for i in range(10000)])
    gen_mem = sys.getsizeof((i * 2 for i in range(10000)))
    print(f"列表推导内存: {list_mem} bytes")
    print(f"生成器内存:   {gen_mem} bytes")
    print(f"内存节省: {(1-gen_mem/list_mem)*100:.1f}%")

    # 8. itertools
    print_section("8. itertools 高效迭代")
    setup = "import itertools"
    t1 = benchmark("嵌套循环笛卡尔积", "r=[]\nfor i in range(50):\n    for j in range(50):\n        r.append((i,j))")
    t2 = benchmark("itertools.product", "list(itertools.product(range(50), range(50)))", setup=setup)
    print(f"itertools.product 提升: {(t1/t2):.2f}x")

    print("\n" + "="*60)
    print("基准测试完成！")
    print("="*60)

if __name__ == "__main__":
    run_benchmarks()
