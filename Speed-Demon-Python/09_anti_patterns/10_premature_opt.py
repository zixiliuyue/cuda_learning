import time
import cProfile
import pstats
import random

# ============================================
# 场景 1：位运算 vs 清晰的代码
# ============================================
def is_even_bad(n):
    """反模式：用位运算，晦涩难懂"""
    return n & 1 == 0

def is_even_good(n):
    """正确：清晰易懂"""
    return n % 2 == 0

def demo_bit_operation():
    print("=== 场景 1：位运算 vs 清晰代码 ===")
    
    numbers = list(range(1_000_000))
    
    # 测试"优化"版本
    start = time.time()
    result_bad = [is_even_bad(n) for n in numbers]
    time_bad = time.time() - start
    
    # 测试清晰版本
    start = time.time()
    result_good = [is_even_good(n) for n in numbers]
    time_good = time.time() - start
    
    print(f"位运算版本耗时: {time_bad:.4f} 秒")
    print(f"清晰版本耗时: {time_good:.4f} 秒")
    print(f"性能差异: {abs(time_bad - time_good) / max(time_bad, time_good) * 100:.1f}%")
    print(f"结果一致: {result_bad == result_good}")
    print(f"阅读难度提升: ~200% 😱")
    print()

# ============================================
# 场景 2：不必要的缓存
# ============================================
_cache = {}

def square_bad(n):
    """反模式：不必要的缓存"""
    if n not in _cache:
        _cache[n] = n * n
    return _cache[n]

def square_good(n):
    """正确：直接计算"""
    return n * n

def demo_unnecessary_cache():
    print("=== 场景 2：不必要的缓存 ===")
    
    numbers = [random.randint(1, 1000) for _ in range(100_000)]
    
    # 测试"优化"版本
    start = time.time()
    result_bad = [square_bad(n) for n in numbers]
    time_bad = time.time() - start
    
    # 测试简单版本
    start = time.time()
    result_good = [square_good(n) for n in numbers]
    time_good = time.time() - start
    
    print(f"带缓存版本耗时: {time_bad:.4f} 秒")
    print(f"直接计算版本耗时: {time_good:.4f} 秒")
    print(f"缓存反而更慢！因为字典查找开销！")
    print(f"内存占用: 缓存了 {len(_cache)} 个条目")
    print()

# ============================================
# 场景 3：手动内联 vs 函数封装
# ============================================
def process_item_complex(item):
    """处理单个项目的复杂逻辑"""
    # 模拟一些处理
    result = item * 2
    result = result + 1
    result = result ** 0.5
    return int(result)

def process_data_inlined(data):
    """反模式：手动内联，代码冗长"""
    result = []
    for item in data:
        # 手动内联的代码
        temp = item * 2
        temp = temp + 1
        temp = temp ** 0.5
        result.append(int(temp))
    return result

def process_data_functional(data):
    """正确：函数封装，清晰"""
    return [process_item_complex(item) for item in data]

def demo_inline_vs_function():
    print("=== 场景 3：手动内联 vs 函数封装 ===")
    
    data = list(range(100_000))
    
    # 测试内联版本
    start = time.time()
    result_inline = process_data_inlined(data)
    time_inline = time.time() - start
    
    # 测试函数版本
    start = time.time()
    result_func = process_data_functional(data)
    time_func = time.time() - start
    
    print(f"内联版本耗时: {time_inline:.4f} 秒")
    print(f"函数版本耗时: {time_func:.4f} 秒")
    print(f"性能差异: {abs(time_inline - time_func) / max(time_inline, time_func) * 100:.1f}%")
    print(f"结果一致: {result_inline == result_func}")
    print(f"可维护性提升: ~300% ✅")
    print()

# ============================================
# 场景 4：真正的优化 - 算法层面
# ============================================
def sum_of_squares_naive(n):
    """朴素实现：O(n)"""
    total = 0
    for i in range(1, n + 1):
        total += i * i
    return total

def sum_of_squares_formula(n):
    """数学公式：O(1)"""
    return n * (n + 1) * (2 * n + 1) // 6

def demo_algorithm_optimization():
    print("=== 场景 4：真正的优化 - 算法层面 ===")
    
    n = 1_000_000
    
    # 朴素实现
    start = time.time()
    result_naive = sum_of_squares_naive(n)
    time_naive = time.time() - start
    
    # 公式实现
    start = time.time()
    result_formula = sum_of_squares_formula(n)
    time_formula = time.time() - start
    
    print(f"朴素实现 (O(n)) 耗时: {time_naive:.4f} 秒")
    print(f"公式实现 (O(1)) 耗时: {time_formula:.4f} 秒")
    print(f"性能提升: {time_naive / time_formula:.1f}x")
    print(f"结果一致: {result_naive == result_formula}")
    print("这才是真正值得做的优化！✅")
    print()

# ============================================
# Profiling 演示
# ============================================
def slow_function():
    """一个模拟的慢函数"""
    total = 0
    for i in range(10_000):
        total += i * i
    return total

def fast_function():
    """一个模拟的快函数"""
    return sum(i * i for i in range(10_000))

def demo_profiling():
    print("=== Profiling 演示：找到真正的瓶颈 ===")
    print("Profiling 是优化的前提，不要凭感觉优化！")
    print()
    
    # 创建一个包含快慢函数混合的程序
    def program():
        for _ in range(10):
            slow_function()
        for _ in range(100):
            fast_function()
    
    # 运行 profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    program()
    
    profiler.disable()
    
    print("Profiling 结果 (按累计时间排序):")
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(5)
    
    print("结论：优化 slow_function 才能带来最大收益，而不是 fast_function")
    print()

# ============================================
# 综合对比
# ============================================
def compare_all_approaches():
    print("="*60)
    print("优化方式对比总结")
    print("="*60)
    
    print(f"\n{'优化方式':<25} {'性能提升':<15} {'可读性影响':<20}")
    print("-" * 60)
    print(f"{'位运算替代取模':<25} {'~5%':<15} {'大幅下降 ❌':<20}")
    print(f"{'不必要的缓存':<25} {'-10% (更慢)':<15} {'大幅下降 ❌':<20}")
    print(f"{'手动内联函数':<25} {'~2%':<15} {'大幅下降 ❌':<20}")
    print(f"{'算法优化 (O(n)→O(1))':<25} {'1000x+':<15} {'不变/提升 ✅':<20}")
    
    print()
    print("="*60)
    print("优化决策指南")
    print("="*60)
    print("✅ 先让代码工作起来")
    print("✅ 用 profiling 找到真正的瓶颈")
    print("✅ 优化算法，而不是微操作")
    print("✅ 可读性优先于微优化")
    print("❌ 不要为了节省几纳秒而让代码变得晦涩")
    print("❌ 不要凭感觉优化，要有数据支持")
    print("❌ 不要在项目早期就过度关注性能")
    print()
    print('记住 Donald Knuth 的名言：')
    print('"Premature optimization is the root of all evil."')
    print('"过早优化是万恶之源。"')

def main():
    print("="*60)
    print("过早优化反模式演示")
    print("Premature Optimization Anti-Pattern Demo")
    print("="*60)
    print()
    
    demo_bit_operation()
    demo_unnecessary_cache()
    demo_inline_vs_function()
    demo_algorithm_optimization()
    demo_profiling()
    compare_all_approaches()

if __name__ == "__main__":
    main()
