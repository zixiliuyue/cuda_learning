import pandas as pd
import numpy as np
import time
import os
from typing import Dict, Tuple


class PandasOptimizationBenchmark:
    def __init__(self, num_rows: int = 1_000_000):
        self.num_rows = num_rows
        self.results = {}
        self.data_path = 'test_data_optimization.csv'

    def generate_large_dataset(self) -> pd.DataFrame:
        """生成100万行测试数据"""
        np.random.seed(42)
        
        categories = ['A', 'B', 'C', 'D', 'E']
        
        data = {
            'id': range(self.num_rows),
            'age': np.random.randint(18, 80, self.num_rows),
            'salary': np.random.randint(3000, 50000, self.num_rows),
            'score': np.random.uniform(0, 100, self.num_rows),
            'category': np.random.choice(categories, self.num_rows),
            'flag': np.random.choice([True, False], self.num_rows),
            'value1': np.random.randint(1, 100, self.num_rows),
            'value2': np.random.randint(1, 100, self.num_rows)
        }
        
        df = pd.DataFrame(data)
        return df

    def save_test_data(self, df: pd.DataFrame):
        """保存测试数据到CSV"""
        df.to_csv(self.data_path, index=False)
        print(f"数据已保存: {self.data_path}, 文件大小: {os.path.getsize(self.data_path) / 1024 / 1024:.2f} MB")

    def load_test_data(self) -> pd.DataFrame:
        """加载测试数据"""
        return pd.read_csv(self.data_path)

    def get_memory_usage(self, df: pd.DataFrame) -> float:
        """获取DataFrame内存使用量(MB)"""
        return df.memory_usage(deep=True).sum() / 1024 / 1024

    def benchmark_dtype_optimization(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """1) 数据类型优化benchmark"""
        print("\n" + "="*60)
        print("1. 数据类型优化 (astype)")
        print("="*60)
        
        df_original = df.copy()
        memory_before = self.get_memory_usage(df_original)
        print(f"优化前内存使用: {memory_before:.2f} MB")
        
        start_time = time.time()
        df_optimized = df_original.copy()
        
        df_optimized['age'] = df_optimized['age'].astype('int8')
        df_optimized['salary'] = df_optimized['salary'].astype('int32')
        df_optimized['score'] = df_optimized['score'].astype('float32')
        df_optimized['flag'] = df_optimized['flag'].astype('bool')
        df_optimized['value1'] = df_optimized['value1'].astype('int8')
        df_optimized['value2'] = df_optimized['value2'].astype('int8')
        
        conversion_time = time.time() - start_time
        memory_after = self.get_memory_usage(df_optimized)
        memory_saved = memory_before - memory_after
        
        print(f"优化后内存使用: {memory_after:.2f} MB")
        print(f"节省内存: {memory_saved:.2f} MB ({(memory_saved/memory_before*100):.1f}%)")
        print(f"转换耗时: {conversion_time:.4f}秒")
        
        return memory_before, memory_after, conversion_time

    def benchmark_iteration_methods(self, df: pd.DataFrame) -> Dict[str, float]:
        """2) 向量化操作 vs apply vs iterrows benchmark"""
        print("\n" + "="*60)
        print("2. 迭代方法对比 (向量化 vs apply vs iterrows)")
        print("="*60)
        
        df_test = df.copy()
        results = {}
        
        start = time.time()
        result_vectorized = df_test['salary'] * 1.1 + df_test['score'] * 0.5
        results['向量化操作'] = time.time() - start
        print(f"向量化操作: {results['向量化操作']:.4f}秒")
        
        start = time.time()
        result_apply = df_test.apply(lambda row: row['salary'] * 1.1 + row['score'] * 0.5, axis=1)
        results['apply方法'] = time.time() - start
        print(f"apply方法: {results['apply方法']:.4f}秒 (慢{results['apply方法']/results['向量化操作']:.0f}倍)")
        
        start = time.time()
        result_iterrows = []
        for _, row in df_test.iterrows():
            result_iterrows.append(row['salary'] * 1.1 + row['score'] * 0.5)
        results['iterrows方法'] = time.time() - start
        print(f"iterrows方法: {results['iterrows方法']:.4f}秒 (慢{results['iterrows方法']/results['向量化操作']:.0f}倍)")
        
        return results

    def benchmark_builtin_functions(self, df: pd.DataFrame) -> Dict[str, float]:
        """3) 内置函数优先 benchmark"""
        print("\n" + "="*60)
        print("3. 内置函数 vs 自定义函数")
        print("="*60)
        
        df_test = df.copy()
        results = {}
        
        start = time.time()
        mean1 = df_test['salary'].mean()
        std1 = df_test['salary'].std()
        max1 = df_test['salary'].max()
        min1 = df_test['salary'].min()
        results['内置聚合函数'] = time.time() - start
        print(f"内置聚合函数 (mean/std/max/min): {results['内置聚合函数']:.4f}秒")
        
        start = time.time()
        mean2 = df_test['salary'].apply(lambda x: x).mean()
        std2 = df_test['salary'].apply(lambda x: x).std()
        results['apply聚合'] = time.time() - start
        print(f"apply聚合: {results['apply聚合']:.4f}秒 (慢{results['apply聚合']/results['内置聚合函数']:.0f}倍)")
        
        return results

    def benchmark_query_method(self, df: pd.DataFrame) -> Dict[str, float]:
        """4) query方法优化 benchmark"""
        print("\n" + "="*60)
        print("4. query方法 vs 布尔索引")
        print("="*60)
        
        df_test = df.copy()
        results = {}
        
        start = time.time()
        result_bool = df_test[(df_test['age'] > 30) & (df_test['salary'] > 10000) & (df_test['score'] > 50)]
        results['布尔索引'] = time.time() - start
        print(f"布尔索引: {results['布尔索引']:.4f}秒, 结果行数: {len(result_bool)}")
        
        start = time.time()
        result_query = df_test.query('age > 30 and salary > 10000 and score > 50')
        results['query方法'] = time.time() - start
        print(f"query方法: {results['query方法']:.4f}秒, 结果行数: {len(result_query)}")
        
        speedup = results['布尔索引'] / results['query方法']
        print(f"query加速比: {speedup:.2f}x")
        
        return results

    def benchmark_eval_expression(self, df: pd.DataFrame) -> Dict[str, float]:
        """5) eval表达式优化 benchmark"""
        print("\n" + "="*60)
        print("5. eval表达式 vs 普通计算")
        print("="*60)
        
        df_test = df.copy()
        results = {}
        
        start = time.time()
        result_normal = df_test['salary'] * 1.5 + df_test['score'] * 0.3 - df_test['value1'] * 0.1
        results['普通计算'] = time.time() - start
        print(f"普通计算: {results['普通计算']:.4f}秒")
        
        start = time.time()
        result_eval = df_test.eval('salary * 1.5 + score * 0.3 - value1 * 0.1')
        results['eval表达式'] = time.time() - start
        print(f"eval表达式: {results['eval表达式']:.4f}秒")
        
        speedup = results['普通计算'] / results['eval表达式']
        print(f"eval加速比: {speedup:.2f}x")
        
        return results

    def benchmark_chunk_reading(self) -> Dict[str, float]:
        """6) 分块读取大文件 benchmark"""
        print("\n" + "="*60)
        print("6. 分块读取 vs 一次性读取")
        print("="*60)
        
        results = {}
        
        start = time.time()
        df_full = pd.read_csv(self.data_path)
        results['一次性读取'] = time.time() - start
        print(f"一次性读取: {results['一次性读取']:.4f}秒, 内存: {self.get_memory_usage(df_full):.2f} MB")
        
        start = time.time()
        chunk_size = 100_000
        total_rows = 0
        for chunk in pd.read_csv(self.data_path, chunksize=chunk_size):
            total_rows += len(chunk)
        results['分块读取'] = time.time() - start
        print(f"分块读取 (chunksize={chunk_size}): {results['分块读取']:.4f}秒, 总行数: {total_rows}")
        
        return results

    def benchmark_categorical_dtype(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """7) 分类数据类型优化 benchmark"""
        print("\n" + "="*60)
        print("7. category数据类型优化")
        print("="*60)
        
        df_test = df.copy()
        memory_before = self.get_memory_usage(df_test[['category']])
        print(f"优化前category列内存: {memory_before:.4f} MB")
        print(f"优化前category列dtype: {df_test['category'].dtype}")
        
        start = time.time()
        df_test['category'] = df_test['category'].astype('category')
        conversion_time = time.time() - start
        memory_after = self.get_memory_usage(df_test[['category']])
        memory_saved = memory_before - memory_after
        
        print(f"优化后category列内存: {memory_after:.4f} MB")
        print(f"优化后category列dtype: {df_test['category'].dtype}")
        print(f"节省内存: {memory_saved:.4f} MB ({(memory_saved/memory_before*100):.1f}%)")
        print(f"转换耗时: {conversion_time:.4f}秒")
        
        start = time.time()
        groupby_result = df_test.groupby('category')['salary'].mean()
        results_groupby = time.time() - start
        print(f"category分组耗时: {results_groupby:.4f}秒")
        
        return memory_before, memory_after, results_groupby

    def benchmark_full_memory_optimization(self, df: pd.DataFrame) -> Tuple[float, float]:
        """8) 完整内存优化前后对比 benchmark"""
        print("\n" + "="*60)
        print("8. 完整内存优化前后对比")
        print("="*60)
        
        df_before = df.copy()
        memory_before = self.get_memory_usage(df_before)
        print(f"优化前总内存: {memory_before:.2f} MB")
        
        start = time.time()
        df_after = df_before.copy()
        
        df_after['age'] = df_after['age'].astype('int8')
        df_after['salary'] = df_after['salary'].astype('int32')
        df_after['score'] = df_after['score'].astype('float32')
        df_after['flag'] = df_after['flag'].astype('bool')
        df_after['category'] = df_after['category'].astype('category')
        df_after['value1'] = df_after['value1'].astype('int8')
        df_after['value2'] = df_after['value2'].astype('int8')
        
        conversion_time = time.time() - start
        memory_after = self.get_memory_usage(df_after)
        memory_saved = memory_before - memory_after
        
        print(f"优化后总内存: {memory_after:.2f} MB")
        print(f"节省内存: {memory_saved:.2f} MB ({(memory_saved/memory_before*100):.1f}%)")
        print(f"优化耗时: {conversion_time:.4f}秒")
        
        return memory_before, memory_after

    def run_all_benchmarks(self):
        """运行所有benchmark测试"""
        print("="*60)
        print(f"Pandas性能优化 Benchmark (数据量: {self.num_rows:,}行)")
        print("="*60)
        
        print("\n生成测试数据...")
        df = self.generate_large_dataset()
        print(f"测试数据生成完成, 行数: {len(df):,}, 列数: {len(df.columns)}")
        
        self.save_test_data(df)
        
        self.benchmark_dtype_optimization(df)
        self.benchmark_iteration_methods(df)
        self.benchmark_builtin_functions(df)
        self.benchmark_query_method(df)
        self.benchmark_eval_expression(df)
        self.benchmark_chunk_reading()
        self.benchmark_categorical_dtype(df)
        self.benchmark_full_memory_optimization(df)
        
        self.print_summary()

    def print_summary(self):
        """打印总结"""
        print("\n" + "="*60)
        print("优化总结")
        print("="*60)
        print("""
1. 数据类型优化:
   - 使用astype转换为更小的数据类型可以大幅减少内存使用
   - int8代替int64可节省87.5%内存
   - float32代替float64可节省50%内存

2. 迭代方法对比:
   - 向量化操作 >> apply >> iterrows (性能差异可达千倍)
   - 永远避免在循环中处理DataFrame

3. 内置函数优先:
   - Pandas内置函数经过高度优化
   - 避免使用apply进行简单聚合

4. query方法:
   - query方法比布尔索引更简洁高效
   - 复杂查询条件下性能优势明显

5. eval表达式:
   - 复杂表达式计算时性能优于普通计算
   - 基于numexpr引擎，支持多核并行

6. 分块读取:
   - 处理超大型文件时避免内存溢出
   - 逐块处理，内存占用低

7. category数据类型:
   - 低基数字符串列转换为category类型
   - 分组、排序操作性能显著提升

8. 完整优化:
   - 综合应用所有优化技巧
   - 内存节省可达50-80%
        """)


if __name__ == '__main__':
    benchmark = PandasOptimizationBenchmark(num_rows=1_000_000)
    benchmark.run_all_benchmarks()
