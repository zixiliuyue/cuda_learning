import threading
import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

def get_memory_usage():
    """获取当前进程内存使用（MB）"""
    import psutil
    return psutil.Process().memory_info().rss / 1024 / 1024

def cpu_heavy_task(x):
    """模拟 CPU 密集型任务"""
    result = 0
    for i in range(10000):
        result += i * x
    return result

def simulate_unbounded_threads():
    """模拟无限制创建线程"""
    print("=== 无限制创建线程 (演示前 500 个) ===")
    
    threads = []
    start_time = time.time()
    
    for i in range(1, 501):
        t = threading.Thread(target=cpu_heavy_task, args=(i,))
        threads.append(t)
        t.start()
        
        if i % 100 == 0:
            elapsed = time.time() - start_time
            print(f"创建 {i} 个线程耗时: {elapsed:.2f} 秒, "
                  f"当前活跃线程数: {threading.active_count()}")
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    print(f"总耗时: {time.time() - start_time:.2f} 秒")
    return len(threads)

def simulate_thread_pool():
    """使用线程池"""
    print("\n=== 使用线程池 (10 个工作线程) ===")
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for i in range(1, 501):
            futures.append(executor.submit(cpu_heavy_task, i))
            
            if i % 100 == 0:
                elapsed = time.time() - start_time
                print(f"提交 {i} 个任务耗时: {elapsed:.2f} 秒, "
                      f"当前活跃线程数: {threading.active_count()}")
        
        # 等待所有任务完成
        results = [f.result() for f in as_completed(futures)]
    
    print(f"总耗时: {time.time() - start_time:.2f} 秒")
    print(f"完成任务数: {len(results)}")
    return len(results)

class BoundedThreadPool:
    """带信号量限制的线程池"""
    def __init__(self, max_workers=10, max_queue=50):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = threading.Semaphore(max_queue)
        self.max_workers = max_workers
    
    def submit(self, func, *args, **kwargs):
        if not self.semaphore.acquire(blocking=False):
            raise queue.Full("任务队列已满，请稍后重试")
        
        try:
            future = self.executor.submit(func, *args, **kwargs)
            future.add_done_callback(lambda _: self.semaphore.release())
            return future
        except Exception:
            self.semaphore.release()
            raise

def simulate_bounded_thread_pool():
    """模拟带边界的线程池"""
    print("\n=== 带边界的线程池 (最大队列 50) ===")
    
    pool = BoundedThreadPool(max_workers=10, max_queue=50)
    
    success_count = 0
    reject_count = 0
    
    start_time = time.time()
    
    for i in range(100):
        try:
            pool.submit(cpu_heavy_task, i)
            success_count += 1
        except queue.Full:
            reject_count += 1
        
        if i % 20 == 0:
            print(f"已提交 {i+1} 个任务, 成功: {success_count}, "
                  f"拒绝: {reject_count}, 活跃线程: {threading.active_count()}")
        time.sleep(0.01)
    
    print(f"总耗时: {time.time() - start_time:.2f} 秒")
    print(f"最终: 成功 {success_count}, 拒绝 {reject_count}")

def simulate_worker_pool():
    """模拟工作线程 + 任务队列模式"""
    print("\n=== 工作线程 + 任务队列模式 ===")
    
    work_queue = queue.Queue(maxsize=30)
    results = []
    results_lock = threading.Lock()
    
    def worker():
        while True:
            try:
                task = work_queue.get(timeout=1)
                if task is None:
                    break
                
                result = cpu_heavy_task(task)
                with results_lock:
                    results.append(result)
                work_queue.task_done()
            except queue.Empty:
                continue
    
    # 启动 5 个工作线程
    workers = []
    for _ in range(5):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        workers.append(t)
    
    # 提交任务
    start_time = time.time()
    for i in range(100):
        try:
            work_queue.put_nowait(i)
        except queue.Full:
            print(f"队列已满，拒绝任务 {i}")
        
        if i % 20 == 0:
            print(f"已提交 {i+1} 个任务, 队列大小: {work_queue.qsize()}, "
                  f"已完成: {len(results)}")
    
    # 等待所有任务完成
    work_queue.join()
    total_time = time.time() - start_time
    
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"完成任务数: {len(results)}")
    
    # 停止工作线程
    for _ in workers:
        work_queue.put(None)
    for t in workers:
        t.join()

def compare_performance():
    """对比性能"""
    print("\n=== 性能对比 ===")
    
    # 无限制线程（演示 200 个）
    start = time.time()
    threads = []
    for i in range(200):
        t = threading.Thread(target=cpu_heavy_task, args=(i,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    unbounded_time = time.time() - start
    
    # 线程池
    start = time.time()
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(cpu_heavy_task, range(200)))
    pool_time = time.time() - start
    
    print(f"无限制线程 (200个) 耗时: {unbounded_time:.2f} 秒")
    print(f"线程池 (10线程) 耗时: {pool_time:.2f} 秒")
    print(f"线程池快: {unbounded_time / pool_time:.2f}x")

if __name__ == "__main__":
    print(f"系统 CPU 核心数: {os.cpu_count()}")
    print(f"初始活跃线程数: {threading.active_count()}")
    print(f"初始内存使用: {get_memory_usage():.2f} MB\n")
    
    # 演示无限制创建线程
    unbounded_count = simulate_unbounded_threads()
    
    # 演示线程池
    pool_count = simulate_thread_pool()
    
    # 演示带边界的线程池
    simulate_bounded_thread_pool()
    
    # 演示工作线程模式
    simulate_worker_pool()
    
    # 性能对比
    compare_performance()
    
    print("\n=== 总结 ===")
    print("❌ 无限制创建线程: 资源耗尽，系统崩溃")
    print("✅ 线程池: 资源可控，性能稳定")
    print("✅ 信号量: 动态限制并发数")
    print("✅ 工作线程+队列: 优雅处理过载")
