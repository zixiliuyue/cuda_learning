import threading
import time
from multiprocessing import Pool

def cpu_heavy_task(n):
    result = 0
    for i in range(n):
        result += i * i
    return result

def single_thread_version():
    for _ in range(4):
        cpu_heavy_task(10_000_000)

def threaded_version():
    threads = []
    for _ in range(4):
        t = threading.Thread(target=cpu_heavy_task, args=(10_000_000,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()

def multiprocess_version():
    with Pool(4) as pool:
        pool.map(cpu_heavy_task, [10_000_000] * 4)

if __name__ == "__main__":
    start = time.time()
    single_thread_version()
    print(f"单线程耗时: {time.time() - start:.2f} 秒")

    start = time.time()
    threaded_version()
    print(f"多线程耗时: {time.time() - start:.2f} 秒")

    start = time.time()
    multiprocess_version()
    print(f"多进程耗时: {time.time() - start:.2f} 秒")
