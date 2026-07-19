#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
线程安全和锁示例
演示：竞态条件、Lock、RLock、Semaphore、Condition 等同步机制
"""

import threading
import time
import random
from queue import Queue


def example_race_condition():
    """示例1：竞态条件演示（不加锁的情况）"""
    print(f"\n{'='*60}")
    print("示例1：竞态条件演示（不加锁）")
    print(f"{'='*60}")
    
    counter = 0
    iterations = 100000
    
    def increment():
        nonlocal counter
        for _ in range(iterations):
            counter += 1  # 这不是原子操作！
    
    threads = [threading.Thread(target=increment) for _ in range(4)]
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    expected = 4 * iterations
    print(f"期望结果: {expected}")
    print(f"实际结果: {counter}")
    print(f"差异: {expected - counter}")
    
    if counter != expected:
        print("✓ 发现竞态条件！多个线程同时修改共享变量导致数据不一致")
    else:
        print("✗ 意外：结果正确（可能是因为执行时间太短）")


def example_lock():
    """示例2：使用 Lock 解决竞态条件"""
    print(f"\n{'='*60}")
    print("示例2：使用 Lock 解决竞态条件")
    print(f"{'='*60}")
    
    counter = 0
    lock = threading.Lock()
    iterations = 100000
    
    def increment():
        nonlocal counter
        for _ in range(iterations):
            with lock:  # 使用 with 语句自动获取和释放锁
                counter += 1
    
    threads = [threading.Thread(target=increment) for _ in range(4)]
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    expected = 4 * iterations
    print(f"期望结果: {expected}")
    print(f"实际结果: {counter}")
    
    if counter == expected:
        print("✓ 结果正确！使用锁保证了线程安全")
    else:
        print("✗ 结果错误！")


def example_rlock():
    """示例3：RLock 可重入锁"""
    print(f"\n{'='*60}")
    print("示例3：RLock - 可重入锁（同一个线程可以多次获取）")
    print(f"{'='*60}")
    
    rlock = threading.RLock()
    
    def nested_function():
        """嵌套函数也需要锁"""
        with rlock:
            print("  嵌套函数: 获取并使用锁")
            time.sleep(0.1)
            print("  嵌套函数: 释放锁")
    
    def outer_function():
        """外层函数"""
        print("外层函数: 第一次获取锁")
        with rlock:
            print("外层函数: 持有锁，调用嵌套函数")
            nested_function()  # 同一个线程可以再次获取锁
            print("外层函数: 嵌套函数返回，继续持有锁")
        print("外层函数: 释放锁")
    
    thread = threading.Thread(target=outer_function)
    thread.start()
    thread.join()
    
    print("\n✓ RLock 允许同一个线程多次获取同一个锁")
    print("  这在递归函数或嵌套调用中很有用")


def example_semaphore():
    """示例4：Semaphore 信号量"""
    print(f"\n{'='*60}")
    print("示例4：Semaphore - 控制同时访问资源的线程数量")
    print(f"{'='*60}")
    
    max_concurrent = 2
    semaphore = threading.Semaphore(max_concurrent)
    active_threads = []
    
    def worker(worker_id):
        print(f"线程 {worker_id}: 等待获取信号量...")
        
        with semaphore:
            active_threads.append(worker_id)
            print(f"线程 {worker_id}: 获取到信号量！当前活动线程: {active_threads}")
            time.sleep(0.5)  # 模拟使用资源
            active_threads.remove(worker_id)
        
        print(f"线程 {worker_id}: 释放信号量")
    
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    print(f"\n✓ Semaphore 限制了最多 {max_concurrent} 个线程同时执行")
    print("  可用于限制数据库连接数、API 调用频率等")


def example_condition():
    """示例5：Condition 条件变量"""
    print(f"\n{'='*60}")
    print("示例5：Condition - 生产者消费者模型（线程间协作）")
    print(f"{'='*60}")
    
    condition = threading.Condition()
    buffer = []
    max_size = 3
    
    def producer(producer_id):
        for i in range(3):
            item = f"产品-{producer_id}-{i}"
            
            with condition:
                # 等待缓冲区有空间
                while len(buffer) >= max_size:
                    print(f"生产者 {producer_id}: 缓冲区满，等待...")
                    condition.wait()
                
                # 生产产品
                buffer.append(item)
                print(f"生产者 {producer_id}: 生产了 {item}，缓冲区: {buffer}")
                
                # 通知消费者
                condition.notify_all()
            
            time.sleep(random.uniform(0.1, 0.3))
    
    def consumer(consumer_id):
        consumed = 0
        while consumed < 3:
            with condition:
                # 等待缓冲区有数据
                while len(buffer) == 0:
                    print(f"消费者 {consumer_id}: 缓冲区空，等待...")
                    condition.wait()
                
                # 消费产品
                item = buffer.pop(0)
                consumed += 1
                print(f"消费者 {consumer_id}: 消费了 {item}，缓冲区: {buffer}")
                
                # 通知生产者
                condition.notify_all()
            
            time.sleep(random.uniform(0.2, 0.4))
    
    producers = [threading.Thread(target=producer, args=(i,)) for i in range(2)]
    consumers = [threading.Thread(target=consumer, args=(i,)) for i in range(2)]
    
    for p in producers:
        p.start()
    for c in consumers:
        c.start()
    
    for p in producers:
        p.join()
    for c in consumers:
        c.join()
    
    print("\n✓ Condition 实现了线程间的协作")
    print("  生产者和消费者可以互相通知对方状态变化")


def example_event():
    """示例6：Event 事件"""
    print(f"\n{'='*60}")
    print("示例6：Event - 事件通知机制")
    print(f"{'='*60}")
    
    event = threading.Event()
    
    def waiter(name):
        print(f"{name}: 等待事件触发...")
        event.wait()  # 等待事件被设置
        print(f"{name}: 收到事件通知！继续执行")
    
    def setter():
        print("设置者: 准备触发事件...")
        time.sleep(1)
        print("设置者: 触发事件！")
        event.set()  # 设置事件，唤醒所有等待的线程
    
    waiters = [threading.Thread(target=waiter, args=(f"等待者{i}",)) for i in range(3)]
    setter_thread = threading.Thread(target=setter)
    
    for w in waiters:
        w.start()
    setter_thread.start()
    
    for w in waiters:
        w.join()
    setter_thread.join()
    
    print("\n✓ Event 可用于一次性通知多个线程")
    print("  常用于实现'开始'、'停止'等控制信号")


def compare_locks():
    """对比各种锁机制"""
    print(f"\n{'='*60}")
    print("各种同步机制对比总结")
    print(f"{'='*60}")
    
    comparison = """
| 机制        | 用途                                  | 特点                                        |
|-------------|---------------------------------------|---------------------------------------------|
| Lock        | 简单的互斥锁                          | 同一时间只允许一个线程访问，不可重入        |
| RLock       | 可重入互斥锁                          | 同一个线程可以多次获取，适合递归/嵌套场景   |
| Semaphore   | 控制并发数量                          | 允许多个线程同时访问，可限制资源使用数      |
| Condition   | 线程间协作                            | 等待/通知机制，实现生产者消费者模型         |
| Event       | 简单事件通知                          | 一次性通知多个线程，类似开关                |

选择建议：
1. 简单的互斥访问 -> 使用 Lock
2. 递归/嵌套调用需要锁 -> 使用 RLock
3. 限制并发数量（如连接池）-> 使用 Semaphore
4. 线程间需要协作（如生产者消费者）-> 使用 Condition
5. 简单的事件通知（如开始/停止）-> 使用 Event

最佳实践：
- 始终使用 with 语句管理锁，避免死锁
- 锁定范围尽量小，只保护真正需要的代码
- 避免嵌套获取多个锁，或按固定顺序获取
- 不要在持有锁的情况下执行耗时的 I/O 操作
    """
    
    print(comparison)


def main():
    print("线程安全和锁示例程序")
    print("本程序演示各种线程同步机制\n")
    
    example_race_condition()
    example_lock()
    example_rlock()
    example_semaphore()
    example_condition()
    example_event()
    compare_locks()
    
    print("\n所有示例完成！")


if __name__ == "__main__":
    main()
