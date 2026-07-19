#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
进程间通信（IPC）示例
演示：Queue、Pipe、Manager 等多种进程间通信方式
"""

import time
import random
from multiprocessing import Process, Queue, Pipe, Manager, Value, Array, Lock


def example_queue():
    """示例1：使用 Queue 进行进程间通信"""
    print(f"\n{'='*60}")
    print("示例1：Queue - 多生产者多消费者模型")
    print(f"{'='*60}")
    
    def producer(q, producer_id, num_items):
        """生产者进程"""
        for i in range(num_items):
            item = f"产品-{producer_id}-{i}"
            q.put(item)
            print(f"生产者 {producer_id}: 生产了 {item}")
            time.sleep(random.uniform(0.1, 0.3))
        q.put(None)  # 发送结束信号
    
    def consumer(q, consumer_id):
        """消费者进程"""
        count = 0
        while True:
            item = q.get()
            if item is None:
                q.put(None)  # 把结束信号放回去，让其他消费者也能收到
                break
            count += 1
            print(f"消费者 {consumer_id}: 消费了 {item} (累计: {count})")
            time.sleep(random.uniform(0.2, 0.4))
    
    q = Queue(maxsize=10)
    
    # 创建 2 个生产者，3 个消费者
    producers = [Process(target=producer, args=(q, i, 5)) for i in range(2)]
    consumers = [Process(target=consumer, args=(q, i)) for i in range(3)]
    
    for p in producers:
        p.start()
    for c in consumers:
        c.start()
    
    for p in producers:
        p.join()
    for c in consumers:
        c.join()
    
    print("Queue 示例完成！")


def example_pipe():
    """示例2：使用 Pipe 进行进程间通信"""
    print(f"\n{'='*60}")
    print("示例2：Pipe - 双向通信管道")
    print(f"{'='*60}")
    
    def parent_process(conn):
        """父进程：发送命令，接收结果"""
        commands = ["计算 1+1", "计算 2+3", "计算 5+8", "退出"]
        
        for cmd in commands:
            print(f"父进程: 发送命令 -> {cmd}")
            conn.send(cmd)
            
            if cmd == "退出":
                break
            
            result = conn.recv()
            print(f"父进程: 收到结果 -> {result}")
        
        conn.close()
    
    def child_process(conn):
        """子进程：接收命令，执行并返回结果"""
        while True:
            cmd = conn.recv()
            print(f"子进程: 收到命令 -> {cmd}")
            
            if cmd == "退出":
                print("子进程: 退出")
                break
            
            if cmd.startswith("计算 "):
                expr = cmd[3:]
                try:
                    result = eval(expr)
                    conn.send(f"{expr} = {result}")
                except Exception as e:
                    conn.send(f"计算错误: {e}")
        
        conn.close()
    
    parent_conn, child_conn = Pipe()
    
    p_parent = Process(target=parent_process, args=(parent_conn,))
    p_child = Process(target=child_process, args=(child_conn,))
    
    p_parent.start()
    p_child.start()
    
    p_parent.join()
    p_child.join()
    
    print("Pipe 示例完成！")


def example_manager():
    """示例3：使用 Manager 共享数据"""
    print(f"\n{'='*60}")
    print("示例3：Manager - 共享复杂数据结构 (dict, list)")
    print(f"{'='*60}")
    
    def worker(shared_dict, shared_list, worker_id, lock):
        """工作进程：修改共享数据"""
        with lock:  # 使用锁保证操作原子性
            shared_dict[f'worker_{worker_id}'] = f'完成了任务'
            shared_list.append(worker_id)
            print(f"工作进程 {worker_id}: dict={dict(shared_dict)}, list={list(shared_list)}")
    
    with Manager() as manager:
        shared_dict = manager.dict()
        shared_list = manager.list()
        lock = manager.Lock()
        
        processes = [Process(target=worker, args=(shared_dict, shared_list, i, lock)) for i in range(5)]
        
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        
        print(f"\n最终结果:")
        print(f"共享字典: {dict(shared_dict)}")
        print(f"共享列表: {list(shared_list)}")
    
    print("Manager 示例完成！")


def example_shared_memory():
    """示例4：使用 Value 和 Array 共享简单数据"""
    print(f"\n{'='*60}")
    print("示例4：Value/Array - 共享基本类型数据（性能更好）")
    print(f"{'='*60}")
    
    def counter(shared_value, lock, worker_id, iterations):
        """计数工作进程"""
        for _ in range(iterations):
            with lock:
                shared_value.value += 1
        
        print(f"工作进程 {worker_id}: 完成，当前值={shared_value.value}")
    
    def array_worker(shared_array, lock, worker_id):
        """数组工作进程"""
        for i in range(len(shared_array)):
            with lock:
                shared_array[i] += worker_id * 10
        
        print(f"工作进程 {worker_id}: 数组={list(shared_array)}")
    
    # 示例1：共享计数器
    print("\n--- 共享计数器 ---")
    lock = Lock()
    shared_value = Value('i', 0)  # 'i' 表示 int 类型
    
    processes = [Process(target=counter, args=(shared_value, lock, i, 1000)) for i in range(4)]
    
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    print(f"最终计数值: {shared_value.value} (期望: 4000)")
    
    # 示例2：共享数组
    print("\n--- 共享数组 ---")
    shared_array = Array('i', [0, 0, 0, 0, 0])  # 5 个整数的数组
    
    processes = [Process(target=array_worker, args=(shared_array, lock, i)) for i in range(3)]
    
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    print(f"最终数组: {list(shared_array)}")
    
    print("Value/Array 示例完成！")


def compare_ipc_methods():
    """对比各种 IPC 方式"""
    print(f"\n{'='*60}")
    print("各种 IPC 方式对比总结")
    print(f"{'='*60}")
    
    comparison = """
| 方式    | 适用场景                  | 性能 | 数据大小 | 使用复杂度 |
|---------|---------------------------|------|---------|-----------|
| Queue   | 多生产者多消费者          | 中   | 中等    | 低        |
| Pipe    | 两个进程之间双向通信      | 高   | 中等    | 低        |
| Manager | 共享复杂数据结构(dict/list) | 低 | 较大 | 中 |
| Value/Array | 共享简单数据类型        | 高   | 小      | 中        |

选择建议：
1. 简单的任务分发 -> 使用 Queue
2. 两个进程一对一通信 -> 使用 Pipe
3. 需要共享字典/列表 -> 使用 Manager
4. 只共享数字/简单数组 -> 使用 Value/Array

注意事项：
- 所有 IPC 方式传输的数据都必须是可序列化的（pickleable）
- 共享数据时注意使用锁，避免竞态条件
- 进程间通信有一定开销，不要过度频繁通信
    """
    
    print(comparison)


def main():
    print("进程间通信（IPC）示例程序")
    print("本程序演示多种进程间通信方式\n")
    
    example_queue()
    example_pipe()
    example_manager()
    example_shared_memory()
    compare_ipc_methods()
    
    print("\n所有示例完成！")


if __name__ == "__main__":
    main()
