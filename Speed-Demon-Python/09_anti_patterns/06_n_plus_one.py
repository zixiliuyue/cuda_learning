import time
import sqlite3
from dataclasses import dataclass
from typing import List

@dataclass
class Author:
    id: int
    name: str

@dataclass
class Book:
    id: int
    title: str
    author_id: int

class MockORM:
    def __init__(self):
        self.conn = sqlite3.connect(':memory:')
        self._create_tables()
        self._seed_data()
        self.query_count = 0
    
    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE authors (
                id INTEGER PRIMARY KEY,
                name TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE books (
                id INTEGER PRIMARY KEY,
                title TEXT,
                author_id INTEGER,
                FOREIGN KEY (author_id) REFERENCES authors (id)
            )
        ''')
        self.conn.commit()
    
    def _seed_data(self):
        cursor = self.conn.cursor()
        for i in range(10):
            cursor.execute('INSERT INTO authors (name) VALUES (?)', (f'Author {i}',))
        for i in range(100):
            cursor.execute('INSERT INTO books (title, author_id) VALUES (?, ?)',
                         (f'Book {i}', (i % 10) + 1))
        self.conn.commit()
    
    def get_all_books(self) -> List[Book]:
        self.query_count += 1
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, title, author_id FROM books')
        return [Book(*row) for row in cursor.fetchall()]
    
    def get_author_by_id(self, author_id: int) -> Author:
        self.query_count += 1
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, name FROM authors WHERE id = ?', (author_id,))
        row = cursor.fetchone()
        return Author(*row) if row else None
    
    def get_all_books_with_authors_eager(self) -> List[tuple]:
        self.query_count += 1
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT b.id, b.title, b.author_id, a.name
            FROM books b
            JOIN authors a ON b.author_id = a.id
        ''')
        return [(Book(row[0], row[1], row[2]), Author(row[2], row[3])) for row in cursor.fetchall()]

def n_plus_one_query(orm: MockORM):
    """反模式：N+1 查询"""
    orm.query_count = 0
    books = orm.get_all_books()
    
    result = []
    for book in books:
        author = orm.get_author_by_id(book.author_id)
        result.append(f"{book.title} by {author.name}")
    
    return result, orm.query_count

def eager_loading_query(orm: MockORM):
    """修复方案：Eager Loading (使用 JOIN)"""
    orm.query_count = 0
    books_with_authors = orm.get_all_books_with_authors_eager()
    
    result = []
    for book, author in books_with_authors:
        result.append(f"{book.title} by {author.name}")
    
    return result, orm.query_count

def simulate_batch_prefetch(orm: MockORM):
    """修复方案：批量预取 (两次查询)"""
    orm.query_count = 0
    
    books = orm.get_all_books()
    author_ids = list(set(book.author_id for book in books))
    
    authors = {}
    cursor = orm.conn.cursor()
    orm.query_count += 1
    cursor.execute('SELECT id, name FROM authors WHERE id IN ({})'.format(
        ','.join('?' * len(author_ids))
    ), author_ids)
    for row in cursor.fetchall():
        authors[row[0]] = Author(*row)
    
    result = []
    for book in books:
        author = authors[book.author_id]
        result.append(f"{book.title} by {author.name}")
    
    return result, orm.query_count

if __name__ == "__main__":
    orm = MockORM()
    
    print("=== N+1 查询 vs Eager Loading 对比 ===")
    print("测试数据: 100 本书, 10 个作者\n")
    
    start = time.time()
    result1, query_count1 = n_plus_one_query(orm)
    time1 = time.time() - start
    
    print("N+1 查询模式:")
    print(f"  查询次数: {query_count1} 次")
    print(f"  总耗时: {time1:.4f} 秒")
    print(f"  每本书平均耗时: {time1 / 100 * 1000:.2f} 毫秒")
    
    start = time.time()
    result2, query_count2 = eager_loading_query(orm)
    time2 = time.time() - start
    
    print("\nEager Loading 模式 (JOIN):")
    print(f"  查询次数: {query_count2} 次")
    print(f"  总耗时: {time2:.4f} 秒")
    print(f"  每本书平均耗时: {time2 / 100 * 1000:.2f} 毫秒")
    print(f"  性能提升: {time1 / time2:.1f}x")
    
    start = time.time()
    result3, query_count3 = simulate_batch_prefetch(orm)
    time3 = time.time() - start
    
    print("\n批量预取模式 (两次查询):")
    print(f"  查询次数: {query_count3} 次")
    print(f"  总耗时: {time3:.4f} 秒")
    print(f"  每本书平均耗时: {time3 / 100 * 1000:.2f} 毫秒")
    print(f"  性能提升: {time1 / time3:.1f}x")
    
    print("\n=== 结果验证 ===")
    print(f"三种方式结果一致: {result1 == result2 == result3}")
    
    print("\n=== 前 5 条结果 ===")
    for r in result1[:5]:
        print(f"  - {r}")
    
    print("\n=== 扩展性测试: 查询次数随数据量增长 ===")
    print("| 书籍数量 | N+1 查询次数 | Eager 查询次数 |")
    print("|---------|-------------|---------------|")
    for n in [10, 100, 1000]:
        orm2 = MockORM()
        
        books = orm2.get_all_books()[:n]
        orm2.query_count = 0
        for book in books:
            orm2.get_author_by_id(book.author_id)
        n_plus_one_count = orm2.query_count + 1
        
        print(f"| {n:7} | {n_plus_one_count:11} | {2:13} |")
