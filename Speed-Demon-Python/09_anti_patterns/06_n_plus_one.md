# 06 - ORM N+1 查询

## 问题描述

N+1 查询是 ORM 中最常见的性能反模式之一。当你查询一组对象（1次查询），然后遍历每个对象并访问其关联对象时（N次查询），总共会产生 N+1 次数据库查询。这会导致性能急剧下降。

## 示例代码

```python
# 假设使用 Django ORM
class Author(models.Model):
    name = models.CharField(max_length=100)

class Book(models.Model):
    title = models.CharField(max_length=100)
    author = models.ForeignKey(Author, on_delete=models.CASCADE)

# 反模式：N+1 查询
def get_books_with_authors_bad():
    # 1 次查询：获取所有书籍
    books = Book.objects.all()
    
    result = []
    for book in books:
        # N 次查询：每次访问 book.author 都会查询数据库
        result.append(f"{book.title} by {book.author.name}")
    
    return result
```

## 性能对比

运行 `python 06_n_plus_one.py` 会输出类似：

```
N+1 查询模式:
  查询次数: 101 次 (1 + 100)
  总耗时: 0.52 秒
  每本书平均耗时: 5.2 毫秒

Eager Loading 模式:
  查询次数: 2 次
  总耗时: 0.03 秒
  每本书平均耗时: 0.3 毫秒
  性能提升: 17.3x
```

## 为什么会这样？

1. **延迟加载（Lazy Loading）**：ORM 默认只在访问关联对象时才加载数据
2. **网络往返开销**：每次数据库查询都有网络延迟
3. **查询解析开销**：数据库需要解析和执行每个查询
4. **无法利用批量优化**：多次小查询无法像单个大查询那样优化

## 修复方案

### 方案 1：使用 select_related（ForeignKey/OneToOneField）

```python
def get_books_with_authors_good1():
    # 使用 select_related 一次性加载关联的 author
    books = Book.objects.select_related('author').all()
    
    result = []
    for book in books:
        # 不会产生额外查询
        result.append(f"{book.title} by {book.author.name}")
    
    return result
```

### 方案 2：使用 prefetch_related（ManyToManyField/反向 ForeignKey）

```python
def get_authors_with_books_good():
    # 使用 prefetch_related 一次性加载所有书籍
    authors = Author.objects.prefetch_related('book_set').all()
    
    result = []
    for author in authors:
        # 不会产生额外查询
        book_titles = [book.title for book in author.book_set.all()]
        result.append(f"{author.name}: {book_titles}")
    
    return result
```

### 方案 3：使用 only/defer 减少字段加载

```python
def get_books_minimal():
    # 只加载需要的字段
    books = Book.objects.select_related('author').only('title', 'author__name')
    
    result = []
    for book in books:
        result.append(f"{book.title} by {book.author.name}")
    
    return result
```

### 方案 4：使用 annotate 进行聚合查询

```python
from django.db.models import Count

def get_authors_with_book_count():
    # 在数据库层面进行聚合
    authors = Author.objects.annotate(book_count=Count('book'))
    
    result = []
    for author in authors:
        result.append(f"{author.name} has {author.book_count} books")
    
    return result
```

## SQLAlchemy 中的对应方案

```python
# SQLAlchemy
from sqlalchemy.orm import joinedload

# 相当于 select_related
books = Book.query.options(joinedload(Book.author)).all()

# 相当于 prefetch_related
from sqlalchemy.orm import selectinload
authors = Author.query.options(selectinload(Author.books)).all()
```

## 如何检测 N+1 查询

### 方法 1：使用 Django Debug Toolbar

Django Debug Toolbar 可以显示每个请求的 SQL 查询，很容易发现 N+1 问题。

### 方法 2：使用 django-nplusone 库

```python
# 安装
# pip install django-nplusone

# 在 settings.py 中启用
INSTALLED_APPS += ['nplusone']
MIDDLEWARE += ['nplusone.ext.django.NPlusOneMiddleware']
```

### 方法 3：手动检查查询数量

```python
from django.db import connection

def test_n_plus_one():
    books = Book.objects.all()
    initial_count = len(connection.queries)
    
    for book in books:
        _ = book.author.name
    
    final_count = len(connection.queries)
    print(f"查询数量: {final_count - initial_count}")
```

## 总结

| ORM 方法 | 适用关系类型 | 查询次数 |
|---------|------------|---------|
| select_related | ForeignKey, OneToOne | 1 次 JOIN |
| prefetch_related | ManyToMany, 反向 ForeignKey | 2 次查询 |
| annotate | 聚合操作 | 1 次查询 |

**记住**：
1. **永远不要在循环中访问关联对象**，除非你已经预加载
2. **使用 Django Debug Toolbar 监控查询** - 发现问题的最佳方式
3. **对于列表接口，考虑使用 values_list** - 直接返回字典列表，不加载完整对象
4. **对于只读操作，使用 only() 或 defer()** - 减少数据传输

N+1 查询问题在小数据集上可能不明显，但当数据量增大时，性能会急剧下降。
