"""
Python 数据库性能优化实战代码示例

包含以下主题的代码示例：
1. ORM N+1 查询问题与解决方案
2. 连接池配置与监控
3. 批量操作优化
4. 索引优化
5. 查询分页优化
6. 读写分离基础

运行环境：
- SQLAlchemy >= 1.4
- SQLite (默认) 或 PostgreSQL
"""

import time
import random
import string
from contextlib import contextmanager
from functools import lru_cache
from datetime import datetime, timedelta

from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime,
    ForeignKey, Index, text, func, case
)
from sqlalchemy.orm import (
    sessionmaker, relationship, joinedload, selectinload,
    Session
)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


# ============================================================================
# 数据模型定义
# ============================================================================

class User(Base):
    """用户模型"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 关联关系
    posts = relationship('Post', back_populates='author', lazy='select')
    
    # 索引
    __table_args__ = (
        Index('idx_users_name', name),
        Index('idx_users_created_at', created_at),
        Index('idx_users_lower_email', func.lower(email)),
    )
    
    def __repr__(self):
        return f"<User(id={self.id}, name='{self.name}')>"


class Post(Base):
    """文章模型"""
    __tablename__ = 'posts'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    title = Column(String(200), nullable=False)
    content = Column(String(5000))
    status = Column(String(20), default='draft')  # draft, published, archived
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 关联关系
    author = relationship('User', back_populates='posts')
    comments = relationship('Comment', back_populates='post', lazy='select')
    
    # 复合索引
    __table_args__ = (
        Index('idx_posts_user_status_created', user_id, status, created_at),
        Index('idx_posts_published_created', created_at,
              postgresql_where=(status == 'published')),
    )
    
    def __repr__(self):
        return f"<Post(id={self.id}, title='{self.title[:30]}...')>"


class Comment(Base):
    """评论模型"""
    __tablename__ = 'comments'
    
    id = Column(Integer, primary_key=True)
    post_id = Column(Integer, ForeignKey('posts.id'), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    content = Column(String(1000), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 关联关系
    post = relationship('Post', back_populates='comments')
    
    # 索引
    __table_args__ = (
        Index('idx_comments_post_created', post_id, created_at),
    )
    
    def __repr__(self):
        return f"<Comment(id={self.id}, post_id={self.post_id})>"


# ============================================================================
# 1. ORM N+1 查询问题与解决方案
# ============================================================================

class NPlusOneExample:
    """N+1 查询示例"""
    
    def __init__(self, session):
        self.session = session
    
    def benchmark_n_plus_one(self):
        """N+1 查询 - 低效方式"""
        start = time.time()
        
        # 第一次查询：获取所有用户
        users = self.session.query(User).all()
        
        # 后续 N 次查询：每个用户的文章
        post_count = 0
        for user in users:
            post_count += len(user.posts)  # 触发额外查询
        
        elapsed = time.time() - start
        return {
            'method': 'N+1 查询',
            'elapsed': elapsed,
            'user_count': len(users),
            'post_count': post_count,
            'query_count': len(users) + 1
        }
    
    def benchmark_joinedload(self):
        """joinedload - JOIN 方式预先加载"""
        start = time.time()
        
        # 单次查询：使用 LEFT JOIN 一次性加载所有数据
        users = self.session.query(User).options(
            joinedload(User.posts)
        ).all()
        
        post_count = 0
        for user in users:
            post_count += len(user.posts)  # 不会触发额外查询
        
        elapsed = time.time() - start
        return {
            'method': 'joinedload',
            'elapsed': elapsed,
            'user_count': len(users),
            'post_count': post_count,
            'query_count': 1
        }
    
    def benchmark_selectinload(self):
        """selectinload - 子查询方式预先加载"""
        start = time.time()
        
        # 两次查询：1. 查用户 2. 查所有用户的文章
        users = self.session.query(User).options(
            selectinload(User.posts)
        ).all()
        
        post_count = 0
        for user in users:
            post_count += len(user.posts)  # 不会触发额外查询
        
        elapsed = time.time() - start
        return {
            'method': 'selectinload',
            'elapsed': elapsed,
            'user_count': len(users),
            'post_count': post_count,
            'query_count': 2
        }
    
    def benchmark_nested_eager_loading(self):
        """多层嵌套预先加载：用户 -> 文章 -> 评论"""
        start = time.time()
        
        users = self.session.query(User).options(
            selectinload(User.posts).selectinload(Post.comments)
        ).all()
        
        total_comments = 0
        for user in users:
            for post in user.posts:
                total_comments += len(post.comments)
        
        elapsed = time.time() - start
        return {
            'method': '嵌套 selectinload',
            'elapsed': elapsed,
            'user_count': len(users),
            'comment_count': total_comments,
            'query_count': 3
        }
    
    def run_all_benchmarks(self):
        """运行所有 N+1 相关的性能测试"""
        results = [
            self.benchmark_n_plus_one(),
            self.benchmark_joinedload(),
            self.benchmark_selectinload(),
            self.benchmark_nested_eager_loading(),
        ]
        
        print("\n" + "=" * 70)
        print("N+1 查询优化性能对比")
        print("=" * 70)
        print(f"{'方法':<25} {'耗时(ms)':>10} {'查询次数':>10} {'性能比':>10}")
        print("-" * 70)
        
        baseline = results[0]['elapsed']
        for r in results:
            ratio = baseline / r['elapsed']
            print(f"{r['method']:<25} {r['elapsed']*1000:>10.2f} "
                  f"{r['query_count']:>10} {ratio:>9.1f}x")
        
        return results


# ============================================================================
# 2. 连接池配置与监控
# ============================================================================

class ConnectionPoolExample:
    """连接池配置示例"""
    
    @staticmethod
    def create_web_app_engine():
        """Web 应用场景的连接池配置"""
        return create_engine(
            'sqlite:///example.db',
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=3600,
            pool_pre_ping=True,
            pool_use_lifo=True,
            echo=False
        )
    
    @staticmethod
    def create_background_task_engine():
        """后台任务场景的连接池配置"""
        return create_engine(
            'sqlite:///example.db',
            pool_size=5,
            max_overflow=10,
            pool_timeout=60,
            pool_recycle=3600,
            pool_pre_ping=True,
            echo=False
        )
    
    @staticmethod
    def get_pool_status(engine):
        """获取连接池状态"""
        pool = engine.pool
        if hasattr(pool, 'checkedin'):
            return {
                'checkedin': pool.checkedin(),
                'checkedout': pool.checkedout(),
                'overflow': getattr(pool, 'overflow', lambda: 0)(),
                'pool_size': pool.size(),
            }
        return None
    
    @staticmethod
    @contextmanager
    def safe_session(engine):
        """安全的 Session 上下文管理器，确保连接归还池"""
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()  # 确保连接归还池
    
    def demo_pool_monitoring(self):
        """演示连接池监控"""
        engine = self.create_web_app_engine()
        
        print("\n" + "=" * 70)
        print("连接池监控演示")
        print("=" * 70)
        
        # 初始状态
        status = self.get_pool_status(engine)
        if status:
            print(f"\n初始状态:")
            print(f"  池中连接: {status['checkedin']}")
            print(f"  已借出: {status['checkedout']}")
            print(f"  溢出连接: {status['overflow']}")
            print(f"  池大小: {status['pool_size']}")
        
        # 模拟并发请求
        sessions = []
        for i in range(5):
            Session = sessionmaker(bind=engine)
            session = Session()
            sessions.append(session)
            session.execute(text("SELECT 1"))
        
        # 借出后的状态
        status = self.get_pool_status(engine)
        if status:
            print(f"\n借出 5 个连接后:")
            print(f"  池中连接: {status['checkedin']}")
            print(f"  已借出: {status['checkedout']}")
        
        # 归还连接
        for session in sessions:
            session.close()
        
        # 归还后的状态
        status = self.get_pool_status(engine)
        if status:
            print(f"\n归还连接后:")
            print(f"  池中连接: {status['checkedin']}")
            print(f"  已借出: {status['checkedout']}")
        
        return engine


# ============================================================================
# 3. 批量操作优化
# ============================================================================

class BatchOperationExample:
    """批量操作优化示例"""
    
    def __init__(self, session):
        self.session = session
    
    @staticmethod
    def generate_test_data(count=1000):
        """生成测试数据"""
        return [
            {
                'name': f'User_{i:04d}',
                'email': f'user_{i:04d}@example.com'
            }
            for i in range(count)
        ]
    
    @staticmethod
    def batch_iterate(data, batch_size=1000):
        """分批迭代器"""
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]
    
    def benchmark_individual_insert(self, data):
        """逐条插入 - 低效方式"""
        start = time.time()
        
        for item in data:
            user = User(**item)
            self.session.add(user)
        self.session.commit()
        
        elapsed = time.time() - start
        return {
            'method': '逐条插入',
            'elapsed': elapsed,
            'count': len(data),
            'rows_per_sec': len(data) / elapsed
        }
    
    def benchmark_bulk_save(self, data):
        """bulk_save_objects 批量插入"""
        start = time.time()
        
        users = [User(**item) for item in data]
        self.session.bulk_save_objects(users)
        self.session.commit()
        
        elapsed = time.time() - start
        return {
            'method': 'bulk_save_objects',
            'elapsed': elapsed,
            'count': len(data),
            'rows_per_sec': len(data) / elapsed
        }
    
    def benchmark_bulk_insert_mappings(self, data):
        """bulk_insert_mappings 批量插入"""
        start = time.time()
        
        self.session.bulk_insert_mappings(User, data)
        self.session.commit()
        
        elapsed = time.time() - start
        return {
            'method': 'bulk_insert_mappings',
            'elapsed': elapsed,
            'count': len(data),
            'rows_per_sec': len(data) / elapsed
        }
    
    def benchmark_execute_batch(self, data):
        """execute 批量插入 - 最高性能"""
        start = time.time()
        
        self.session.execute(
            text("INSERT INTO users (name, email) VALUES (:name, :email)"),
            data
        )
        self.session.commit()
        
        elapsed = time.time() - start
        return {
            'method': 'execute 批量',
            'elapsed': elapsed,
            'count': len(data),
            'rows_per_sec': len(data) / elapsed
        }
    
    def benchmark_batch_update(self, count=1000):
        """批量更新性能测试"""
        # 准备更新数据
        users = self.session.query(User.id).limit(count).all()
        updates = [
            {'id': u.id, 'name': f'Updated_User_{u.id}'}
            for u in users
        ]
        
        start = time.time()
        
        # 使用 CASE WHEN 批量更新
        update_query = User.__table__.update().where(
            User.id.in_([u['id'] for u in updates])
        ).values(
            name=case(
                *[(User.id == u['id'], u['name']) for u in updates]
            )
        )
        self.session.execute(update_query)
        self.session.commit()
        
        elapsed = time.time() - start
        return {
            'method': '批量更新 (CASE WHEN)',
            'elapsed': elapsed,
            'count': len(updates),
            'rows_per_sec': len(updates) / elapsed
        }
    
    def benchmark_batch_delete(self, count=1000):
        """批量删除性能测试"""
        # 获取要删除的 ID
        ids_to_delete = [
            u.id for u in self.session.query(User.id)
            .order_by(User.id.desc()).limit(count).all()
        ]
        
        start = time.time()
        
        self.session.query(User).filter(
            User.id.in_(ids_to_delete)
        ).delete(synchronize_session=False)
        self.session.commit()
        
        elapsed = time.time() - start
        return {
            'method': '批量删除',
            'elapsed': elapsed,
            'count': len(ids_to_delete),
            'rows_per_sec': len(ids_to_delete) / elapsed
        }
    
    def run_all_benchmarks(self, data_size=5000):
        """运行所有批量操作测试"""
        test_data = self.generate_test_data(data_size)
        
        print("\n" + "=" * 70)
        print("批量操作优化性能对比")
        print("=" * 70)
        print(f"{'方法':<25} {'耗时(ms)':>10} {'行数/秒':>12} {'性能比':>10}")
        print("-" * 70)
        
        # 清空表
        self.session.query(User).delete()
        self.session.commit()
        
        # 插入测试
        results = []
        results.append(self.benchmark_individual_insert(test_data[:1000]))
        
        self.session.query(User).delete()
        self.session.commit()
        results.append(self.benchmark_bulk_save(test_data))
        
        self.session.query(User).delete()
        self.session.commit()
        results.append(self.benchmark_bulk_insert_mappings(test_data))
        
        self.session.query(User).delete()
        self.session.commit()
        results.append(self.benchmark_execute_batch(test_data))
        
        # 输出结果
        baseline = results[0]['elapsed'] / results[0]['count'] * 1000
        for r in results:
            ratio = baseline / (r['elapsed'] / r['count'] * 1000)
            print(f"{r['method']:<25} {r['elapsed']*1000:>10.2f} "
                  f"{r['rows_per_sec']:>12.0f} {ratio:>9.1f}x")
        
        # 更新和删除测试
        update_result = self.benchmark_batch_update(1000)
        delete_result = self.benchmark_batch_delete(1000)
        
        print(f"\n{update_result['method']:<25} {update_result['elapsed']*1000:>10.2f} "
              f"{update_result['rows_per_sec']:>12.0f}")
        print(f"{delete_result['method']:<25} {delete_result['elapsed']*1000:>10.2f} "
              f"{delete_result['rows_per_sec']:>12.0f}")
        
        return results


# ============================================================================
# 4. 索引优化
# ============================================================================

class IndexOptimizationExample:
    """索引优化示例"""
    
    def __init__(self, session):
        self.session = session
    
    def explain_query(self, query_str):
        """分析查询执行计划"""
        result = self.session.execute(text(f"EXPLAIN QUERY PLAN {query_str}"))
        print("\n查询执行计划:")
        for row in result:
            print(f"  {row[0]}")
    
    def benchmark_query_with_index(self):
        """测试索引查询性能"""
        print("\n" + "=" * 70)
        print("索引优化性能测试")
        print("=" * 70)
        
        # 主键查询 (有索引)
        start = time.time()
        user = self.session.query(User).filter(User.id == 1).first()
        pk_time = time.time() - start
        print(f"\n主键查询: {pk_time * 1000:.4f}ms")
        
        # 索引列查询
        start = time.time()
        user = self.session.query(User).filter(User.email == 'user_0001@example.com').first()
        idx_time = time.time() - start
        print(f"索引列查询 (email): {idx_time * 1000:.4f}ms")
        
        # 普通列查询 (无索引)
        start = time.time()
        users = self.session.query(User).filter(User.name == 'User_0001').all()
        no_idx_time = time.time() - start
        print(f"普通列查询 (name): {no_idx_time * 1000:.4f}ms")
        
        # 复合索引查询
        start = time.time()
        posts = self.session.query(Post).filter(
            Post.user_id == 1,
            Post.status == 'published'
        ).order_by(Post.created_at.desc()).limit(10).all()
        composite_idx_time = time.time() - start
        print(f"复合索引查询: {composite_idx_time * 1000:.4f}ms")
        
        # 分析查询计划
        print("\n分析查询计划示例:")
        self.explain_query("SELECT * FROM users WHERE email = 'test@example.com'")
        
        return {
            'pk_query': pk_time,
            'indexed_query': idx_time,
            'non_indexed_query': no_idx_time,
            'composite_index_query': composite_idx_time
        }


# ============================================================================
# 5. 查询分页优化
# ============================================================================

class PaginationOptimizationExample:
    """分页优化示例"""
    
    def __init__(self, session):
        self.session = session
    
    def benchmark_offset_pagination(self, page=100, per_page=20):
        """传统 OFFSET 分页"""
        offset = (page - 1) * per_page
        
        start = time.time()
        posts = self.session.query(Post).order_by(
            Post.id
        ).limit(per_page).offset(offset).all()
        elapsed = time.time() - start
        
        return {
            'method': f'OFFSET 分页 (page={page})',
            'elapsed': elapsed,
            'page': page,
            'offset': offset,
            'count': len(posts)
        }
    
    def benchmark_cursor_pagination(self, last_id=2000, limit=20):
        """游标分页 (Seek Method)"""
        start = time.time()
        posts = self.session.query(Post).filter(
            Post.id > last_id
        ).order_by(Post.id).limit(limit).all()
        elapsed = time.time() - start
        
        return {
            'method': f'游标分页 (last_id={last_id})',
            'elapsed': elapsed,
            'last_id': last_id,
            'count': len(posts)
        }
    
    def benchmark_delay_join_pagination(self, page=100, per_page=20):
        """延迟关联分页"""
        offset = (page - 1) * per_page
        
        start = time.time()
        
        # 第一步：只查询 ID（覆盖索引）
        subquery = self.session.query(Post.id).order_by(
            Post.id
        ).limit(per_page).offset(offset).subquery()
        
        # 第二步：JOIN 获取完整数据
        posts = self.session.query(Post).join(
            subquery,
            Post.id == subquery.c.id
        ).order_by(Post.id).all()
        
        elapsed = time.time() - start
        
        return {
            'method': f'延迟关联分页 (page={page})',
            'elapsed': elapsed,
            'page': page,
            'count': len(posts)
        }
    
    @staticmethod
    def get_estimated_count(session):
        """使用统计信息估算总数"""
        result = session.execute(text("""
            SELECT reltuples::bigint AS estimate
            FROM pg_class
            WHERE relname = 'posts'
        """))
        try:
            return result.scalar()
        except Exception:
            # SQLite fallback
            result = session.execute(text("SELECT COUNT(*) FROM posts"))
            return result.scalar()
    
    def run_all_benchmarks(self):
        """运行所有分页测试"""
        print("\n" + "=" * 70)
        print("分页优化性能对比")
        print("=" * 70)
        print(f"{'方法':<30} {'页码/位置':>15} {'耗时(ms)':>12}")
        print("-" * 70)
        
        # 不同深度的 OFFSET 分页
        for page in [1, 10, 100, 500]:
            result = self.benchmark_offset_pagination(page=page)
            print(f"{result['method']:<30} page={result['page']:>6} "
                  f"{result['elapsed']*1000:>12.4f}")
        
        # 游标分页
        for last_id in [0, 200, 2000, 10000]:
            result = self.benchmark_cursor_pagination(last_id=last_id)
            print(f"{result['method']:<30} id={result['last_id']:>8} "
                  f"{result['elapsed']*1000:>12.4f}")
        
        # 延迟关联分页
        for page in [100, 500]:
            result = self.benchmark_delay_join_pagination(page=page)
            print(f"{result['method']:<30} page={result['page']:>6} "
                  f"{result['elapsed']*1000:>12.4f}")


# ============================================================================
# 6. 读写分离基础
# ============================================================================

class MasterSlaveRouter:
    """主从读写分离路由"""
    
    def __init__(self, master_engine, slave_engines):
        self.master_engine = master_engine
        self.slave_engines = slave_engines
        self._slave_index = 0
        self.MasterSession = sessionmaker(bind=master_engine)
        self.SlaveSessions = [sessionmaker(bind=e) for e in slave_engines]
    
    def get_read_session(self):
        """获取读会话（轮询从库）"""
        session_factory = self.SlaveSessions[self._slave_index]
        self._slave_index = (self._slave_index + 1) % len(self.SlaveSessions)
        return session_factory()
    
    def get_write_session(self):
        """获取写会话（主库）"""
        return self.MasterSession()
    
    def force_master_read(self):
        """强制主库读取（用于刚写入后的数据）"""
        return self.MasterSession()


class ReadWriteSeparationExample:
    """读写分离示例"""
    
    @staticmethod
    def create_router():
        """创建读写分离路由"""
        # 实际环境中应该配置不同的主从连接
        # 这里为了演示使用同一个 SQLite 数据库
        master_engine = create_engine('sqlite:///example.db', echo=False)
        slave_engines = [
            create_engine('sqlite:///example.db', echo=False),
        ]
        
        return MasterSlaveRouter(master_engine, slave_engines)
    
    @staticmethod
    def demo_read_write_separation():
        """演示读写分离"""
        print("\n" + "=" * 70)
        print("读写分离演示")
        print("=" * 70)
        
        router = ReadWriteSeparationExample.create_router()
        
        # 写操作：走主库
        print("\n写操作（主库）:")
        with router.get_write_session() as session:
            new_user = User(
                name=f'TestUser_{random.randint(1, 1000)}',
                email=f'test_{random.randint(1, 1000)}@example.com'
            )
            session.add(new_user)
            session.commit()
            print(f"  写入用户: {new_user}")
        
        # 读操作：走从库
        print("\n读操作（从库）:")
        with router.get_read_session() as session:
            count = session.query(User).count()
            print(f"  用户总数: {count}")
        
        # 强制主库读：避免主从延迟
        print("\n强制主库读（写入后立即读取）:")
        with router.force_master_read() as session:
            user = session.query(User).order_by(User.id.desc()).first()
            print(f"  刚写入的用户: {user}")


# ============================================================================
# 辅助函数：数据生成
# ============================================================================

def generate_sample_data(session, user_count=100, posts_per_user=5, comments_per_post=3):
    """生成示例数据"""
    print(f"\n生成示例数据: {user_count} 用户, "
          f"{user_count * posts_per_user} 文章, "
          f"{user_count * posts_per_user * comments_per_post} 评论...")
    
    # 生成用户
    users = []
    for i in range(user_count):
        user = User(
            name=f'User_{i:04d}',
            email=f'user_{i:04d}@example.com',
            created_at=datetime.utcnow() - timedelta(days=random.randint(0, 365))
        )
        users.append(user)
    session.bulk_save_objects(users)
    session.commit()
    
    # 生成文章
    user_ids = [u.id for u in session.query(User.id).all()]
    posts = []
    statuses = ['draft', 'published', 'archived']
    
    for user_id in user_ids:
        for j in range(posts_per_user):
            post = Post(
                user_id=user_id,
                title=f'Post Title {j} by User {user_id}',
                content='This is the content of the post... ' * 10,
                status=random.choice(statuses),
                created_at=datetime.utcnow() - timedelta(days=random.randint(0, 30))
            )
            posts.append(post)
    session.bulk_save_objects(posts)
    session.commit()
    
    # 生成评论
    post_ids = [p.id for p in session.query(Post.id).all()]
    comments = []
    for post_id in post_ids:
        for k in range(comments_per_post):
            commenter_id = random.choice(user_ids)
            comment = Comment(
                post_id=post_id,
                user_id=commenter_id,
                content=f'Comment {k} on post {post_id}',
                created_at=datetime.utcnow() - timedelta(days=random.randint(0, 7))
            )
            comments.append(comment)
    session.bulk_save_objects(comments)
    session.commit()
    
    print("数据生成完成!")


# ============================================================================
# 主运行函数
# ============================================================================

def main():
    """运行所有示例"""
    print("=" * 70)
    print("Python 数据库性能优化实战代码示例")
    print("=" * 70)
    
    # 创建引擎和 Session
    engine = create_engine('sqlite:///example.db', echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    
    # 生成测试数据
    with Session() as session:
        user_count = session.query(User).count()
        if user_count < 50:
            generate_sample_data(session, user_count=100, posts_per_user=10)
    
    # 1. N+1 查询优化
    with Session() as session:
        n_plus_one = NPlusOneExample(session)
        n_plus_one.run_all_benchmarks()
    
    # 2. 连接池监控
    pool_example = ConnectionPoolExample()
    pool_example.demo_pool_monitoring()
    
    # 3. 批量操作优化
    with Session() as session:
        batch_example = BatchOperationExample(session)
        batch_example.run_all_benchmarks(data_size=5000)
    
    # 4. 索引优化
    with Session() as session:
        index_example = IndexOptimizationExample(session)
        index_example.benchmark_query_with_index()
    
    # 5. 分页优化
    with Session() as session:
        pagination_example = PaginationOptimizationExample(session)
        pagination_example.run_all_benchmarks()
    
    # 6. 读写分离
    ReadWriteSeparationExample.demo_read_write_separation()
    
    print("\n" + "=" * 70)
    print("所有示例运行完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
