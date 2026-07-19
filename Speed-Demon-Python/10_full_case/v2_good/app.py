from flask import Flask, jsonify
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import asyncio
import threading
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# 优化5: 使用LRU缓存替代全局变量避免内存泄漏
# 使用有界缓存，最多缓存1000个商品数据
product_cache = {}
cache_lock = threading.Lock()
MAX_CACHE_SIZE = 1000
CACHE_TTL = 60  # 缓存过期时间60秒

# 模拟商品数据库
products_db = {
    1: {"id": 1, "name": "iPhone 15 Pro", "price": 7999, "category_id": 1, "seller_id": 1},
    2: {"id": 2, "name": "MacBook Pro", "price": 14999, "category_id": 2, "seller_id": 1},
    3: {"id": 3, "name": "AirPods Pro", "price": 1899, "category_id": 3, "seller_id": 2},
    4: {"id": 4, "name": "iPad Air", "price": 4799, "category_id": 4, "seller_id": 1},
    5: {"id": 5, "name": "Apple Watch", "price": 2999, "category_id": 5, "seller_id": 2},
}

# 模拟分类数据库
categories_db = {
    1: {"id": 1, "name": "手机", "description": "智能手机"},
    2: {"id": 2, "name": "笔记本", "description": "笔记本电脑"},
    3: {"id": 3, "name": "耳机", "description": "无线耳机"},
    4: {"id": 4, "name": "平板", "description": "平板电脑"},
    5: {"id": 5, "name": "手表", "description": "智能手表"},
}

# 模拟卖家数据库
sellers_db = {
    1: {"id": 1, "name": "Apple官方旗舰店", "rating": 4.9},
    2: {"id": 2, "name": "Apple授权专营店", "rating": 4.8},
}

# 模拟评论数据
reviews_db = {
    1: [
        {"id": 1, "user_id": 101, "content": "很好用", "rating": 5},
        {"id": 2, "user_id": 102, "content": "质量不错", "rating": 4},
        {"id": 3, "user_id": 103, "content": "发货快", "rating": 5},
    ],
    2: [
        {"id": 4, "user_id": 104, "content": "性能强劲", "rating": 5},
        {"id": 5, "user_id": 105, "content": "散热一般", "rating": 4},
    ],
    3: [
        {"id": 6, "user_id": 106, "content": "音质很好", "rating": 5},
        {"id": 7, "user_id": 107, "content": "降噪很棒", "rating": 5},
        {"id": 8, "user_id": 108, "content": "续航一般", "rating": 3},
    ],
}

# 优化4: 创建HTTP连接池 - 使用Session复用连接
def create_http_session():
    """创建带连接池和重试的HTTP会话"""
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.1)
    adapter = HTTPAdapter(
        pool_connections=10,  # 连接池大小
        pool_maxsize=100,     # 最大连接数
        max_retries=retry
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

http_session = create_http_session()

# 优化2: 使用LRU缓存数据库查询
@lru_cache(maxsize=128)
def get_product_from_db(product_id):
    """优化后的数据库查询 - 带缓存"""
    time.sleep(0.05)  # 模拟数据库延迟
    return products_db.get(product_id)

@lru_cache(maxsize=32)
def get_category_from_db(category_id):
    """优化后的数据库查询 - 带缓存"""
    time.sleep(0.03)  # 模拟数据库延迟
    return categories_db.get(category_id)

@lru_cache(maxsize=32)
def get_seller_from_db(seller_id):
    """优化后的数据库查询 - 带缓存"""
    time.sleep(0.03)  # 模拟数据库延迟
    return sellers_db.get(seller_id)

@lru_cache(maxsize=128)
def get_reviews_from_db(product_id):
    """优化后的数据库查询 - 带缓存"""
    time.sleep(0.04)  # 模拟数据库延迟
    return reviews_db.get(product_id, [])

# 优化3: 使用连接池进行HTTP调用
def get_inventory_external(product_id):
    """优化后的HTTP调用 - 使用连接池"""
    try:
        time.sleep(0.1)  # 模拟网络延迟
        # 使用连接池的session
        response = http_session.get("http://httpbin.org/delay/0.1", timeout=2)
        import random
        return {"stock": random.randint(0, 100), "warehouse": "上海仓"}
    except:
        return {"stock": 0, "warehouse": "未知"}

def get_price_history_external(product_id):
    """优化后的HTTP调用 - 使用连接池"""
    try:
        time.sleep(0.08)  # 模拟网络延迟
        response = http_session.get("http://httpbin.org/delay/0.08", timeout=2)
        return {"lowest_price": 7000, "highest_price": 8500, "current_price": 7999}
    except:
        return {"lowest_price": 0, "highest_price": 0, "current_price": 0}

# 优化6: 使用列表推导式和字典直接构建替代低效的append
def build_response(product, category, seller, reviews, inventory, price_history):
    """优化后的响应构建 - 使用高效的数据结构构建"""
    # 使用列表推导式替代循环append
    review_list = [
        {
            "id": review["id"],
            "user_id": review["user_id"],
            "content": review["content"],
            "rating": review["rating"]
        }
        for review in reviews
    ]
    
    # 使用字典字面量直接构建
    return {
        "product": product,
        "category": {
            "id": category["id"],
            "name": category["name"],
            "description": category["description"]
        },
        "seller": {
            "id": seller["id"],
            "name": seller["name"],
            "rating": seller["rating"]
        },
        "reviews": review_list,
        "inventory": inventory,
        "price_history": price_history
    }

def get_from_cache(cache_key):
    """从缓存获取数据"""
    with cache_lock:
        if cache_key in product_cache:
            cache_item = product_cache[cache_key]
            if time.time() - cache_item["timestamp"] < CACHE_TTL:
                return cache_item["data"]
            else:
                # 缓存过期，删除
                del product_cache[cache_key]
    return None

def set_to_cache(cache_key, data):
    """设置缓存"""
    with cache_lock:
        # 如果缓存已满，删除最旧的
        if len(product_cache) >= MAX_CACHE_SIZE:
            oldest_key = min(product_cache.keys(), key=lambda k: product_cache[k]["timestamp"])
            del product_cache[oldest_key]
        product_cache[cache_key] = {
            "data": data,
            "timestamp": time.time()
        }

# 优化1: 解决N+1查询 - 使用批量加载和预取
def get_product_detail_batch(product_id):
    """批量获取商品详情数据"""
    # 第1次查询: 获取商品基本信息
    product = get_product_from_db(product_id)
    if not product:
        return None, None, None, None
    
    # 使用线程池并行查询，避免N+1串行
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_category = executor.submit(get_category_from_db, product["category_id"])
        future_seller = executor.submit(get_seller_from_db, product["seller_id"])
        future_reviews = executor.submit(get_reviews_from_db, product_id)
        
        category = future_category.result()
        seller = future_seller.result()
        reviews = future_reviews.result()
    
    return product, category, seller, reviews

# 优化3: 并发HTTP调用
def get_external_data_parallel(product_id):
    """并行获取外部数据"""
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_inventory = executor.submit(get_inventory_external, product_id)
        future_price_history = executor.submit(get_price_history_external, product_id)
        
        inventory = future_inventory.result()
        price_history = future_price_history.result()
    
    return inventory, price_history

@app.route('/product/<int:product_id>')
def get_product_detail(product_id):
    """电商商品详情接口 - 优化后的版本
    
    优化清单:
    1. 解决N+1查询: 使用线程池并行查询分类、卖家、评论
    2. 添加缓存: 使用LRU缓存数据库查询结果
    3. 并发HTTP调用: 使用线程池并行调用外部接口
    4. HTTP连接池: 复用HTTP连接，避免每次创建新连接
    5. 解决内存泄漏: 使用有界缓存替代无限增长的全局变量
    6. 高效数据构建: 使用列表推导式和字典字面量
    """
    start_time = time.time()
    
    # 优化2: 尝试从缓存获取完整响应
    cache_key = f"product:{product_id}"
    cached_response = get_from_cache(cache_key)
    if cached_response:
        cached_response["execution_time"] = time.time() - start_time
        cached_response["cached"] = True
        cached_response["cache_size"] = len(product_cache)
        return jsonify(cached_response)
    
    # 优化1: 批量获取商品相关数据（并行查询，避免N+1）
    product, category, seller, reviews = get_product_detail_batch(product_id)
    if not product:
        return jsonify({"error": "商品不存在"}), 404
    
    # 优化3: 并行获取外部HTTP数据
    inventory, price_history = get_external_data_parallel(product_id)
    
    # 优化6: 高效构建响应
    response = build_response(product, category, seller, reviews, inventory, price_history)
    
    # 优化2: 缓存响应结果
    set_to_cache(cache_key, response)
    
    # 添加耗时信息
    response["execution_time"] = time.time() - start_time
    response["cached"] = False
    response["cache_size"] = len(product_cache)
    
    return jsonify(response)

@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "cache_size": len(product_cache),
        "max_cache_size": MAX_CACHE_SIZE
    })

@app.route('/cache/clear')
def clear_cache():
    """清空缓存"""
    with cache_lock:
        product_cache.clear()
    get_product_from_db.cache_clear()
    get_category_from_db.cache_clear()
    get_seller_from_db.cache_clear()
    get_reviews_from_db.cache_clear()
    return jsonify({"status": "cache cleared"})

if __name__ == '__main__':
    print("=" * 60)
    print("电商商品详情接口 - 优化版本 (v2_good)")
    print("=" * 60)
    print("优化清单:")
    print("  1. 解决N+1查询: 使用线程池并行查询分类、卖家、评论")
    print("  2. 添加缓存: 使用LRU缓存数据库查询结果")
    print("  3. 并发HTTP调用: 使用线程池并行调用外部接口")
    print("  4. HTTP连接池: 复用HTTP连接，避免每次创建新连接")
    print("  5. 解决内存泄漏: 使用有界缓存替代无限增长的全局变量")
    print("  6. 高效数据构建: 使用列表推导式和字典字面量")
    print("=" * 60)
    print("测试接口:")
    print("  GET http://localhost:5000/product/1")
    print("  GET http://localhost:5000/health")
    print("  GET http://localhost:5000/cache/clear")
    print("=" * 60)
    app.run(debug=False, port=5001)
