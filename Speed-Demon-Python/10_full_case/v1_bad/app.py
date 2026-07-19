from flask import Flask, jsonify
import time
import requests
import random

app = Flask(__name__)

# 性能坑点5: 全局变量内存泄漏 - 不断累积的请求历史
request_history = []

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

def get_product_from_db(product_id):
    """模拟数据库查询 - 慢查询"""
    time.sleep(0.05)  # 模拟数据库延迟
    return products_db.get(product_id)

def get_category_from_db(category_id):
    """模拟数据库查询 - N+1查询中的单个查询"""
    time.sleep(0.03)  # 模拟数据库延迟
    return categories_db.get(category_id)

def get_seller_from_db(seller_id):
    """模拟数据库查询 - N+1查询中的单个查询"""
    time.sleep(0.03)  # 模拟数据库延迟
    return sellers_db.get(seller_id)

def get_reviews_from_db(product_id):
    """模拟数据库查询 - N+1查询中的单个查询"""
    time.sleep(0.04)  # 模拟数据库延迟
    return reviews_db.get(product_id, [])

def get_inventory_external(product_id):
    """性能坑点3: 同步HTTP调用 + 性能坑点4: 无连接池
    
    每次调用都创建新的HTTP连接
    """
    try:
        # 模拟外部HTTP调用
        time.sleep(0.1)  # 模拟网络延迟
        response = requests.get(f"http://httpbin.org/delay/0.1", timeout=2)
        return {"stock": random.randint(0, 100), "warehouse": "上海仓"}
    except:
        return {"stock": 0, "warehouse": "未知"}

def get_price_history_external(product_id):
    """性能坑点3: 同步HTTP调用"""
    try:
        time.sleep(0.08)  # 模拟网络延迟
        response = requests.get(f"http://httpbin.org/delay/0.08", timeout=2)
        return {"lowest_price": 7000, "highest_price": 8500, "current_price": 7999}
    except:
        return {"lowest_price": 0, "highest_price": 0, "current_price": 0}

def build_response(product, category, seller, reviews, inventory, price_history):
    """性能坑点6: 循环append低效
    
    使用低效的字符串拼接和列表append
    """
    result = {}
    result["product"] = product
    
    # 低效的列表构建方式
    review_list = []
    for review in reviews:
        # 每次都创建新的字典并append
        review_list.append({
            "id": review["id"],
            "user_id": review["user_id"],
            "content": review["content"],
            "rating": review["rating"]
        })
    result["reviews"] = review_list
    
    # 低效的字典构建
    result["category"] = {}
    result["category"]["id"] = category["id"]
    result["category"]["name"] = category["name"]
    result["category"]["description"] = category["description"]
    
    result["seller"] = {}
    result["seller"]["id"] = seller["id"]
    result["seller"]["name"] = seller["name"]
    result["seller"]["rating"] = seller["rating"]
    
    result["inventory"] = inventory
    result["price_history"] = price_history
    
    return result

@app.route('/product/<int:product_id>')
def get_product_detail(product_id):
    """电商商品详情接口 - 存在多个性能问题
    
    性能坑点清单:
    1. N+1查询: 获取商品后分别查询分类、卖家、评论
    2. 无缓存: 每次请求都重新查询所有数据
    3. 同步HTTP调用: 串行调用库存和价格历史接口
    4. 无连接池: 每次HTTP调用都创建新连接
    5. 全局变量内存泄漏: request_history不断累积
    6. 循环append低效: 使用低效的列表和字典构建方式
    """
    start_time = time.time()
    
    # 性能坑点5: 全局变量内存泄漏
    request_history.append({
        "product_id": product_id,
        "timestamp": time.time()
    })
    
    # 性能坑点1: N+1查询
    # 第1次查询: 获取商品基本信息
    product = get_product_from_db(product_id)
    if not product:
        return jsonify({"error": "商品不存在"}), 404
    
    # 第2次查询: N+1 - 查询分类
    category = get_category_from_db(product["category_id"])
    
    # 第3次查询: N+1 - 查询卖家
    seller = get_seller_from_db(product["seller_id"])
    
    # 第4次查询: N+1 - 查询评论
    reviews = get_reviews_from_db(product_id)
    
    # 性能坑点3: 同步HTTP调用 - 串行执行
    inventory = get_inventory_external(product_id)
    price_history = get_price_history_external(product_id)
    
    # 性能坑点6: 循环append低效
    response = build_response(product, category, seller, reviews, inventory, price_history)
    
    # 添加耗时信息
    response["execution_time"] = time.time() - start_time
    response["request_count"] = len(request_history)
    
    return jsonify(response)

@app.route('/health')
def health():
    return jsonify({"status": "ok", "memory_leak_items": len(request_history)})

if __name__ == '__main__':
    print("=" * 60)
    print("电商商品详情接口 - 性能问题版本 (v1_bad)")
    print("=" * 60)
    print("性能坑点:")
    print("  1. N+1查询: 获取商品后分别查询分类、卖家、评论")
    print("  2. 无缓存: 每次请求都重新查询所有数据")
    print("  3. 同步HTTP调用: 串行调用库存和价格历史接口")
    print("  4. 无连接池: 每次HTTP调用都创建新连接")
    print("  5. 全局变量内存泄漏: request_history不断累积")
    print("  6. 循环append低效: 使用低效的列表和字典构建方式")
    print("=" * 60)
    print("测试接口:")
    print("  GET http://localhost:5000/product/1")
    print("  GET http://localhost:5000/health")
    print("=" * 60)
    app.run(debug=False, port=5000)
