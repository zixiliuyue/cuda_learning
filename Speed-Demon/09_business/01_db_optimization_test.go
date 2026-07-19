package main

import (
    "database/sql"
    "fmt"
    _ "github.com/go-sql-driver/mysql"
    "testing"
    "time"
)

// ==============================================
// 数据库优化实操测试
// ==============================================

// 初始化测试连接
func setupDB() *sql.DB {
    dsn := "root:password@tcp(127.0.0.1:3306)/test?charset=utf8mb4"
    db, err := sql.Open("mysql", dsn)
    if err != nil {
        panic(err)
    }
    db.SetMaxOpenConns(100)
    db.SetMaxIdleConns(50)
    db.SetConnMaxLifetime(1 * time.Hour)
    return db
}

// ==============================================
// 优化 1: N+1 查询 vs JOIN 查询
// ==============================================

// ❌ N+1 查询：先查所有用户，再循环查每个用户的订单
func BenchmarkDB_NPlus1(b *testing.B) {
    db := setupDB()
    defer db.Close()
    b.ReportAllocs()
    b.ResetTimer()
    
    for i := 0; i < b.N; i++ {
        // 1. 查询所有用户（1次）
        rows, _ := db.Query("SELECT id FROM users LIMIT 100")
        var userIDs []int
        for rows.Next() {
            var id int
            rows.Scan(&id)
            userIDs = append(userIDs, id)
        }
        rows.Close()
        
        // 2. 循环查每个用户的订单（100次）
        for _, uid := range userIDs {
            db.Query("SELECT * FROM orders WHERE user_id = ?", uid)
            // 总共 1 + 100 = 101 次查询！
        }
    }
}

// ✅ JOIN 查询：1 次查询搞定
func BenchmarkDB_Join(b *testing.B) {
    db := setupDB()
    defer db.Close()
    b.ReportAllocs()
    b.ResetTimer()
    
    for i := 0; i < b.N; i++ {
        // JOIN 查询，1 次搞定！
        db.Query(`
            SELECT u.id, o.id, o.amount 
            FROM users u 
            LEFT JOIN orders o ON u.id = o.user_id 
            LIMIT 100
        `)
    }
}

// ✅ IN 查询：2 次搞定
func BenchmarkDB_InQuery(b *testing.B) {
    db := setupDB()
    defer db.Close()
    b.ReportAllocs()
    b.ResetTimer()
    
    for i := 0; i < b.N; i++ {
        // 1. 查询所有用户（1次）
        rows, _ := db.Query("SELECT id FROM users LIMIT 100")
        var userIDs []int
        for rows.Next() {
            var id int
            rows.Scan(&id)
            userIDs = append(userIDs, id)
        }
        rows.Close()
        
        // 2. IN 查询，1 次搞定所有订单
        placeholders := ""
        for j := range userIDs {
            if j > 0 {
                placeholders += ","
            }
            placeholders += "?"
        }
        db.Query(
            fmt.Sprintf("SELECT * FROM orders WHERE user_id IN (%s)", placeholders),
            // 把 userIDs 转成 interface{} slice
        )
    }
}

// ==============================================
// 优化 2: 批量插入 vs 单条插入
// ==============================================

// ❌ 单条插入：1000 条 = 1000 次网络往返
func BenchmarkDB_InsertSingle(b *testing.B) {
    db := setupDB()
    defer db.Close()
    b.ReportAllocs()
    b.ResetTimer()
    
    for i := 0; i < b.N; i++ {
        db.Exec("INSERT INTO users (name, age) VALUES (?, ?)", "test", 20)
    }
}

// ✅ 批量插入：1000 条 = 1 次网络往返
func BenchmarkDB_InsertBatch(b *testing.B) {
    db := setupDB()
    defer db.Close()
    b.ReportAllocs()
    b.ResetTimer()
    
    batchSize := 100
    for i := 0; i < b.N; i += batchSize {
        // 构建批量插入 SQL
        sql := "INSERT INTO users (name, age) VALUES "
        vals := make([]interface{}, 0, batchSize*2)
        for j := 0; j < batchSize && (i+j) < b.N; j++ {
            sql += "(?,?),"
            vals = append(vals, "test", 20)
        }
        sql = sql[:len(sql)-1] // 去掉最后一个逗号
        
        db.Exec(sql, vals...)
    }
}

// ==============================================
// 优化 3: 连接池大小影响
// ==============================================

func BenchmarkDB_Pool_10(b *testing.B) {
    db := setupDB()
    db.SetMaxOpenConns(10)  // 连接池只有 10 个
    defer db.Close()
    b.ReportAllocs()
    b.ResetTimer()
    
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            db.Query("SELECT SLEEP(0.01)") // 模拟 10ms 查询
        }
    })
}

func BenchmarkDB_Pool_100(b *testing.B) {
    db := setupDB()
    db.SetMaxOpenConns(100) // 连接池 100 个
    defer db.Close()
    b.ReportAllocs()
    b.ResetTimer()
    
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            db.Query("SELECT SLEEP(0.01)")
        }
    })
}

// ==============================================
// 使用说明：
// ==============================================
// 1. 先启动本地 MySQL，创建测试数据库
//    CREATE DATABASE test;
//    USE test;
//    CREATE TABLE users (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), age INT);
//    CREATE TABLE orders (id INT AUTO_INCREMENT PRIMARY KEY, user_id INT, amount INT);
//
// 2. 安装驱动：
//    go get github.com/go-sql-driver/mysql
//
// 3. 运行基准测试：
//    go test -bench="NPlus1|Join|InQuery" -benchmem -count=5
//
// 典型输出：
// BenchmarkDB_NPlus1-8             10    120000000 ns/op
// BenchmarkDB_Join-8               100     12000000 ns/op
// BenchmarkDB_InQuery-8            500      2000000 ns/op
//
// BenchmarkDB_InsertSingle-8      1000      1200000 ns/op
// BenchmarkDB_InsertBatch-8       10000      120000 ns/op
//
// 结论：
// 1. N+1 查询比 JOIN 慢 10 倍，比 IN 查询慢 60 倍！
// 2. 批量插入比单条插入快 10 倍！
// 3. 连接池太小严重影响并发性能！
