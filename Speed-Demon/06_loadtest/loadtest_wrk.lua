-- ============================================================
-- wrk 压测 Lua 脚本
-- 使用方法: wrk -t8 -c200 -d30s -s loadtest_wrk.lua http://localhost:8080
-- ============================================================

local counter = 1
local threads = {}

-- 随机 ID 范围
local min_id = 1
local max_id = 10000

-- 请求统计
local request_stats = {
    total = 0,
    success = 0,
    errors = 0
}

function setup(thread)
    thread:set("id", counter)
    table.insert(threads, thread)
    counter = counter + 1
end

function init(args)
    math.randomseed(os.time())
    requests  = 0
    responses = 0
end

function request()
    local product_id = math.random(min_id, max_id)
    local path = "/api/product?id=" .. product_id
    
    requests = requests + 1
    
    return wrk.format("GET", path)
end

function response(status, headers, body)
    responses = responses + 1
    
    if status == 200 then
        request_stats.success = request_stats.success + 1
    else
        request_stats.errors = request_stats.errors + 1
    end
end

function done(summary, latency, requests)
    io.write("\n")
    io.write("========================================\n")
    io.write("       压测结果统计分析\n")
    io.write("========================================\n")
    io.write("\n")
    
    io.write(string.format("总请求数:     %d\n", summary.requests))
    io.write(string.format("总耗时:       %.2f 秒\n", summary.duration / 1000000))
    io.write(string.format("QPS:          %.2f req/sec\n", 
        summary.requests / (summary.duration / 1000000)))
    io.write(string.format("数据传输:     %.2f MB\n", summary.bytes / 1024 / 1024))
    io.write("\n")
    
    io.write("----------------------------------------\n")
    io.write("延迟分布 (ms):\n")
    io.write("----------------------------------------\n")
    
    local percentiles = {50, 75, 90, 95, 99, 99.9}
    for _, p in ipairs(percentiles) do
        local ms = latency:percentile(p) / 1000
        io.write(string.format("  P%-5.1f:    %8.2f ms\n", p, ms))
    end
    
    io.write("\n")
    io.write("----------------------------------------\n")
    io.write("延迟统计:\n")
    io.write("----------------------------------------\n")
    io.write(string.format("  最小值:     %8.2f ms\n", latency.min / 1000))
    io.write(string.format("  最大值:     %8.2f ms\n", latency.max / 1000))
    io.write(string.format("  平均值:     %8.2f ms\n", latency.mean / 1000))
    io.write(string.format("  标准差:     %8.2f ms\n", latency.stdev / 1000))
    
    io.write("\n")
    io.write("----------------------------------------\n")
    io.write("HTTP 状态码:\n")
    io.write("----------------------------------------\n")
    
    for status, count in pairs(summary.status) do
        local status_text = ""
        if status == 200 then
            status_text = "OK"
        elseif status >= 400 and status < 500 then
            status_text = "Client Error"
        elseif status >= 500 then
            status_text = "Server Error"
        end
        io.write(string.format("  %d %s: %d\n", status, status_text, count))
    end
    
    io.write("\n")
    io.write("----------------------------------------\n")
    io.write("错误统计:\n")
    io.write("----------------------------------------\n")
    
    if summary.errors.connect > 0 then
        io.write(string.format("  连接错误:   %d\n", summary.errors.connect))
    end
    if summary.errors.read > 0 then
        io.write(string.format("  读取错误:   %d\n", summary.errors.read))
    end
    if summary.errors.write > 0 then
        io.write(string.format("  写入错误:   %d\n", summary.errors.write))
    end
    if summary.errors.timeout > 0 then
        io.write(string.format("  超时错误:   %d\n", summary.errors.timeout))
    end
    
    local total_errors = summary.errors.connect + summary.errors.read + 
                         summary.errors.write + summary.errors.timeout
    if total_errors == 0 then
        io.write("  无错误 ✓\n")
    end
    
    io.write("\n")
    io.write("========================================\n")
    io.write("\n")
    
    io.write("性能评估建议:\n")
    local p99 = latency:percentile(99) / 1000
    local qps = summary.requests / (summary.duration / 1000000)
    
    if p99 > 500 then
        io.write("  ⚠️  P99 延迟较高 (> 500ms)，检查:\n")
        io.write("     - 是否有全局大锁\n")
        io.write("     - DB/Redis 查询是否优化\n")
        io.write("     - GC 是否频繁\n")
    elseif p99 > 200 then
        io.write("  ⚠️  P99 延迟偏高 (> 200ms)，可进一步优化\n")
    else
        io.write("  ✓  P99 延迟良好 (< 200ms)\n")
    end
    
    if qps < 100 then
        io.write("  ⚠️  QPS 较低 (< 100)，检查是否有阻塞操作\n")
    elseif qps < 500 then
        io.write("  ⚠️  QPS 中等 (< 500)，有优化空间\n")
    else
        io.write("  ✓  QPS 表现良好 (> 500)\n")
    end
    
    io.write("\n")
    io.write("========================================\n")
end
