#!/bin/bash

# ============================================================
# hey 压测脚本
# 使用方法: ./loadtest_hey.sh <目标>
# 目标: v1_bad | v2_good
# ============================================================

set -e

TARGET=$1
BASE_URL="http://localhost:8080"
OUTPUT_DIR="./results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTPUT_DIR"

if [ -z "$TARGET" ]; then
    echo "用法: $0 <目标>"
    echo "目标: v1_bad | v2_good"
    exit 1
fi

if ! command -v hey &> /dev/null; then
    echo "错误: hey 未安装"
    echo "安装方法: go install github.com/rakyll/hey@latest"
    exit 1
fi

echo "=========================================="
echo "开始压测: $TARGET"
echo "目标地址: $BASE_URL"
echo "时间: $TIMESTAMP"
echo "=========================================="

run_hey_test() {
    local name=$1
    local requests=$2
    local concurrency=$3
    local output_file="$OUTPUT_DIR/${TARGET}_${name}_${TIMESTAMP}.txt"
    
    echo ""
    echo "测试: $name ($requests 请求, $concurrency 并发)"
    echo "输出文件: $output_file"
    
    hey -n "$requests" -c "$concurrency" \
        -o csv \
        "$BASE_URL/api/product?id=123" > "$output_file.csv" 2>/dev/null
    
    hey -n "$requests" -c "$concurrency" \
        "$BASE_URL/api/product?id=123" > "$output_file"
    
    echo "完成!"
}

echo ""
echo "=========================================="
echo "步骤 1: 预热 (低并发)"
echo "=========================================="
run_hey_test "warmup" 500 10

sleep 5

echo ""
echo "=========================================="
echo "步骤 2: 基准测试 (中等并发)"
echo "=========================================="
run_hey_test "baseline" 2000 50

sleep 10

echo ""
echo "=========================================="
echo "步骤 3: 高并发测试"
echo "=========================================="
run_hey_test "high_concurrency" 5000 100

echo ""
echo "=========================================="
echo "压测完成! 结果保存在: $OUTPUT_DIR"
echo "=========================================="

echo ""
echo "最新结果摘要:"
latest_file=$(ls -t "$OUTPUT_DIR/${TARGET}_high_concurrency"*.txt | head -1)
if [ -f "$latest_file" ]; then
    grep -E "(Requests/sec|Summary|Slowest|Fastest|Average|P99|P95|P90|P50)" "$latest_file" | head -15
fi

echo ""
echo "提示: 可以对比 v1_bad 和 v2_good 的结果差异!"
