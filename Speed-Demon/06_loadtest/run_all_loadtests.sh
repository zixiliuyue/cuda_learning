#!/bin/bash

# ============================================================
# 一键运行所有压测对比脚本
# 使用方法: ./run_all_loadtests.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="http://localhost:8080"
OUTPUT_DIR="$SCRIPT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTPUT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

check_service() {
    log_info "检查服务是否启动..."
    for i in {1..30}; do
        if curl -s "$BASE_URL/stats" > /dev/null 2>&1; then
            log_success "服务已启动!"
            return 0
        fi
        sleep 1
    done
    log_error "服务未启动! 请先启动 v1_bad 或 v2_good 服务"
    log_error "启动命令:"
    log_error "  cd ../05_full_case/v1_bad && go run main.go"
    log_error "  或"
    log_error "  cd ../05_full_case/v2_good && go run main.go"
    exit 1
}

run_hey_test() {
    local name=$1
    local requests=$2
    local concurrency=$3
    local output_file="$OUTPUT_DIR/${name}_${TIMESTAMP}.txt"
    
    log_info "运行 $name: $requests 请求, $concurrency 并发..."
    
    if hey -n "$requests" -c "$concurrency" \
        -disable-keepalive=false \
        "$BASE_URL/api/product?id=123" > "$output_file" 2>&1; then
        log_success "$name 完成"
    else
        log_warning "$name 可能有问题，检查输出文件: $output_file"
    fi
}

extract_metrics() {
    local file=$1
    
    qps=$(grep "Requests/sec:" "$file" | awk '{print $2}')
    p50=$(grep "50% in" "$file" | awk '{print $3}')
    p90=$(grep "90% in" "$file" | awk '{print $3}')
    p95=$(grep "95% in" "$file" | awk '{print $3}')
    p99=$(grep "99% in" "$file" | awk '{print $3}')
    slowest=$(grep "Slowest:" "$file" | awk '{print $2}')
    
    echo "$qps|$p50|$p90|$p95|$p99|$slowest"
}

generate_report() {
    local v1_file=$1
    local v2_file=$2
    local report_file="$OUTPUT_DIR/comparison_report_${TIMESTAMP}.md"
    
    log_info "生成对比报告..."
    
    echo "# 性能对比报告" > "$report_file"
    echo "" >> "$report_file"
    echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$report_file"
    echo "" >> "$report_file"
    
    local v1_metrics=$(extract_metrics "$v1_file")
    local v2_metrics=$(extract_metrics "$v2_file")
    
    local v1_qps=$(echo "$v1_metrics" | cut -d'|' -f1)
    local v1_p50=$(echo "$v1_metrics" | cut -d'|' -f2)
    local v1_p90=$(echo "$v1_metrics" | cut -d'|' -f3)
    local v1_p95=$(echo "$v1_metrics" | cut -d'|' -f4)
    local v1_p99=$(echo "$v1_metrics" | cut -d'|' -f5)
    
    local v2_qps=$(echo "$v2_metrics" | cut -d'|' -f1)
    local v2_p50=$(echo "$v2_metrics" | cut -d'|' -f2)
    local v2_p90=$(echo "$v2_metrics" | cut -d'|' -f3)
    local v2_p95=$(echo "$v2_metrics" | cut -d'|' -f4)
    local v2_p99=$(echo "$v2_metrics" | cut -d'|' -f5)
    
    local qps_improvement=$(awk "BEGIN {printf \"%.1f\", ($v2_qps / $v1_qps - 1) * 100}")
    local p99_improvement=$(awk "BEGIN {printf \"%.1f\", ($v1_p99 / $v2_p99 - 1) * 100}")
    
    echo "## 性能对比表" >> "$report_file"
    echo "" >> "$report_file"
    echo "| 指标 | v1_bad | v2_good | 提升 |" >> "$report_file"
    echo "|------|--------|---------|------|" >> "$report_file"
    echo "| QPS | $v1_qps | $v2_qps | **+${qps_improvement}%** |" >> "$report_file"
    echo "| P50 延迟 | $v1_p50 s | $v2_p50 s | |" >> "$report_file"
    echo "| P90 延迟 | $v1_p90 s | $v2_p90 s | |" >> "$report_file"
    echo "| P95 延迟 | $v1_p95 s | $v2_p95 s | |" >> "$report_file"
    echo "| P99 延迟 | $v1_p99 s | $v2_p99 s | **-${p99_improvement}%** |" >> "$report_file"
    echo "" >> "$report_file"
    
    echo "## 优化效果总结" >> "$report_file"
    echo "" >> "$report_file"
    
    if (( $(echo "$qps_improvement > 100" | bc -l) )); then
        echo "- ✅ QPS 提升显著 (> 100%)" >> "$report_file"
    else
        echo "- ⚠️ QPS 提升空间较大" >> "$report_file"
    fi
    
    if (( $(echo "$p99_improvement > 50" | bc -l) )); then
        echo "- ✅ P99 延迟降低显著 (> 50%)" >> "$report_file"
    else
        echo "- ⚠️ P99 延迟降低空间较大" >> "$report_file"
    fi
    
    echo "" >> "$report_file"
    echo "## 原始输出文件" >> "$report_file"
    echo "" >> "$report_file"
    echo "- v1_bad: $(basename "$v1_file")" >> "$report_file"
    echo "- v2_good: $(basename "$v2_file")" >> "$report_file"
    
    log_success "报告已生成: $report_file"
}

# ============================================================
# Main
# ============================================================

clear
echo ""
echo "╔═════════════════════════════════════════╗"
echo "║      Speed Demon 压测对比工具           ║"
echo "╚═════════════════════════════════════════╝"
echo ""

cd "$SCRIPT_DIR"

check_service

print_header "收集当前服务信息"
curl -s "$BASE_URL/stats"
echo ""

print_header "开始压测"
log_info "压测目标: $BASE_URL"
log_info "结果目录: $OUTPUT_DIR"
log_info "时间戳: $TIMESTAMP"

echo ""
read -p "请确认服务版本 (v1_bad/v2_good): " version

if [[ "$version" != "v1_bad" && "$version" != "v2_good" ]]; then
    log_error "无效的版本!"
    exit 1
fi

log_info "将对 $version 进行压测..."

print_header "步骤 1: 预热"
run_hey_test "${version}_warmup" 500 10
sleep 5

print_header "步骤 2: 基准测试"
run_hey_test "${version}_baseline" 2000 50
sleep 10

print_header "步骤 3: 高并发测试"
run_hey_test "${version}_high" 5000 100

print_header "压测完成!"

high_file=$(ls -t "$OUTPUT_DIR/${version}_high"*.txt | head -1)
log_success "高并发结果文件: $high_file"

echo ""
echo "关键指标:"
grep -E "(Requests/sec|P99|P95|P90|P50)" "$high_file"

echo ""
log_info "提示: 你可以切换到另一个版本的服务后再次运行此脚本"
log_info "      然后手动对比两个版本的结果"
echo ""
echo "下一步操作:"
echo "  1. Ctrl+C 停止当前服务"
echo "  2. 启动另一个版本的服务"
echo "  3. 再次运行此脚本"
echo "  4. 对比两个版本的性能差异"
echo ""
