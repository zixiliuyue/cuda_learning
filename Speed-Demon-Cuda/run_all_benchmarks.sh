#!/bin/bash
echo "========================================"
echo "  Speed-Demon-Cuda 基准测试 (Linux/Mac)"
echo "========================================"
echo ""

# 检查是否已构建
if [ ! -d "build" ]; then
    echo "错误：请先运行 build.sh 构建项目！"
    exit 1
fi

# 创建结果目录
mkdir -p benchmark_results

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_FILE="benchmark_results/result_${TIMESTAMP}.txt"

echo "开始运行所有基准测试..."
echo "结果将保存到: ${RESULT_FILE}"
echo "" > ${RESULT_FILE}

# 运行08_benchmarks目录下的所有可执行文件
echo "[1/1] 运行基准测试..."
echo ""
for f in 08_benchmarks/*; do
    if [ -x "$f" ] && [ ! -d "$f" ]; then
        filename=$(basename "$f")
        echo "正在运行: ${filename}"
        echo "=== ${filename} ===" >> ${RESULT_FILE}
        $f >> ${RESULT_FILE} 2>&1
        echo "" >> ${RESULT_FILE}
    fi
done

echo ""
echo "========================================"
echo "  所有基准测试完成！✅"
echo "  结果已保存到: ${RESULT_FILE}"
echo "========================================"
