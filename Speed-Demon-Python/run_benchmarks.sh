#!/bin/bash

RED='\033[91m'
GREEN='\033[92m'
YELLOW='\033[93m'
BLUE='\033[94m'
MAGENTA='\033[95m'
CYAN='\033[96m'
WHITE='\033[97m'
RESET='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e ""
echo -e "${MAGENTA}========================================================================${RESET}"
echo -e "${MAGENTA}=${RESET}   ${CYAN}运行所有 Benchmark - 性能测试套件${RESET}                ${MAGENTA}=${RESET}"
echo -e "${MAGENTA}========================================================================${RESET}"
echo -e ""

if [ -d "venv" ]; then
    echo -e "${CYAN}- 激活虚拟环境...${RESET}"
    source venv/bin/activate
fi

echo -e "${BLUE}[1/2]${RESET} ${YELLOW}检查依赖...${RESET}"
pip install pytest-benchmark numpy numba --quiet 2>/dev/null || true
echo -e "${GREEN}OK - 依赖检查完成${RESET}"
echo -e ""

echo -e "${BLUE}[2/2]${RESET} ${YELLOW}开始运行 Benchmark...${RESET}"
echo -e ""

cd 08_benchmarks
python3 run_all_benchmarks.py
EXIT_CODE=$?
cd ..

echo -e ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}========================================================================${RESET}"
    echo -e "${GREEN}=${RESET}   Benchmark 运行完成！"
    echo -e "${GREEN}========================================================================${RESET}"
else
    echo -e "${RED}========================================================================${RESET}"
    echo -e "${RED}=${RESET}   Benchmark 运行完成，但有部分错误"
    echo -e "${RED}========================================================================${RESET}"
fi
echo -e ""
