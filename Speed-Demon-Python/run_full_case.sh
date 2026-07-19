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
echo -e "${MAGENTA}=${RESET}   ${CYAN}完整案例对比 - v1_bad vs v2_good${RESET}                   ${MAGENTA}=${RESET}"
echo -e "${MAGENTA}========================================================================${RESET}"
echo -e ""

if [ -d "venv" ]; then
    echo -e "${CYAN}- 激活虚拟环境...${RESET}"
    source venv/bin/activate
fi

echo -e "${BLUE}[1/5]${RESET} ${YELLOW}检查依赖...${RESET}"
pip install flask requests --quiet 2>/dev/null || true
echo -e "${GREEN}OK - 依赖检查完成${RESET}"
echo -e ""

echo -e "${BLUE}[2/5]${RESET} ${YELLOW}启动 v1_bad 服务 (端口 5000)...${RESET}"
cd 10_full_case/v1_bad
python3 app.py > /dev/null 2>&1 &
PID_V1=$!
cd ../../
sleep 3
echo -e "${GREEN}OK - v1_bad 服务已启动 (PID: $PID_V1)${RESET}"
echo -e ""

echo -e "${BLUE}[3/5]${RESET} ${YELLOW}启动 v2_good 服务 (端口 5001)...${RESET}"
cd 10_full_case/v2_good
python3 app.py > /dev/null 2>&1 &
PID_V2=$!
cd ../../
sleep 3
echo -e "${GREEN}OK - v2_good 服务已启动 (PID: $PID_V2)${RESET}"
echo -e ""

echo -e "${BLUE}[4/5]${RESET} ${YELLOW}等待服务就绪...${RESET}"
sleep 2
echo -e "${GREEN}OK - 服务就绪${RESET}"
echo -e ""

echo -e "${BLUE}[5/5]${RESET} ${YELLOW}开始运行压测对比...${RESET}"
echo -e ""
cd 10_full_case
python3 loadtest.py --compare
cd ..

echo -e ""
echo -e "${CYAN}正在停止服务...${RESET}"
kill $PID_V1 $PID_V2 > /dev/null 2>&1 || true
sleep 1

echo -e ""
echo -e "${GREEN}========================================================================${RESET}"
echo -e "${GREEN}=${RESET}   完整案例对比完成！"
echo -e "${GREEN}========================================================================${RESET}"
echo -e ""
