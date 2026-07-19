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
echo -e "${MAGENTA}=${RESET}   ${CYAN}代码质量检查${RESET}                                            ${MAGENTA}=${RESET}"
echo -e "${MAGENTA}========================================================================${RESET}"
echo -e ""

if [ -d "venv" ]; then
    echo -e "${CYAN}- 激活虚拟环境...${RESET}"
    source venv/bin/activate
fi

echo -e "${BLUE}[1/4]${RESET} ${YELLOW}安装检查工具...${RESET}"
pip install flake8 pylint mypy --quiet 2>/dev/null || true
echo -e "${GREEN}OK - 检查工具已安装${RESET}"
echo -e ""

echo -e "${BLUE}[2/4]${RESET} ${YELLOW}运行 flake8 代码风格检查...${RESET}"
echo -e "${WHITE}-------------------------------------------------------------${RESET}"
flake8 --max-line-length=120 --exclude=venv,__pycache__ --ignore=E501,W503 10_full_case 08_benchmarks
if [ $? -eq 0 ]; then
    echo -e "${GREEN}flake8 - 未发现严重的代码风格问题${RESET}"
else
    echo -e "${YELLOW}flake8 - 发现代码风格问题，建议修复${RESET}"
fi
echo -e "${WHITE}-------------------------------------------------------------${RESET}"
echo -e ""

echo -e "${BLUE}[3/4]${RESET} ${YELLOW}运行 pylint 代码质量检查...${RESET}"
echo -e "${WHITE}-------------------------------------------------------------${RESET}"
pylint --disable=C0114,C0116,C0301 10_full_case 08_benchmarks
if [ $? -eq 0 ]; then
    echo -e "${GREEN}pylint - 未发现严重的代码质量问题${RESET}"
else
    echo -e "${YELLOW}pylint - 发现代码质量问题，建议修复${RESET}"
fi
echo -e "${WHITE}-------------------------------------------------------------${RESET}"
echo -e ""

echo -e "${BLUE}[4/4]${RESET} ${YELLOW}运行 mypy 类型检查...${RESET}"
echo -e "${WHITE}-------------------------------------------------------------${RESET}"
mypy --ignore-missing-imports 10_full_case 08_benchmarks
if [ $? -eq 0 ]; then
    echo -e "${GREEN}mypy - 未发现类型错误${RESET}"
else
    echo -e "${YELLOW}mypy - 发现类型错误，建议修复${RESET}"
fi
echo -e "${WHITE}-------------------------------------------------------------${RESET}"
echo -e ""

echo -e "${GREEN}========================================================================${RESET}"
echo -e "${GREEN}=${RESET}   代码质量检查完成！"
echo -e "${GREEN}========================================================================${RESET}"
echo -e ""
