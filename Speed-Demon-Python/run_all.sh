#!/bin/bash

set -e

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
echo -e "${MAGENTA}=${RESET}   ${CYAN}Speed Demon Python - 一键运行脚本${RESET}              ${MAGENTA}=${RESET}"
echo -e "${MAGENTA}=${RESET}   ${WHITE}Python性能优化实战教程${RESET}                                  ${MAGENTA}=${RESET}"
echo -e "${MAGENTA}========================================================================${RESET}"
echo -e ""

echo -e "${BLUE}[1/6]${RESET} ${YELLOW}检查Python环境...${RESET}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到Python3，请先安装Python 3.8+${RESET}"
    exit 1
fi
PY_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}OK - Python版本: ${PY_VERSION}${RESET}"
echo -e ""

echo -e "${BLUE}[2/6]${RESET} ${YELLOW}检查并安装依赖...${RESET}"
if [ ! -d "venv" ]; then
    echo -e "${CYAN}- 创建虚拟环境...${RESET}"
    python3 -m venv venv
    echo -e "${GREEN}OK - 虚拟环境创建成功${RESET}"
fi

echo -e "${CYAN}- 激活虚拟环境...${RESET}"
source venv/bin/activate

echo -e "${CYAN}- 安装/更新依赖包...${RESET}"
pip install -r requirements.txt --quiet 2>/dev/null || true
echo -e "${GREEN}OK - 依赖安装完成${RESET}"
echo -e ""

show_menu() {
    echo -e "${BLUE}[3/6]${RESET} ${YELLOW}请选择要运行的内容:${RESET}"
    echo -e ""
    echo -e "${CYAN}  1.${RESET} 运行所有 Benchmark (性能测试)"
    echo -e "${CYAN}  2.${RESET} 完整案例对比 (v1_bad vs v2_good)"
    echo -e "${CYAN}  3.${RESET} 代码质量检查"
    echo -e "${CYAN}  4.${RESET} 运行全部 (Benchmark + 案例对比 + 代码检查)"
    echo -e "${CYAN}  5.${RESET} 仅安装依赖并退出"
    echo -e "${CYAN}  0.${RESET} 退出"
    echo -e ""
}

while true; do
    show_menu
    read -p "$(echo -e "${YELLOW}请输入选项 [0-5]: ${RESET}")" CHOICE

    case $CHOICE in
        1)
            echo -e ""
            echo -e "${BLUE}[4/6]${RESET} ${YELLOW}运行所有 Benchmark...${RESET}"
            bash run_benchmarks.sh
            ;;
        2)
            echo -e ""
            echo -e "${BLUE}[4/6]${RESET} ${YELLOW}运行完整案例对比...${RESET}"
            bash run_full_case.sh
            ;;
        3)
            echo -e ""
            echo -e "${BLUE}[4/6]${RESET} ${YELLOW}运行代码质量检查...${RESET}"
            bash run_checks.sh
            ;;
        4)
            echo -e ""
            echo -e "${BLUE}[4/6]${RESET} ${YELLOW}运行全部测试...${RESET}"
            echo -e ""
            echo -e "${CYAN}=== 第一部分: Benchmarks ===${RESET}"
            bash run_benchmarks.sh
            echo -e ""
            echo -e "${CYAN}=== 第二部分: 完整案例对比 ===${RESET}"
            bash run_full_case.sh
            echo -e ""
            echo -e "${CYAN}=== 第三部分: 代码质量检查 ===${RESET}"
            bash run_checks.sh
            break
            ;;
        5)
            echo -e "${GREEN}依赖已安装完成，退出...${RESET}"
            break
            ;;
        0)
            echo -e "${GREEN}退出...${RESET}"
            break
            ;;
        *)
            echo -e "${RED}无效选项，请重新选择${RESET}"
            ;;
    esac

    if [ "$CHOICE" != "4" ] && [ "$CHOICE" != "5" ] && [ "$CHOICE" != "0" ]; then
        echo -e ""
        read -p "$(echo -e "${YELLOW}是否继续运行其他项目? (y/n): ${RESET}")" CONTINUE
        if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
            break
        fi
        clear
    fi
done

echo -e ""
echo -e "${MAGENTA}========================================================================${RESET}"
echo -e "${MAGENTA}=${RESET}   ${GREEN}运行完成！感谢使用 Speed Demon Python${RESET}            ${MAGENTA}=${RESET}"
echo -e "${MAGENTA}========================================================================${RESET}"
echo -e ""
