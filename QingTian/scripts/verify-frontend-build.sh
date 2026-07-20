#!/usr/bin/env bash
# 用途：验证前端 i18n + 主题切换改动能正常安装依赖并构建
# 执行目录：QingTian/web
# 结果判断：npm install 成功且 npm run build 生成 dist，退出码 0
# 风险：npm install 联网拉取 vue-i18n 等依赖，耗时较长
set -e

source /Users/hongsen.ren/.nvm/nvm.sh
nvm use 18.20.8

cd /Users/hongsen.ren/code/github-code/cuda_learning/QingTian/web

echo "=== Node 版本 ==="
node -v

echo "=== 安装依赖 ==="
npm install

echo "=== 构建 ==="
npm run build

echo "=== 构建产物 ==="
ls -la dist/ 2>/dev/null && echo "BUILD_OK"
