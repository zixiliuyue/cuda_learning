#!/usr/bin/env bash
# check-secrets.sh —— 轻量级敏感信息预提交扫描脚本（零外部依赖）
#
# 用途：在提交前扫描暂存区改动，拦截常见的硬编码敏感信息，
#       防止 API 密钥、密码、内网地址、私钥等误入仓库。
#
# 安装为 git pre-commit 钩子：
#   ln -sf ../../scripts/check-secrets.sh .git/hooks/pre-commit
#   chmod +x .git/hooks/pre-commit
#
# 手动全量扫描：
#   bash scripts/check-secrets.sh --all
#
# 退出码：发现疑似敏感信息返回 1，否则返回 0。

set -euo pipefail

# 敏感信息正则规则（按需扩展）。使用 ERE。
PATTERNS=(
  # 私钥
  '-----BEGIN (RSA|OPENSSH|EC|DSA|PGP) PRIVATE KEY-----'
  # 硬编码密码/密钥赋值（排除占位符 change-me / <...> / env 引用 / 空值）
  '(password|passwd|secret|api[_-]?key|access[_-]?key|token)[[:space:]]*[:=][[:space:]]*["'"'"']?[A-Za-z0-9/+_-]{6,}'
  # 私有网段 IP（RFC1918）
  '\b(10\.[0-9]{1,3}|192\.168|172\.(1[6-9]|2[0-9]|3[01]))\.[0-9]{1,3}\.[0-9]{1,3}\b'
  # AWS Access Key ID
  'AKIA[0-9A-Z]{16}'
)

# 允许的占位符/示例白名单（命中这些则不算敏感）。
ALLOWLIST='change-me|<[A-Z_]+>|\$\{[A-Z_]+|example\.com|minioadmin|127\.0\.0\.1|0\.0\.0\.0|192\.168\.1\.100|your-|placeholder|dummy|test'

scan_text() {
  local text="$1" file="$2" found=0
  for pat in "${PATTERNS[@]}"; do
    while IFS= read -r line; do
      [ -z "$line" ] && continue
      # 命中白名单则跳过
      if echo "$line" | grep -Eiq "$ALLOWLIST"; then continue; fi
      echo "  [疑似敏感] $file: $line"
      found=1
    done < <(echo "$text" | grep -Eino "$pat" || true)
  done
  return $found
}

RC=0
if [ "${1:-}" = "--all" ]; then
  # 全量扫描已跟踪的文本文件
  while IFS= read -r f; do
    [ -f "$f" ] || continue
    content=$(cat "$f" 2>/dev/null || true)
    scan_text "$content" "$f" || RC=1
  done < <(git ls-files)
else
  # 仅扫描暂存区改动
  while IFS= read -r f; do
    [ -f "$f" ] || continue
    content=$(git show ":$f" 2>/dev/null || true)
    scan_text "$content" "$f" || RC=1
  done < <(git diff --cached --name-only --diff-filter=ACM)
fi

if [ "$RC" -ne 0 ]; then
  echo ""
  echo "[check-secrets] 检测到疑似敏感信息，提交被拦截。"
  echo "  - 若为误报，请将占位符加入 ALLOWLIST 或改用环境变量注入。"
  echo "  - 确认无误后可用 'git commit --no-verify' 跳过（不推荐）。"
fi
exit $RC
