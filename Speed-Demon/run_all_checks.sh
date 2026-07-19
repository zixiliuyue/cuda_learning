#!/bin/bash

set -e

echo "========================================"
echo "   Speed-Demon Code Quality Checks"
echo "========================================"
echo ""

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$PROJECT_ROOT/check_results.log"

echo "Project Root: $PROJECT_ROOT"
echo "Log File: $LOG_FILE"
echo ""

echo "# Code Quality Check Results" > "$LOG_FILE"
echo "Generated at: $(date)" >> "$LOG_FILE"
echo "Go Version: $(go version)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

TOTAL_CHECKS=0
PASSED_CHECKS=0

run_check() {
    local name="$1"
    local cmd="$2"
    local description="$3"
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    echo "----------------------------------------"
    echo "Running: $name"
    echo "Description: $description"
    echo "Command: $cmd"
    echo "----------------------------------------"
    
    echo "## $name" >> "$LOG_FILE"
    echo "$description" >> "$LOG_FILE"
    echo '```' >> "$LOG_FILE"
    
    if eval "$cmd" 2>&1 | tee -a "$LOG_FILE"; then
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        echo "✅ PASSED"
        echo "Status: PASSED ✅" >> "$LOG_FILE"
    else
        echo "❌ FAILED"
        echo "Status: FAILED ❌" >> "$LOG_FILE"
    fi
    
    echo '```' >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    echo ""
}

cd "$PROJECT_ROOT"

run_check "Go Build" "go build ./..." "Check if all packages compile successfully"

run_check "Go Vet" "go vet ./..." "Run static analysis for suspicious constructs"

run_check "Go Fmt" "diff -u <(echo -n) <(gofmt -d ./...)" "Check code formatting (no auto-fix)"

if command -v golint &> /dev/null; then
    run_check "Go Lint" "golint ./... || true" "Run golint (may need: go install golang.org/x/lint/golint@latest)"
else
    echo "golint not found, skipping"
    echo "## Go Lint (SKIPPED)" >> "$LOG_FILE"
    echo "golint not installed. Install with: go install golang.org/x/lint/golint@latest" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
fi

if command -v staticcheck &> /dev/null; then
    run_check "StaticCheck" "staticcheck ./... || true" "Run advanced static analysis (may need: go install honnef.co/go/tools/cmd/staticcheck@latest)"
else
    echo "staticcheck not found, skipping"
    echo "## StaticCheck (SKIPPED)" >> "$LOG_FILE"
    echo "staticcheck not installed. Install with: go install honnef.co/go/tools/cmd/staticcheck@latest" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
fi

run_check "Go Mod Tidy" "go mod tidy && git diff --exit-code go.mod go.sum || true" "Check if go.mod is clean (no missing dependencies)"

echo ""
echo "========================================"
echo "   Check Summary"
echo "========================================"
echo ""
echo "Passed: $PASSED_CHECKS / $TOTAL_CHECKS"
echo ""

if [ "$PASSED_CHECKS" -eq "$TOTAL_CHECKS" ]; then
    echo "🎉 All checks passed!"
else
    echo "⚠️  Some checks failed. See $LOG_FILE for details."
fi

echo ""
echo "Full log: $LOG_FILE"
echo ""

echo "To install missing tools:"
echo "  go install golang.org/x/lint/golint@latest"
echo "  go install honnef.co/go/tools/cmd/staticcheck@latest"
echo ""
