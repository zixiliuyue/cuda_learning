#!/bin/bash

set -e

echo "========================================"
echo "   Speed-Demon Benchmark Runner"
echo "========================================"
echo ""

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$PROJECT_ROOT/benchmark_results"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_FILE="$RESULTS_DIR/summary_$TIMESTAMP.md"

echo "Project Root: $PROJECT_ROOT"
echo "Results Dir: $RESULTS_DIR"
echo ""

echo "# Benchmark Summary" > "$SUMMARY_FILE"
echo "Generated at: $(date)" >> "$SUMMARY_FILE"
echo "Go Version: $(go version)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

run_benchmark() {
    local dir="$1"
    local name="$2"
    
    echo "----------------------------------------"
    echo "Running benchmark: $name"
    echo "Directory: $dir"
    echo "----------------------------------------"
    
    if [ -d "$dir" ]; then
        cd "$dir"
        
        local output_file="$RESULTS_DIR/${name}_$TIMESTAMP.txt"
        local bench_file="$RESULTS_DIR/${name}_$TIMESTAMP.bench"
        
        echo "## $name" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
        
        if go test -bench=. -benchmem -run=^$ -count=1 2>&1 | tee "$output_file"; then
            go test -bench=. -benchmem -run=^$ -count=1 > "$bench_file" 2>/dev/null || true
            
            echo "Results:" >> "$SUMMARY_FILE"
            echo '```' >> "$SUMMARY_FILE"
            grep -E "(Benchmark|PASS|FAIL|ok)" "$output_file" | tail -n +2 >> "$SUMMARY_FILE" 2>/dev/null || echo "No benchmark results found" >> "$SUMMARY_FILE"
            echo '```' >> "$SUMMARY_FILE"
            echo "" >> "$SUMMARY_FILE"
            
            echo "✅ Benchmark completed successfully"
        else
            echo "❌ Benchmark failed"
            echo "Status: FAILED" >> "$SUMMARY_FILE"
            echo "" >> "$SUMMARY_FILE"
        fi
        
        cd "$PROJECT_ROOT"
    else
        echo "⚠️  Directory not found: $dir"
        echo "Status: SKIPPED (directory not found)" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
    fi
    echo ""
}

run_benchmark "$PROJECT_ROOT/02_practice/06_benchmark" "06_benchmark_basics"
run_benchmark "$PROJECT_ROOT/08_concurrent_optimization" "08_concurrent_optimization"
run_benchmark "$PROJECT_ROOT/09_business" "09_business_optimization"
run_benchmark "$PROJECT_ROOT/11_escape_analysis" "11_escape_analysis"

echo "========================================"
echo "   Benchmark Summary"
echo "========================================"
echo ""
echo "All results saved to: $RESULTS_DIR"
echo "Summary report: $SUMMARY_FILE"
echo ""

if command -v gofmt &> /dev/null; then
    if command -v benchstat &> /dev/null; then
        echo "To compare results, install benchstat:"
        echo "  go install golang.org/x/perf/cmd/benchstat@latest"
        echo "  benchstat old.bench new.bench"
    fi
fi
