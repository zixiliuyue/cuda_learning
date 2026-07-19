#!/bin/bash

set -e

echo "========================================"
echo "   Speed-Demon Full Demo Script"
echo "========================================"
echo ""

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$PROJECT_ROOT/03_projects/01_web_service"
DEMO_PID=""

cleanup() {
    echo ""
    echo "----------------------------------------"
    echo "Cleaning up..."
    if [ -n "$DEMO_PID" ]; then
        echo "Stopping web service (PID: $DEMO_PID)"
        kill $DEMO_PID 2>/dev/null || true
        wait $DEMO_PID 2>/dev/null || true
    fi
    echo "Cleanup complete"
}

trap cleanup EXIT INT TERM

cd "$PROJECT_ROOT"

echo "Step 1: Running code quality checks..."
echo "----------------------------------------"
if command -v bash &> /dev/null && [ -f "./run_all_checks.sh" ]; then
    bash ./run_all_checks.sh || echo "Some checks failed, continuing anyway..."
fi
echo ""

echo "Step 2: Building web service..."
echo "----------------------------------------"
cd "$DEMO_DIR"
go build -o demo_service main.go
echo "✅ Build successful"
echo ""

echo "Step 3: Starting web service on port 8080..."
echo "----------------------------------------"
./demo_service &
DEMO_PID=$!
echo "Service started with PID: $DEMO_PID"
echo "Waiting for service to be ready..."
sleep 3

if ps -p $DEMO_PID > /dev/null; then
    echo "✅ Service is running"
else
    echo "❌ Service failed to start"
    exit 1
fi
echo ""

echo "Step 4: Testing API endpoints..."
echo "----------------------------------------"
if command -v curl &> /dev/null; then
    echo "Testing GET /health..."
    curl -s http://localhost:8080/health || echo "⚠️  Health check failed"
    echo ""
    
    echo "Testing GET /metrics..."
    curl -s http://localhost:8080/metrics || echo "⚠️  Metrics check failed"
    echo ""
    
    echo "Testing POST /api/data..."
    curl -s -X POST http://localhost:8080/api/data -H "Content-Type: application/json" -d '{"test": "data"}' || echo "⚠️  Data endpoint failed"
    echo ""
fi
echo ""

echo "Step 5: Starting load test (if available)..."
echo "----------------------------------------"
LOAD_TEST_DIR="$PROJECT_ROOT/06_loadtest"
if [ -d "$LOAD_TEST_DIR" ]; then
    echo "Load test directory found"
    if command -v ab &> /dev/null; then
        echo "Running Apache Bench..."
        ab -n 1000 -c 50 http://localhost:8080/health 2>/dev/null || echo "⚠️  Apache Bench failed"
    elif command -v wrk &> /dev/null; then
        echo "Running wrk..."
        wrk -t4 -c100 -d10s http://localhost:8080/health 2>/dev/null || echo "⚠️  wrk failed"
    else
        echo "No load testing tool found (ab or wrk)"
        echo "Install one of them for load testing:"
        echo "  Ubuntu/Debian: sudo apt install apache2-utils"
        echo "  macOS: brew install wrk"
    fi
else
    echo "Load test directory not found, skipping"
fi
echo ""

echo "Step 6: Collecting pprof profiles..."
echo "----------------------------------------"
if command -v curl &> /dev/null; then
    echo "Collecting CPU profile..."
    curl -s "http://localhost:8080/debug/pprof/profile?seconds=5" > /dev/null 2>&1 && echo "✅ CPU profile collected" || echo "⚠️  CPU profile endpoint not available"
    
    echo "Collecting heap profile..."
    curl -s "http://localhost:8080/debug/pprof/heap" > /dev/null 2>&1 && echo "✅ Heap profile collected" || echo "⚠️  Heap profile endpoint not available"
    
    echo "Collecting goroutine profile..."
    curl -s "http://localhost:8080/debug/pprof/goroutine" > /dev/null 2>&1 && echo "✅ Goroutine profile collected" || echo "⚠️  Goroutine profile endpoint not available"
fi
echo ""

echo "Step 7: Running benchmarks..."
echo "----------------------------------------"
cd "$PROJECT_ROOT"
if [ -f "./run_all_benchmarks.sh" ]; then
    bash ./run_all_benchmarks.sh || echo "Some benchmarks failed"
fi
echo ""

echo "========================================"
echo "   Demo Complete!"
echo "========================================"
echo ""
echo "What was demonstrated:"
echo "  1. Code quality checks (go build, go vet, gofmt)"
echo "  2. Web service startup and API testing"
echo "  3. Load testing (if tools available)"
echo "  4. Pprof profiling (CPU, heap, goroutine)"
echo "  5. Benchmark execution"
echo ""
echo "Useful commands:"
echo "  View pprof UI: go tool pprof -http=:8081 http://localhost:8080/debug/pprof/heap"
echo "  View trace:    go tool trace http://localhost:8080/debug/pprof/trace?seconds=10"
echo ""
echo "Press Ctrl+C to stop the web service"

wait
