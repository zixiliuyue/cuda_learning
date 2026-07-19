#!/bin/bash

echo "========================================"
echo "Speed-Demon - Go Performance Debugging"
echo "========================================"
echo ""

echo "[1/4] Checking Go installation..."
if ! command -v go &> /dev/null; then
    echo "Error: Go is not installed or not in PATH"
    exit 1
fi
go version

echo ""
echo "[2/4] Downloading dependencies..."
go mod tidy

echo ""
echo "[3/4] Building examples..."

for dir in 02_practice/*/; do
    if [ -f "$dir/main.go" ]; then
        echo "Building $dir..."
        (cd "$dir" && go build -o demo main.go)
    fi
done

for dir in 03_projects/*/; do
    if [ -f "$dir/main.go" ]; then
        echo "Building $dir..."
        (cd "$dir" && go build -o demo main.go)
    fi
done

echo ""
echo "[4/4] Setup complete!"
echo ""
echo "Usage examples:"
echo "  cd 02_practice/01_cpu_profiling && go run main.go"
echo "  cd 02_practice/02_memory_profiling && go run main.go"
echo "  cd 02_practice/03_goroutine_leak && go run main.go"
echo ""
echo "Access profiles at: http://localhost:6060/debug/pprof/"
echo ""
