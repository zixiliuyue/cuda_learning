@echo off
echo ========================================
echo Speed-Demon - Go Performance Debugging
echo ========================================
echo.

echo [1/4] Checking Go installation...
go version
if %errorlevel% neq 0 (
    echo Error: Go is not installed or not in PATH
    exit /b 1
)

echo.
echo [2/4] Downloading dependencies...
go mod tidy

echo.
echo [3/4] Building examples...

setlocal enabledelayedexpansion

for /d %%d in (02_practice\*) do (
    if exist "%%d\main.go" (
        echo Building %%d...
        cd %%d
        go build -o demo.exe main.go
        cd ..\..
    )
)

for /d %%d in (03_projects\*) do (
    if exist "%%d\main.go" (
        echo Building %%d...
        cd %%d
        go build -o demo.exe main.go
        cd ..\..
    )
)

echo.
echo [4/4] Setup complete!
echo.
echo Usage examples:
echo   cd 02_practice\01_cpu_profiling ^&^& go run main.go
echo   cd 02_practice\02_memory_profiling ^&^& go run main.go
echo   cd 02_practice\03_goroutine_leak ^&^& go run main.go
echo.
echo Access profiles at: http://localhost:6060/debug/pprof/
echo.
pause
