@echo off
setlocal enabledelayedexpansion

echo ========================================
echo    Speed-Demon Full Demo Script
echo ========================================
echo.

set "PROJECT_ROOT=%~dp0"
set "DEMO_DIR=%PROJECT_ROOT%03_projects\01_web_service"
set "DEMO_PID="

cd /d "%PROJECT_ROOT%"

echo Step 1: Running code quality checks...
echo ----------------------------------------
if exist "%PROJECT_ROOT%run_all_checks.bat" (
    call "%PROJECT_ROOT%run_all_checks.bat"
) else (
    echo Running go build ./...
    go build ./...
    echo Running go vet ./...
    go vet ./...
)
echo.

echo Step 2: Building web service...
echo ----------------------------------------
cd /d "%DEMO_DIR%"
go build -o demo_service.exe main.go
if !errorlevel! equ 0 (
    echo [OK] Build successful
) else (
    echo [ERROR] Build failed
    goto :end
)
echo.

echo Step 3: Starting web service on port 8080...
echo ----------------------------------------
start /B cmd /C "cd /d %DEMO_DIR% && demo_service.exe"
timeout /t 3 /nobreak >nul

tasklist /FI "IMAGENAME eq demo_service.exe" 2>NUL | find /I /N "demo_service.exe">NUL
if !errorlevel! equ 0 (
    echo [OK] Service is running
    for /f "tokens=2 delims=," %%a in ('tasklist /FI "IMAGENAME eq demo_service.exe" /FO CSV /NH') do (
        set "DEMO_PID=%%a"
        set "DEMO_PID=!DEMO_PID:~1,-1!"
    )
    echo Service PID: !DEMO_PID!
) else (
    echo [ERROR] Service failed to start
    goto :end
)
echo.

echo Step 4: Testing API endpoints...
echo ----------------------------------------
where curl >nul 2>nul
if !errorlevel! equ 0 (
    echo Testing GET /health...
    curl -s http://localhost:8080/health
    echo.
    
    echo Testing GET /metrics...
    curl -s http://localhost:8080/metrics
    echo.
    
    echo Testing POST /api/data...
    curl -s -X POST http://localhost:8080/api/data -H "Content-Type: application/json" -d "{\"test\": \"data\"}"
    echo.
) else (
    echo curl not found, skipping API tests
)
echo.

echo Step 5: Collecting pprof profiles...
echo ----------------------------------------
where curl >nul 2>nul
if !errorlevel! equ 0 (
    echo Collecting CPU profile ^(5 seconds^)...
    curl -s "http://localhost:8080/debug/pprof/profile?seconds=5" > NUL
    if !errorlevel! equ 0 (
        echo [OK] CPU profile collected
    ) else (
        echo [WARNING] CPU profile endpoint not available
    )
    
    echo Collecting heap profile...
    curl -s "http://localhost:8080/debug/pprof/heap" > NUL
    if !errorlevel! equ 0 (
        echo [OK] Heap profile collected
    ) else (
        echo [WARNING] Heap profile endpoint not available
    )
    
    echo Collecting goroutine profile...
    curl -s "http://localhost:8080/debug/pprof/goroutine" > NUL
    if !errorlevel! equ 0 (
        echo [OK] Goroutine profile collected
    ) else (
        echo [WARNING] Goroutine profile endpoint not available
    )
) else (
    echo curl not found, skipping pprof collection
)
echo.

echo Step 6: Running benchmarks...
echo ----------------------------------------
cd /d "%PROJECT_ROOT%"
if exist "%PROJECT_ROOT%run_all_benchmarks.bat" (
    call "%PROJECT_ROOT%run_all_benchmarks.bat"
) else (
    echo Benchmark script not found
)
echo.

echo ========================================
echo    Demo Complete!
echo ========================================
echo.
echo What was demonstrated:
echo   1. Code quality checks ^(go build, go vet, gofmt^)
echo   2. Web service startup and API testing
echo   3. Pprof profiling ^(CPU, heap, goroutine^)
echo   4. Benchmark execution
echo.
echo Useful commands:
echo   View pprof UI: go tool pprof -http=:8081 http://localhost:8080/debug/pprof/heap
echo   View trace:    go tool trace http://localhost:8080/debug/pprof/trace?seconds=10
echo.
echo The web service is still running in background
echo To stop it, run: taskkill /F /IM demo_service.exe
echo.

:end
endlocal
