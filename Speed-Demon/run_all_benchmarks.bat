@echo off
setlocal enabledelayedexpansion

echo ========================================
echo    Speed-Demon Benchmark Runner
echo ========================================
echo.

set "PROJECT_ROOT=%~dp0"
set "RESULTS_DIR=%PROJECT_ROOT%benchmark_results"
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"

for /f "tokens=1-6 delims=/: " %%a in ("%date% %time%") do (
    set "TIMESTAMP=%%a%%b%%c_%%d%%e%%f"
)
set "SUMMARY_FILE=%RESULTS_DIR%\summary_!TIMESTAMP!.md"

echo Project Root: %PROJECT_ROOT%
echo Results Dir: %RESULTS_DIR%
echo.

echo # Benchmark Summary > "%SUMMARY_FILE%"
echo Generated at: %date% %time% >> "%SUMMARY_FILE%"
for /f "tokens=*" %%v in ('go version') do echo Go Version: %%v >> "%SUMMARY_FILE%"
echo. >> "%SUMMARY_FILE%"

goto :main

:run_benchmark
set "dir=%~1"
set "name=%~2"

echo ----------------------------------------
echo Running benchmark: %name%
echo Directory: %dir%
echo ----------------------------------------

if exist "%dir%" (
    pushd "%dir%"
    
    set "output_file=%RESULTS_DIR%\%name%_!TIMESTAMP!.txt"
    set "bench_file=%RESULTS_DIR%\%name%_!TIMESTAMP!.bench"
    
    echo ## %name% >> "%SUMMARY_FILE%"
    echo. >> "%SUMMARY_FILE%"
    
    go test -bench=. -benchmem -run=^$ -count=1 > "%output_file%" 2>&1
    if !errorlevel! equ 0 (
        type "%output_file%" | findstr /r "Benchmark PASS FAIL ok" >> "%SUMMARY_FILE%" 2>nul
        
        echo. >> "%SUMMARY_FILE%"
        echo.
        echo [OK] Benchmark completed successfully
    ) else (
        type "%output_file%"
        echo Status: FAILED >> "%SUMMARY_FILE%"
        echo. >> "%SUMMARY_FILE%"
        echo.
        echo [ERROR] Benchmark failed
    )
    
    popd
) else (
    echo [WARNING] Directory not found: %dir%
    echo Status: SKIPPED ^(directory not found^) >> "%SUMMARY_FILE%"
    echo. >> "%SUMMARY_FILE%"
)
echo.
goto :eof

:main
call :run_benchmark "%PROJECT_ROOT%02_practice\06_benchmark" "06_benchmark_basics"
call :run_benchmark "%PROJECT_ROOT%08_concurrent_optimization" "08_concurrent_optimization"
call :run_benchmark "%PROJECT_ROOT%09_business" "09_business_optimization"
call :run_benchmark "%PROJECT_ROOT%11_escape_analysis" "11_escape_analysis"

echo ========================================
echo    Benchmark Summary
echo ========================================
echo.
echo All results saved to: %RESULTS_DIR%
echo Summary report: %SUMMARY_FILE%
echo.

where benchstat >nul 2>nul
if %errorlevel% neq 0 (
    echo To compare results, install benchstat:
    echo   go install golang.org/x/perf/cmd/benchstat@latest
    echo   benchstat old.bench new.bench
)

endlocal
