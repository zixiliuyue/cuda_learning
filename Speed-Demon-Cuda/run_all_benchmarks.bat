@echo off
echo ========================================
echo   Speed-Demon-Cuda 基准测试 (Windows)
echo ========================================
echo.

REM 检查是否已构建
if not exist build (
    echo 错误：请先运行 build.bat 构建项目！
    pause
    exit /b 1
)

REM 创建结果目录
if not exist benchmark_results mkdir benchmark_results

set TIMESTAMP=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set RESULT_FILE=benchmark_results\result_%TIMESTAMP%.txt

echo 开始运行所有基准测试...
echo 结果将保存到: %RESULT_FILE%
echo. > %RESULT_FILE%

REM 运行08_benchmarks目录下的所有可执行文件
echo [1/1] 运行基准测试...
echo.
if exist 08_benchmarks\Release (
    for %%f in (08_benchmarks\Release\*.exe) do (
        echo 正在运行: %%~nf.exe
        echo === %%~nf === >> %RESULT_FILE%
        %%f >> %RESULT_FILE% 2>&1
        echo. >> %RESULT_FILE%
    )
) else (
    for %%f in (08_benchmarks\*.exe) do (
        echo 正在运行: %%~nf.exe
        echo === %%~nf === >> %RESULT_FILE%
        %%f >> %RESULT_FILE% 2>&1
        echo. >> %RESULT_FILE%
    )
)

echo.
echo ========================================
echo   所有基准测试完成！✅
echo   结果已保存到: %RESULT_FILE%
echo ========================================
echo.
pause
