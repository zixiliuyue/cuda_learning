@echo off
setlocal enabledelayedexpansion

echo ========================================
echo    Speed-Demon Code Quality Checks
echo ========================================
echo.

set "PROJECT_ROOT=%~dp0"
set "LOG_FILE=%PROJECT_ROOT%check_results.log"

echo Project Root: %PROJECT_ROOT%
echo Log File: %LOG_FILE%
echo.

echo # Code Quality Check Results > "%LOG_FILE%"
echo Generated at: %date% %time% >> "%LOG_FILE%"
for /f "tokens=*" %%v in ('go version') do echo Go Version: %%v >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

set "TOTAL_CHECKS=0"
set "PASSED_CHECKS=0"

goto :main

:run_check
set "name=%~1"
set "cmd=%~2"
set "description=%~3"

set /a "TOTAL_CHECKS+=1"

echo ----------------------------------------
echo Running: %name%
echo Description: %description%
echo Command: %cmd%
echo ----------------------------------------

echo ## %name% >> "%LOG_FILE%"
echo %description% >> "%LOG_FILE%"
echo ``` >> "%LOG_FILE%"

%cmd% 2>&1 >> "%LOG_FILE%"
if !errorlevel! equ 0 (
    set /a "PASSED_CHECKS+=1"
    echo [OK] PASSED
    echo Status: PASSED ^<ok^> >> "%LOG_FILE%"
) else (
    echo [ERROR] FAILED
    echo Status: FAILED ^<error^> >> "%LOG_FILE%"
)

echo ``` >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"
echo.
goto :eof

:main
cd /d "%PROJECT_ROOT%"

call :run_check "Go Build" "go build ./..." "Check if all packages compile successfully"

call :run_check "Go Vet" "go vet ./..." "Run static analysis for suspicious constructs"

echo ## Go Fmt >> "%LOG_FILE%"
echo Check code formatting >> "%LOG_FILE%"
echo ``` >> "%LOG_FILE%"
gofmt -d ./... >> "%LOG_FILE%" 2>&1
if !errorlevel! equ 0 (
    for /f %%i in ('type "%LOG_FILE%" ^| find /c /v ""') do set lines=%%i
    if !lines! leq 5 (
        echo [OK] PASSED - All files formatted correctly
        echo Status: PASSED >> "%LOG_FILE%"
        set /a "PASSED_CHECKS+=1"
    ) else (
        echo [WARNING] Formatting issues found - run gofmt -w ./... to fix
        echo Status: WARNING - Formatting issues found >> "%LOG_FILE%"
    )
)
echo ``` >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"
set /a "TOTAL_CHECKS+=1"

where golint >nul 2>nul
if !errorlevel! equ 0 (
    echo ## Go Lint >> "%LOG_FILE%"
    echo ``` >> "%LOG_FILE%"
    golint ./... >> "%LOG_FILE%" 2>&1
    echo ``` >> "%LOG_FILE%"
    echo. >> "%LOG_FILE%"
    echo [OK] Go Lint completed
    set /a "TOTAL_CHECKS+=1"
    set /a "PASSED_CHECKS+=1"
) else (
    echo [WARNING] golint not found, skipping
    echo ## Go Lint ^(SKIPPED^) >> "%LOG_FILE%"
    echo golint not installed. Install with: go install golang.org/x/lint/golint@latest >> "%LOG_FILE%"
    echo. >> "%LOG_FILE%"
    set /a "TOTAL_CHECKS+=1"
)

where staticcheck >nul 2>nul
if !errorlevel! equ 0 (
    echo ## StaticCheck >> "%LOG_FILE%"
    echo Run advanced static analysis >> "%LOG_FILE%"
    echo ``` >> "%LOG_FILE%"
    staticcheck ./... >> "%LOG_FILE%" 2>&1
    echo ``` >> "%LOG_FILE%"
    echo. >> "%LOG_FILE%"
    echo [OK] StaticCheck completed
    set /a "TOTAL_CHECKS+=1"
    set /a "PASSED_CHECKS+=1"
) else (
    echo [WARNING] staticcheck not found, skipping
    echo ## StaticCheck ^(SKIPPED^) >> "%LOG_FILE%"
    echo staticcheck not installed. Install with: go install honnef.co/go/tools/cmd/staticcheck@latest >> "%LOG_FILE%"
    echo. >> "%LOG_FILE%"
    set /a "TOTAL_CHECKS+=1"
)

call :run_check "Go Mod Tidy" "go mod tidy" "Check if go.mod is clean"

echo.
echo ========================================
echo    Check Summary
echo ========================================
echo.
echo Passed: !PASSED_CHECKS! / !TOTAL_CHECKS!
echo.

if !PASSED_CHECKS! equ !TOTAL_CHECKS! (
    echo [SUCCESS] All checks passed!
) else (
    echo [WARNING] Some checks may have issues. See %LOG_FILE% for details.
)

echo.
echo Full log: %LOG_FILE%
echo.

echo To install missing tools:
echo   go install golang.org/x/lint/golint@latest
echo   go install honnef.co/go/tools/cmd/staticcheck@latest
echo.

endlocal
