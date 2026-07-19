Write-Host "正在下载 plink.exe (PuTTY 工具)..." -ForegroundColor Yellow

$plinkPath = Join-Path $env:TEMP "plink.exe"
$pscpPath = Join-Path $env:TEMP "pscp.exe"

if (-not (Test-Path $plinkPath)) {
    Invoke-WebRequest -Uri "https://the.earth.li/~sgtatham/putty/latest/w64/plink.exe" -OutFile $plinkPath
    Write-Host "✓ plink.exe 下载完成" -ForegroundColor Green
} else {
    Write-Host "✓ plink.exe 已存在" -ForegroundColor Green
}

if (-not (Test-Path $pscpPath)) {
    Invoke-WebRequest -Uri "https://the.earth.li/~sgtatham/putty/latest/w64/pscp.exe" -OutFile $pscpPath
    Write-Host "✓ pscp.exe 下载完成" -ForegroundColor Green
} else {
    Write-Host "✓ pscp.exe 已存在" -ForegroundColor Green
}

Write-Host ""
Write-Host "工具已下载到: $env:TEMP" -ForegroundColor Cyan
Write-Host ""
