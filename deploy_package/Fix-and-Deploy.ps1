$Password = "123"
$User = "sen"
$HostIP = "192.168.10.82"
$HostKey = "SHA256:d94Qzn+DCcR1gBxnlMqpsNMK9DMds1d5rihg30Orh6k"

$ErrorActionPreference = "Continue"

$plink = Join-Path $env:TEMP "plink.exe"
$pscp = Join-Path $env:TEMP "pscp.exe"

Write-Host "Step 1: Uploading fix script..." -ForegroundColor Yellow
& $pscp -pw $Password -batch -hostkey $HostKey "f:\code\cuda_learning\deploy_package\fix_python_distutils.sh" "${User}@${HostIP}:/tmp/"
Write-Host "[OK] Fix script uploaded" -ForegroundColor Green
Write-Host ""

Write-Host "Step 2: Running fix script..." -ForegroundColor Yellow
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "chmod +x /tmp/fix_python_distutils.sh && /tmp/fix_python_distutils.sh"
Write-Host "[OK] Fix completed" -ForegroundColor Green
Write-Host ""

Write-Host "Step 3: Uploading deploy script..." -ForegroundColor Yellow
& $pscp -pw $Password -batch -hostkey $HostKey "f:\code\cuda_learning\deploy_package\deploy_services.sh" "${User}@${HostIP}:/tmp/"
Write-Host "[OK] Deploy script uploaded" -ForegroundColor Green
Write-Host ""

Write-Host "Step 4: Running deploy script..." -ForegroundColor Yellow
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "chmod +x /tmp/deploy_services.sh && /tmp/deploy_services.sh"
Write-Host "[OK] Deploy completed" -ForegroundColor Green
Write-Host ""

Write-Host "Cleaning up..." -ForegroundColor Yellow
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "rm -f /tmp/fix_python_distutils.sh /tmp/deploy_services.sh"
Write-Host "[OK] Cleanup completed" -ForegroundColor Green
Write-Host ""
