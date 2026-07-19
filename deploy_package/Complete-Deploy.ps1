$Password = "123"
$User = "sen"
$HostIP = "192.168.10.82"
$HostKey = "SHA256:d94Qzn+DCcR1gBxnlMqpsNMK9DMds1d5rihg30Orh6k"

$ErrorActionPreference = "Continue"

$plink = Join-Path $env:TEMP "plink.exe"
$pscp = Join-Path $env:TEMP "pscp.exe"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  COMPLETE DEPLOYMENT PROCESS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Step 1/4: Uploading docker-compose v2 install script..." -ForegroundColor Yellow
& $pscp -pw $Password -batch -hostkey $HostKey "f:\code\cuda_learning\deploy_package\install_compose_v2.sh" "${User}@${HostIP}:/tmp/"
Write-Host "[OK] Script uploaded" -ForegroundColor Green
Write-Host ""

Write-Host "Step 2/4: Installing Docker Compose V2..." -ForegroundColor Yellow
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "chmod +x /tmp/install_compose_v2.sh && /tmp/install_compose_v2.sh"
Write-Host "[OK] Docker Compose V2 installed" -ForegroundColor Green
Write-Host ""

Write-Host "Step 3/4: Uploading and running deploy script..." -ForegroundColor Yellow
& $pscp -pw $Password -batch -hostkey $HostKey "f:\code\cuda_learning\deploy_package\deploy_services.sh" "${User}@${HostIP}:/tmp/"
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "chmod +x /tmp/deploy_services.sh && /tmp/deploy_services.sh"
Write-Host "[OK] Deployment completed" -ForegroundColor Green
Write-Host ""

Write-Host "Step 4/4: Cleaning up..." -ForegroundColor Yellow
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "rm -f /tmp/install_compose_v2.sh /tmp/deploy_services.sh"
Write-Host "[OK] Cleanup completed" -ForegroundColor Green
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ALL DONE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Frontend URL: http://192.168.10.82:80" -ForegroundColor Green
Write-Host ""
