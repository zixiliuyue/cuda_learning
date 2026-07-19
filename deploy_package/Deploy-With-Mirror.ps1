$Password = "123"
$User = "sen"
$HostIP = "192.168.10.82"
$HostKey = "SHA256:d94Qzn+DCcR1gBxnlMqpsNMK9DMds1d5rihg30Orh6k"

$ErrorActionPreference = "Continue"

$plink = Join-Path $env:TEMP "plink.exe"
$pscp = Join-Path $env:TEMP "pscp.exe"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DEPLOYMENT WITH DOCKER MIRROR" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Step 1/4: Configuring Docker mirror..." -ForegroundColor Yellow
& $pscp -pw $Password -batch -hostkey $HostKey "f:\code\cuda_learning\deploy_package\config_docker_mirror.sh" "${User}@${HostIP}:/tmp/"
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "chmod +x /tmp/config_docker_mirror.sh && /tmp/config_docker_mirror.sh"
Write-Host "[OK] Docker mirror configured" -ForegroundColor Green
Write-Host ""

Write-Host "Step 2/4: Waiting for Docker to be ready (10 seconds)..."
Start-Sleep -Seconds 10
Write-Host "[OK] Docker ready" -ForegroundColor Green
Write-Host ""

Write-Host "Step 3/4: Running deployment script..." -ForegroundColor Yellow
& $pscp -pw $Password -batch -hostkey $HostKey "f:\code\cuda_learning\deploy_package\deploy_services.sh" "${User}@${HostIP}:/tmp/"
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "chmod +x /tmp/deploy_services.sh && /tmp/deploy_services.sh"
Write-Host "[OK] Deployment completed" -ForegroundColor Green
Write-Host ""

Write-Host "Step 4/4: Cleaning up..." -ForegroundColor Yellow
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "rm -f /tmp/config_docker_mirror.sh /tmp/deploy_services.sh"
Write-Host "[OK] Cleanup completed" -ForegroundColor Green
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DEPLOYMENT COMPLETED!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Frontend URL: http://192.168.10.82:80" -ForegroundColor Green
Write-Host ""
