$Password = "123"
$User = "sen"
$HostIP = "192.168.10.82"
$HostKey = "SHA256:d94Qzn+DCcR1gBxnlMqpsNMK9DMds1d5rihg30Orh6k"

$ErrorActionPreference = "Continue"

$plink = Join-Path $env:TEMP "plink.exe"
$pscp = Join-Path $env:TEMP "pscp.exe"

Write-Host "Uploading deploy_services.sh to remote server..." -ForegroundColor Yellow
& $pscp -pw $Password -batch -hostkey $HostKey "f:\code\cuda_learning\deploy_package\deploy_services.sh" "${User}@${HostIP}:/tmp/"
Write-Host "[OK] Script uploaded" -ForegroundColor Green
Write-Host ""

Write-Host "Making script executable and running..." -ForegroundColor Yellow
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "chmod +x /tmp/deploy_services.sh && /tmp/deploy_services.sh"
Write-Host ""

Write-Host "Cleaning up..." -ForegroundColor Yellow
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "rm -f /tmp/deploy_services.sh"
Write-Host "[OK] Cleanup completed" -ForegroundColor Green
Write-Host ""
