$Password = "123"
$User = "sen"
$HostIP = "192.168.10.82"

$ErrorActionPreference = "Continue"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Installing Docker on Remote Server" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$plink = Join-Path $env:TEMP "plink.exe"

if (-not (Test-Path $plink)) {
    Write-Host "[Prep] Downloading PuTTY plink..." -ForegroundColor Yellow
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri "https://the.earth.li/~sgtatham/putty/latest/w64/plink.exe" -OutFile $plink
    Write-Host "[OK] plink downloaded" -ForegroundColor Green
    Write-Host ""
}

$HostKey = "SHA256:d94Qzn+DCcR1gBxnlMqpsNMK9DMds1d5rihg30Orh6k"

Write-Host "[1/7] Updating package list..." -ForegroundColor Yellow
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "echo '$Password' | sudo -S apt-get update"
Write-Host "[OK] Package list updated" -ForegroundColor Green
Write-Host ""

Write-Host "[2/7] Installing prerequisites..." -ForegroundColor Yellow
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "echo '$Password' | sudo -S apt-get install -y ca-certificates curl gnupg lsb-release"
Write-Host "[OK] Prerequisites installed" -ForegroundColor Green
Write-Host ""

Write-Host "[3/7] Creating keyrings directory..." -ForegroundColor Yellow
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "echo '$Password' | sudo -S mkdir -p /etc/apt/keyrings"
Write-Host "[OK] Directory created" -ForegroundColor Green
Write-Host ""

Write-Host "[4/7] Adding Docker GPG key..." -ForegroundColor Yellow
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "curl -fsSL https://download.docker.com/linux/ubuntu/gpg | echo '$Password' | sudo -S gpg --dearmor -o /etc/apt/keyrings/docker.gpg"
Write-Host "[OK] GPG key added" -ForegroundColor Green
Write-Host ""

Write-Host "[5/7] Setting up Docker repository..." -ForegroundColor Yellow
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "echo '$Password' | sudo -S sh -c 'echo \"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable\" | tee /etc/apt/sources.list.d/docker.list > /dev/null'"
Write-Host "[OK] Repository setup" -ForegroundColor Green
Write-Host ""

Write-Host "[6/7] Installing Docker..." -ForegroundColor Yellow
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "echo '$Password' | sudo -S apt-get update"
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "echo '$Password' | sudo -S apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin"
Write-Host "[OK] Docker installed" -ForegroundColor Green
Write-Host ""

Write-Host "[7/7] Starting and enabling Docker..." -ForegroundColor Yellow
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "echo '$Password' | sudo -S systemctl enable docker"
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "echo '$Password' | sudo -S systemctl start docker"
Write-Host "[OK] Docker started" -ForegroundColor Green
Write-Host ""

Write-Host "Verifying Docker installation..." -ForegroundColor Yellow
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "echo '$Password' | sudo -S docker --version"
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DOCKER INSTALLATION COMPLETED!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
