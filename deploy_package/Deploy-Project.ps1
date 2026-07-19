param(
    [string]$HostIP = "192.168.10.82",
    [int]$Port = 80,
    [string]$User = "sen",
    [string]$Password = "123",
    [string]$RemotePath = "/opt/qingtian",
    [string]$HostKey = "SHA256:d94Qzn+DCcR1gBxnlMqpsNMK9DMds1d5rihg30Orh6k"
)

$ErrorActionPreference = "Continue"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  QingTian Project Auto Deploy Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get plink and pscp paths
$plink = Join-Path $env:TEMP "plink.exe"
$pscp = Join-Path $env:TEMP "pscp.exe"

# Download PuTTY tools if not exist
if (-not (Test-Path $plink) -or -not (Test-Path $pscp)) {
    Write-Host "[Prep] Downloading PuTTY tools..." -ForegroundColor Yellow
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri "https://the.earth.li/~sgtatham/putty/latest/w64/plink.exe" -OutFile $plink
    Invoke-WebRequest -Uri "https://the.earth.li/~sgtatham/putty/latest/w64/pscp.exe" -OutFile $pscp
    Write-Host "[OK] PuTTY tools downloaded" -ForegroundColor Green
    Write-Host ""
}

Write-Host "[1/5] Connecting to remote server..." -ForegroundColor Yellow
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "exit"
Write-Host "[OK] SSH connection ready" -ForegroundColor Green
Write-Host ""

Write-Host "[2/5] Uploading project files..." -ForegroundColor Yellow

# Create remote directory
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "echo '$Password' | sudo -S mkdir -p $RemotePath"

# Change ownership
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "echo '$Password' | sudo -S chown ${User}:${User} $RemotePath"

# Upload zip file
$localZip = Join-Path $PSScriptRoot "qingtian.tar.gz"
Write-Host "Uploading file..." -ForegroundColor Gray
& $pscp -pw $Password -batch -hostkey $HostKey $localZip "${User}@${HostIP}:${RemotePath}/"

# Extract files
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "cd $RemotePath && tar -xzf qingtian.tar.gz && rm -f qingtian.tar.gz"

Write-Host "[OK] Project files uploaded" -ForegroundColor Green
Write-Host ""

Write-Host "[3/5] Checking Docker installation..." -ForegroundColor Yellow
$dockerCheck = & $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "docker --version 2>&1"
Write-Host $dockerCheck

if ($dockerCheck -match "not found" -or $LASTEXITCODE -ne 0) {
    Write-Host "Docker not found, installing..." -ForegroundColor Yellow
    
    $installCommands = @(
        "apt-get update",
        "apt-get install -y ca-certificates curl gnupg lsb-release",
        "mkdir -p /etc/apt/keyrings",
        "curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg",
        "echo 'deb [arch=`$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu `$(lsb_release -cs) stable' | tee /etc/apt/sources.list.d/docker.list > /dev/null",
        "apt-get update",
        "apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin",
        "systemctl enable docker",
        "systemctl start docker"
    )
    
    foreach ($cmd in $installCommands) {
        Write-Host "Executing: $cmd" -ForegroundColor Gray
        & $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "echo '$Password' | sudo -S $cmd"
    }
    
    Write-Host "[OK] Docker installation completed" -ForegroundColor Green
}
else {
    Write-Host "[OK] Docker is already installed" -ForegroundColor Green
}
Write-Host ""

Write-Host "[4/5] Cleaning up existing containers..." -ForegroundColor Yellow
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "cd $RemotePath && echo '$Password' | sudo -S docker compose down 2>/dev/null; exit 0"
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "echo '$Password' | sudo -S docker rm -f `$(docker ps -aq --filter name=qingtian 2>/dev/null) 2>/dev/null; exit 0"
Write-Host "[OK] Cleanup completed" -ForegroundColor Green
Write-Host ""

Write-Host "[5/5] Building and starting services..." -ForegroundColor Yellow
Write-Host "NOTE: This may take 5-15 minutes, please wait..." -ForegroundColor Gray
Write-Host "==================================================" -ForegroundColor Gray

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $plink
$psi.Arguments = "-pw $Password -batch -hostkey $HostKey $User@$HostIP `"cd $RemotePath && echo '$Password' | sudo -S docker compose up -d --build`""
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi
$proc.Start() | Out-Null

while (-not $proc.HasExited) {
    $line = $proc.StandardOutput.ReadLine()
    if ($line) {
        Write-Host $line
    }
    $errLine = $proc.StandardError.ReadLine()
    if ($errLine) {
        Write-Host "ERR: $errLine" -ForegroundColor Red
    }
}

# Read remaining output
while ($null -ne ($line = $proc.StandardOutput.ReadLine())) {
    Write-Host $line
}

$proc.WaitForExit()

Write-Host ""
if ($proc.ExitCode -eq 0) {
    Write-Host "[OK] Services started successfully" -ForegroundColor Green
} else {
    Write-Host "[WARN] Command completed, check status below..." -ForegroundColor Yellow
}
Write-Host ""

Write-Host "Waiting for services to initialize (20 seconds)..." -ForegroundColor Gray
Start-Sleep -Seconds 20

Write-Host ""
Write-Host "Running containers:" -ForegroundColor Yellow
& $plink -pw $Password -batch -hostkey $HostKey $User@$HostIP "echo '$Password' | sudo -S docker ps --filter name=qingtian"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DEPLOY COMPLETED!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Frontend URL: http://${HostIP}:${Port}" -ForegroundColor Green
Write-Host ""
Write-Host "Useful commands (run on server):" -ForegroundColor Yellow
Write-Host "  Enter deploy dir: cd $RemotePath" -ForegroundColor Gray
Write-Host "  List all containers: sudo docker ps -a" -ForegroundColor Gray
Write-Host "  View all logs: sudo docker compose logs -f" -ForegroundColor Gray
Write-Host "  Restart services: sudo docker compose restart" -ForegroundColor Gray
Write-Host "  Stop services: sudo docker compose down" -ForegroundColor Gray
Write-Host ""
