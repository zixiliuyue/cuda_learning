param(
    [string]$HostIP = "192.168.10.82",
    [string]$User = "sen",
    [string]$Password = "123",
    [string]$RemotePath = "/opt/qingtian"
)

$ErrorActionPreference = "Continue"
$plink = "$env:TEMP\plink.exe"
$pscp = "$env:TEMP\pscp.exe"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  QingTian Project Auto Deploy" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Make sure plink/pscp are available
if (-not (Test-Path $plink) -or -not (Test-Path $pscp)) {
    Write-Host "[Prep] Downloading PuTTY tools..." -ForegroundColor Yellow
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri "https://the.earth.li/~sgtatham/putty/latest/w64/plink.exe" -OutFile $plink
    Invoke-WebRequest -Uri "https://the.earth.li/~sgtatham/putty/latest/w64/pscp.exe" -OutFile $pscp
    Write-Host "[OK] Tools downloaded" -ForegroundColor Green
    Write-Host ""
}

# Step 1: Upload files
Write-Host "[1/4] Uploading project files to server..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Gray

$localZip = Join-Path $PSScriptRoot "qingtian.tar.gz"
$remoteZip = "/tmp/qingtian.tar.gz"

& $pscp -pw $Password -hostkey "d94Qzn+DCcR1gBxnlMqpsNMK9DMds1d5rihg30Orh6k" $localZip ${User}@${HostIP}:$remoteZip

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Upload failed, trying alternative method..." -ForegroundColor Yellow
    
    # Try without hostkey check - use echo to auto accept
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $pscp
    $psi.Arguments = "-pw $Password `"$localZip`" `"${User}@${HostIP}:$remoteZip`""
    $psi.UseShellExecute = $false
    $psi.RedirectStandardInput = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    
    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    $proc.Start() | Out-Null
    
    # Send 'y' to accept hostkey
    Start-Sleep -Milliseconds 500
    $proc.StandardInput.WriteLine("y")
    $proc.StandardInput.Flush()
    
    $proc.WaitForExit(60000)
    
    if ($proc.ExitCode -ne 0) {
        Write-Host "[ERROR] Upload failed, try manual upload" -ForegroundColor Red
        exit 1
    }
}

Write-Host "[OK] File upload completed" -ForegroundColor Green
Write-Host ""

# Step 2: Prepare environment and extract
Write-Host "[2/4] Preparing environment and extracting files..." -ForegroundColor Yellow

$commands = @(
    "echo '$Password' | sudo -S mkdir -p $RemotePath",
    "echo '$Password' | sudo -S chown ${User}:${User} $RemotePath",
    "cd $RemotePath && mv /tmp/qingtian.tar.gz . && tar -xzf qingtian.tar.gz && rm -f qingtian.tar.gz"
)

foreach ($cmd in $commands) {
    & $plink -pw $Password -batch -hostkey "d94Qzn+DCcR1gBxnlMqpsNMK9DMds1d5rihg30Orh6k" ${User}@${HostIP} $cmd
}

Write-Host "[OK] Environment prepared" -ForegroundColor Green
Write-Host ""

# Step 3: Cleanup existing containers
Write-Host "[3/4] Cleaning up existing containers..." -ForegroundColor Yellow
$cleanupCmd = "cd $RemotePath && echo '$Password' | sudo -S docker compose down 2>/dev/null || true; echo '$Password' | sudo -S docker rm -f `$(docker ps -aq --filter name=qingtian 2>/dev/null) 2>/dev/null || true"
& $plink -pw $Password -batch -hostkey "d94Qzn+DCcR1gBxnlMqpsNMK9DMds1d5rihg30Orh6k" ${User}@${HostIP} $cleanupCmd
Write-Host "[OK] Cleanup completed" -ForegroundColor Green
Write-Host ""

# Step 4: Build and start services
Write-Host "[4/4] Building and starting services..." -ForegroundColor Yellow
Write-Host "NOTE: This may take 5-15 minutes. Please be patient!" -ForegroundColor Gray
Write-Host "==================================================" -ForegroundColor Gray

$deployCmd = "cd $RemotePath && echo '$Password' | sudo -S timeout 1800 docker compose up -d --build"

# Use plink with longer timeout
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $plink
$psi.Arguments = "-pw $Password -batch -hostkey `"d94Qzn+DCcR1gBxnlMqpsNMK9DMds1d5rihg30Orh6k`" ${User}@${HostIP} `"$deployCmd`""
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi
$proc.Start() | Out-Null

$sb = New-Object System.Text.StringBuilder
while (-not $proc.HasExited) {
    $line = $proc.StandardOutput.ReadLine()
    if ($line) {
        [void]$sb.AppendLine($line)
        Write-Host $line
    }
    
    $errLine = $proc.StandardError.ReadLine()
    if ($errLine) {
        Write-Host "ERR: $errLine" -ForegroundColor Red
    }
}

# Read remaining output
while ($null -ne ($line = $proc.StandardOutput.ReadLine())) {
    [void]$sb.AppendLine($line)
    Write-Host $line
}

$proc.WaitForExit()

Write-Host ""
if ($proc.ExitCode -eq 0) {
    Write-Host "[OK] Services started successfully" -ForegroundColor Green
} else {
    Write-Host "[WARN] Build may have issues, check logs" -ForegroundColor Yellow
}
Write-Host ""

# Wait and check status
Write-Host "Waiting for services to initialize (20 seconds)..." -ForegroundColor Gray
Start-Sleep -Seconds 20

Write-Host ""
Write-Host "Checking running containers..." -ForegroundColor Yellow
& $plink -pw $Password -batch -hostkey "d94Qzn+DCcR1gBxnlMqpsNMK9DMds1d5rihg30Orh6k" ${User}@${HostIP} "echo '$Password' | sudo -S docker ps --filter name=qingtian"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DEPLOY COMPLETED!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Frontend URL: http://$HostIP" -ForegroundColor Green
Write-Host ""
Write-Host "To check logs manually:" -ForegroundColor Yellow
Write-Host "  ssh $User@$HostIP" -ForegroundColor Gray
Write-Host "  cd $RemotePath" -ForegroundColor Gray
Write-Host "  sudo docker compose logs -f" -ForegroundColor Gray
Write-Host ""
