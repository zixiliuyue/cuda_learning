param(
    [string]$HostIP = "192.168.10.82",
    [string]$User = "sen",
    [string]$Password = "123",
    [string]$RemotePath = "/opt/qingtian"
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  QingTian Project Auto Deploy Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Download SSH.NET library
Write-Host "[Prep] Downloading SSH.NET library..." -ForegroundColor Yellow
$sshDllPath = Join-Path $env:TEMP "Renci.SshNet.dll"

if (-not (Test-Path $sshDllPath)) {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri "https://www.nuget.org/api/v2/package/SSH.NET/2024.1.0" -OutFile "$env:TEMP\sshnet.zip"
    Expand-Archive -Path "$env:TEMP\sshnet.zip" -DestinationPath "$env:TEMP\sshnet" -Force
    Copy-Item "$env:TEMP\sshnet\lib\netstandard2.0\Renci.SshNet.dll" $sshDllPath -Force
    Remove-Item "$env:TEMP\sshnet" -Recurse -Force
    Remove-Item "$env:TEMP\sshnet.zip" -Force
}

Add-Type -Path $sshDllPath
Write-Host "[OK] SSH.NET library loaded" -ForegroundColor Green
Write-Host ""

# Create SSH connection
$connInfo = New-Object Renci.SshNet.PasswordConnectionInfo($HostIP, 22, $User, $Password)
$connInfo.Timeout = New-TimeSpan -Seconds 30

Write-Host "[1/5] Connecting to remote server..." -ForegroundColor Yellow
$sshClient = New-Object Renci.SshNet.SshClient($connInfo)
$sshClient.Connect()
Write-Host "[OK] SSH connected successfully" -ForegroundColor Green
Write-Host ""

# Create SFTP client for file upload
$sftpClient = New-Object Renci.SshNet.SftpClient($connInfo)
$sftpClient.Connect()

Write-Host "[2/5] Uploading project files..." -ForegroundColor Yellow
$localZip = Join-Path $PSScriptRoot "qingtian.tar.gz"
$remoteZip = "/tmp/qingtian.tar.gz"

$fileStream = [System.IO.File]::OpenRead($localZip)
$sftpClient.UploadFile($fileStream, $remoteZip, $true)
$fileStream.Close()
Write-Host "[OK] File upload completed" -ForegroundColor Green
Write-Host ""

Write-Host "[3/5] Preparing deploy environment..." -ForegroundColor Yellow

# Create directory
$cmd1 = $sshClient.CreateCommand("echo '$Password' | sudo -S mkdir -p $RemotePath")
$cmd1.Execute()

# Change ownership
$chownCmd = "echo '$Password' | sudo -S chown ${User}:${User} $RemotePath"
$cmd2 = $sshClient.CreateCommand($chownCmd)
$cmd2.Execute()

# Extract files
$cmd3 = $sshClient.CreateCommand("cd $RemotePath && mv /tmp/qingtian.tar.gz . && tar -xzf qingtian.tar.gz && rm -f qingtian.tar.gz")
$cmd3.Execute()
Write-Host "[OK] Environment prepared" -ForegroundColor Green
Write-Host ""

Write-Host "[4/5] Cleaning up existing containers..." -ForegroundColor Yellow

# Stop existing docker compose
$cmd4 = $sshClient.CreateCommand("cd $RemotePath && echo '$Password' | sudo -S docker compose down 2>/dev/null; exit 0")
$cmd4.Execute()

# Remove related containers
$cmd5 = $sshClient.CreateCommand("echo '$Password' | sudo -S docker rm -f `$(docker ps -aq --filter name=qingtian 2>/dev/null) 2>/dev/null; exit 0")
$cmd5.Execute()

Write-Host "[OK] Cleanup completed" -ForegroundColor Green
Write-Host ""

Write-Host "[5/5] Building and starting services..." -ForegroundColor Yellow
Write-Host "NOTE: This may take 5-15 minutes, please wait..." -ForegroundColor Gray
Write-Host "==================================================" -ForegroundColor Gray

$buildCmd = "cd $RemotePath && echo '$Password' | sudo -S docker compose up -d --build"
$cmd6 = $sshClient.CreateCommand($buildCmd)
$cmd6.CommandTimeout = New-TimeSpan -Minutes 30
$result = $cmd6.Execute()

Write-Host $result

if ($cmd6.ExitStatus -eq 0) {
    Write-Host ""
    Write-Host "[OK] Services started successfully" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[WARN] Possible issues during build, check logs" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "Waiting for services to initialize (15 seconds)..." -ForegroundColor Gray
Start-Sleep -Seconds 15

Write-Host ""
Write-Host "Checking running containers..." -ForegroundColor Yellow
$checkCmd = "echo '$Password' | sudo -S docker ps --filter name=qingtian"
$cmd7 = $sshClient.CreateCommand($checkCmd)
$result = $cmd7.Execute()
Write-Host $result
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DEPLOY COMPLETED!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Frontend URL: http://$HostIP" -ForegroundColor Green
Write-Host ""
Write-Host "Useful commands (run on server):" -ForegroundColor Yellow
Write-Host "  Enter deploy dir: cd $RemotePath" -ForegroundColor Gray
Write-Host "  List all containers: sudo docker ps -a" -ForegroundColor Gray
Write-Host "  View all logs: sudo docker compose logs -f" -ForegroundColor Gray
Write-Host "  View API logs: sudo docker compose logs -f apiserver" -ForegroundColor Gray
Write-Host "  Restart services: sudo docker compose restart" -ForegroundColor Gray
Write-Host "  Stop services: sudo docker compose down" -ForegroundColor Gray
Write-Host ""

# Cleanup connections
$sftpClient.Disconnect()
$sftpClient.Dispose()
$sshClient.Disconnect()
$sshClient.Dispose()
