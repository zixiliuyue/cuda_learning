param(
    [string]$HostIP = "192.168.10.82",
    [string]$User = "sen",
    [string]$Password = "123",
    [string]$RemotePath = "/opt/qingtian"
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  QingTian 项目部署脚本 (修复版)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 下载 SSH.NET 库
Write-Host "[准备] 下载 SSH.NET 库..." -ForegroundColor Yellow
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
Write-Host "[OK] SSH.NET 库已加载" -ForegroundColor Green
Write-Host ""

# 创建连接
$connInfo = New-Object Renci.SshNet.PasswordConnectionInfo($HostIP, 22, $User, $Password)
$connInfo.Timeout = New-TimeSpan -Seconds 30

Write-Host "[1/5] 连接到远程服务器..." -ForegroundColor Yellow
$sshClient = New-Object Renci.SshNet.SshClient($connInfo)
$sshClient.Connect()
Write-Host "[OK] SSH 连接成功" -ForegroundColor Green
Write-Host ""

# 创建 SFTP 客户端
$sftpClient = New-Object Renci.SshNet.SftpClient($connInfo)
$sftpClient.Connect()

Write-Host "[2/5] 上传项目文件..." -ForegroundColor Yellow
$localZip = Join-Path $PSScriptRoot "qingtian.tar.gz"
$remoteZip = "/tmp/qingtian.tar.gz"

$fileStream = [System.IO.File]::OpenRead($localZip)
$sftpClient.UploadFile($fileStream, $remoteZip, $true)
$fileStream.Close()
Write-Host "[OK] 文件上传完成" -ForegroundColor Green
Write-Host ""

Write-Host "[3/5] 准备部署环境..." -ForegroundColor Yellow

# 1. 创建目录
$cmd1 = $sshClient.CreateCommand("echo '$Password' | sudo -S mkdir -p $RemotePath")
$cmd1.Execute()

# 2. 修改权限
$chownCmd = "echo '$Password' | sudo -S chown ${User}:${User} $RemotePath"
$cmd2 = $sshClient.CreateCommand($chownCmd)
$cmd2.Execute()

# 3. 解压文件
$cmd3 = $sshClient.CreateCommand("cd $RemotePath && mv /tmp/qingtian.tar.gz . && tar -xzf qingtian.tar.gz && rm -f qingtian.tar.gz")
$cmd3.Execute()
Write-Host "[OK] 环境准备完成" -ForegroundColor Green
Write-Host ""

Write-Host "[4/5] 清理现有容器..." -ForegroundColor Yellow

# 停止现有 compose
$cmd4 = $sshClient.CreateCommand("cd $RemotePath && echo '$Password' | sudo -S docker compose down 2>/dev/null; exit 0")
$cmd4.Execute()

# 删除相关容器
$cmd5 = $sshClient.CreateCommand("echo '$Password' | sudo -S docker rm -f `$(docker ps -aq --filter name=qingtian 2>/dev/null) 2>/dev/null; exit 0")
$cmd5.Execute()

Write-Host "[OK] 清理完成" -ForegroundColor Green
Write-Host ""

Write-Host "[5/5] 构建并启动服务..." -ForegroundColor Yellow
Write-Host "注意：构建可能需要 5-15 分钟，请耐心等待..." -ForegroundColor Gray
Write-Host "==================================================" -ForegroundColor Gray

$buildCmd = "cd $RemotePath && echo '$Password' | sudo -S docker compose up -d --build"
$cmd6 = $sshClient.CreateCommand($buildCmd)
$cmd6.CommandTimeout = New-TimeSpan -Minutes 30
$result = $cmd6.Execute()

Write-Host $result

if ($cmd6.ExitStatus -eq 0) {
    Write-Host ""
    Write-Host "[OK] 服务启动成功" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[警告] 构建过程可能有问题，请检查日志" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "等待服务初始化 (15秒)..." -ForegroundColor Gray
Start-Sleep -Seconds 15

Write-Host ""
Write-Host "检查运行中的容器..." -ForegroundColor Yellow
$checkCmd = "echo '$Password' | sudo -S docker ps --filter name=qingtian"
$cmd7 = $sshClient.CreateCommand($checkCmd)
$result = $cmd7.Execute()
Write-Host $result
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  部署完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "前端访问地址: http://$HostIP" -ForegroundColor Green
Write-Host ""
Write-Host "常用命令（在服务器上执行）:" -ForegroundColor Yellow
Write-Host "  进入部署目录: cd $RemotePath" -ForegroundColor Gray
Write-Host "  查看所有容器: sudo docker ps -a" -ForegroundColor Gray
Write-Host "  查看所有日志: sudo docker compose logs -f" -ForegroundColor Gray
Write-Host "  查看 API 日志: sudo docker compose logs -f apiserver" -ForegroundColor Gray
Write-Host "  重启服务: sudo docker compose restart" -ForegroundColor Gray
Write-Host "  停止服务: sudo docker compose down" -ForegroundColor Gray
Write-Host ""

# 清理连接
$sftpClient.Disconnect()
$sftpClient.Dispose()
$sshClient.Disconnect()
$sshClient.Dispose()
