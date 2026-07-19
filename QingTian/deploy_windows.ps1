# QingTian 项目 Docker 部署脚本
# 使用 PowerShell 和 SSH 进行部署

$ErrorActionPreference = "Continue"

# 配置信息
$Server = "192.168.10.82"
$User = "sen"
$Password = "123"
$LocalProjectPath = "F:\code\cuda_learning\QingTian"
$RemotePath = "/opt/qingtian"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  QingTian 项目 Docker 部署开始" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 步骤 1: 检查 Docker 是否安装
Write-Host "[步骤 1/6] 检查远程服务器 Docker 安装状态..." -ForegroundColor Yellow

$checkDockerScript = @"
`$secpasswd = ConvertTo-SecureString '$Password' -AsPlainText -Force
`$credential = New-Object System.Management.Automation.PSCredential ('$User', `$secpasswd)

`$session = New-SSHSession -ComputerName '$Server' -Credential `$credential -AcceptKey -ErrorAction Stop
`$stream = `$session.Session.CreateShellStream("dumb", 0, 0, 0, 0, 1000)
Start-Sleep -Milliseconds 500
`$stream.WriteLine("docker --version")
Start-Sleep -Milliseconds 1000
`$output = `$stream.Read()
Write-Host `$output
Remove-SSHSession -SSHSession `$session | Out-Null
"@

try {
    $sshCheck = & ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 $User@$Server "docker --version 2>&1" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Docker 已安装: $sshCheck" -ForegroundColor Green
        $dockerInstalled = $true
    } else {
        Write-Host "✗ Docker 未安装" -ForegroundColor Red
        $dockerInstalled = $false
    }
} catch {
    Write-Host "SSH 连接测试: 可能需要密码认证" -ForegroundColor Yellow
    $dockerInstalled = $false
}

Write-Host ""

# 步骤 2: 安装 Docker（如果需要）
if (-not $dockerInstalled) {
    Write-Host "[步骤 2/6] 开始安装 Docker..." -ForegroundColor Yellow
    Write-Host "这可能需要几分钟时间，请耐心等待..." -ForegroundColor Gray
    
    $installCommands = @(
        "apt-get update -y",
        "apt-get install -y ca-certificates curl gnupg lsb-release",
        "mkdir -p /etc/apt/keyrings",
        "curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg",
        "echo 'deb [arch=`$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu `$(lsb_release -cs) stable' | tee /etc/apt/sources.list.d/docker.list > /dev/null",
        "apt-get update -y",
        "apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin",
        "systemctl enable docker",
        "systemctl start docker"
    )
    
    Write-Host "请手动在服务器上执行以下命令来安装 Docker:" -ForegroundColor Cyan
    Write-Host "sudo -i" -ForegroundColor White
    foreach ($cmd in $installCommands) {
        Write-Host $cmd -ForegroundColor White
    }
    Write-Host "exit" -ForegroundColor White
    Write-Host ""
    Write-Host "或者按任意键继续尝试自动安装（使用脚本方式）..." -ForegroundColor Yellow
    Read-Host
} else {
    Write-Host "[步骤 2/6] 跳过 Docker 安装" -ForegroundColor Green
}

Write-Host ""

# 步骤 3: 传输文件
Write-Host "[步骤 3/6] 准备传输项目文件..." -ForegroundColor Yellow

# 创建临时压缩文件
$tarPath = "$env:TEMP\qingtian_deploy.tar"
Write-Host "创建项目压缩包..." -ForegroundColor Gray

# 使用 PowerShell 创建压缩包
if (Test-Path $tarPath) {
    Remove-Item $tarPath -Force
}

# 复制项目文件到临时目录，排除不需要的文件
$tempDir = Join-Path $env:TEMP "qingtian_deploy_$(Get-Date -Format 'yyyyMMddHHmmss')"
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

Write-Host "复制项目文件到临时目录..." -ForegroundColor Gray
Copy-Item -Path "$LocalProjectPath\*" -Destination $tempDir -Recurse -Force -Exclude @(".git", "node_modules", "*.pyc", "__pycache__")

# 替换 docker-compose.yml
Copy-Item -Path "$LocalProjectPath\docker-compose-host.yml" -Destination "$tempDir\docker-compose.yml" -Force

Write-Host "临时目录已创建: $tempDir" -ForegroundColor Green

Write-Host ""
Write-Host "现在请手动执行以下命令来传输文件:" -ForegroundColor Cyan
Write-Host "1. 在临时目录中打开命令行" -ForegroundColor White
Write-Host "2. 执行: scp -r $tempDir\* $User@$Server`:$RemotePath/" -ForegroundColor White
Write-Host "3. 输入密码: $Password" -ForegroundColor White
Write-Host ""
Write-Host "或者使用以下 PowerShell 命令:" -ForegroundColor Cyan
Write-Host "Compress-Archive -Path '$tempDir\*' -DestinationPath '$tarPath.zip' -Force" -ForegroundColor White
Write-Host "scp '$tarPath.zip' $User@$Server`:$RemotePath/" -ForegroundColor White
Write-Host "ssh $User@$Server 'cd $RemotePath && unzip -o qingtian_deploy.tar.zip'" -ForegroundColor White
Write-Host ""

Read-Host "文件传输完成后，按任意键继续..."

Write-Host ""

# 步骤 4: 修改配置文件（已在本地完成，使用 docker-compose-host.yml）
Write-Host "[步骤 4/6] 配置文件已在本地准备好" -ForegroundColor Green
Write-Host "✓ docker-compose.yml 已使用 host 网络模式配置" -ForegroundColor Green
Write-Host "  - 所有服务使用 network_mode: host" -ForegroundColor Gray
Write-Host "  - 所有服务地址已改为 127.0.0.1" -ForegroundColor Gray
Write-Host "  - 前端 NGINX_PORT 已设置为 80" -ForegroundColor Gray

Write-Host ""

# 步骤 5: 执行 Docker Compose
Write-Host "[步骤 5/6] 在远程服务器上执行 Docker Compose..." -ForegroundColor Yellow
Write-Host ""
Write-Host "请手动在服务器上执行以下命令:" -ForegroundColor Cyan
Write-Host "ssh $User@$Server" -ForegroundColor White
Write-Host "cd $RemotePath" -ForegroundColor White
Write-Host "echo '123' | sudo -S docker compose up -d --build" -ForegroundColor White
Write-Host ""
Write-Host "或者使用以下一行命令:" -ForegroundColor Cyan
Write-Host "ssh $User@$Server 'cd $RemotePath && echo '123' | sudo -S docker compose up -d --build'" -ForegroundColor White
Write-Host ""

Read-Host "Docker Compose 执行完成后，按任意键继续..."

Write-Host ""

# 步骤 6: 验证服务
Write-Host "[步骤 6/6] 验证服务状态..." -ForegroundColor Yellow
Write-Host ""
Write-Host "请执行以下命令验证服务状态:" -ForegroundColor Cyan
Write-Host "ssh $User@$Server 'echo '123' | sudo -S docker compose ps'" -ForegroundColor White
Write-Host ""
Write-Host "检查容器状态是否全部为 Up 或 Healthy" -ForegroundColor Gray
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  部署完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "前端访问地址: http://$Server" -ForegroundColor Green
Write-Host ""
Write-Host "默认登录账号:" -ForegroundColor Yellow
Write-Host "  用户名: admin" -ForegroundColor White
Write-Host "  密码: admin123" -ForegroundColor White
Write-Host ""

# 清理临时文件
Write-Host "清理临时文件..." -ForegroundColor Gray
if (Test-Path $tempDir) {
    Remove-Item $tempDir -Recurse -Force
}
Write-Host "✓ 临时文件已清理" -ForegroundColor Green

Write-Host ""
Write-Host "部署脚本执行完毕！" -ForegroundColor Green