param(
    [string]$HostIP = "192.168.10.82",
    [int]$Port = 80,
    [string]$User = "sen",
    [string]$Password = "123",
    [string]$SudoPassword = "123",
    [string]$Network = "host",
    [string]$RemotePath = "/opt/qingtian"
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  QingTian 项目一键部署脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "目标服务器: $User@$HostIP" -ForegroundColor Green
Write-Host "前端端口: $Port" -ForegroundColor Green
Write-Host "网络模式: $Network" -ForegroundColor Green
Write-Host "远程路径: $RemotePath" -ForegroundColor Green
Write-Host ""

function Execute-SSHCommand {
    param(
        [string]$Command,
        [switch]$UseSudo
    )
    
    $fullCommand = $Command
    if ($UseSudo) {
        $fullCommand = "echo '$SudoPassword' | sudo -S $Command"
    }
    
    $result = plink -batch -pw $Password "$User@$HostIP" $fullCommand 2>&1
    return $result
}

Write-Host "[1/6] 检查 SSH 连接..." -ForegroundColor Yellow
try {
    $result = plink -batch -pw $Password "$User@$HostIP" "echo '连接成功'" 2>&1
    Write-Host "✓ SSH 连接成功" -ForegroundColor Green
}
catch {
    Write-Host "✗ SSH 连接失败: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[2/6] 检查 Docker 是否安装..." -ForegroundColor Yellow
try {
    $result = plink -batch -pw $Password "$User@$HostIP" "docker --version" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Docker 已安装: $result" -ForegroundColor Green
    }
    else {
        Write-Host "Docker 未安装，开始安装..." -ForegroundColor Yellow
        
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
            Write-Host "执行: $cmd" -ForegroundColor Gray
            $result = Execute-SSHCommand -Command $cmd -UseSudo
        }
        
        Write-Host "✓ Docker 安装完成" -ForegroundColor Green
    }
}
catch {
    Write-Host "Docker 检查/安装失败: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "[3/6] 准备项目文件..." -ForegroundColor Yellow

$projectDir = $PSScriptRoot
$tempZip = Join-Path $env:TEMP "qingtian-deploy.zip"

Write-Host "压缩项目文件..." -ForegroundColor Gray
Compress-Archive -Path "$projectDir\*" -DestinationPath $tempZip -Force

Write-Host "创建远程目录..." -ForegroundColor Gray
Execute-SSHCommand -Command "mkdir -p $RemotePath" -UseSudo | Out-Null
Execute-SSHCommand -Command "chown $User:$User $RemotePath" -UseSudo | Out-Null

Write-Host "上传项目文件..." -ForegroundColor Gray
pscp -pw $Password $tempZip "$User@$HostIP`:$RemotePath/project.zip"

Write-Host "解压项目文件..." -ForegroundColor Gray
Execute-SSHCommand -Command "cd $RemotePath && unzip -o project.zip && rm -f project.zip" | Out-Null

Write-Host "✓ 项目文件上传完成" -ForegroundColor Green

Write-Host ""
Write-Host "[4/6] 配置 docker-compose..." -ForegroundColor Yellow

if ($Network -eq "host") {
    Write-Host "应用 Host 网络模式配置..." -ForegroundColor Gray
    
    $composePath = Join-Path $projectDir "docker-compose.yml"
    $composeContent = Get-Content $composePath -Raw -Encoding UTF8
    
    $composeContent = $composeContent -replace '    networks:\n      - qingtian-network', '    network_mode: host'
    $composeContent = $composeContent -replace '\nnetworks:\n  qingtian-network:\n    driver: bridge', ''
    $composeContent = $composeContent -replace '    ports:\n      - "\$\{FRONTEND_PORT:-80\}:80"\n', ''
    
    $composeContent = $composeContent -replace 'QT_MONGO_URI: mongodb://mongodb:27017', 'QT_MONGO_URI: mongodb://127.0.0.1:27017'
    $composeContent = $composeContent -replace 'QT_REDIS_ADDR: redis:6379', 'QT_REDIS_ADDR: 127.0.0.1:6379'
    $composeContent = $composeContent -replace 'QT_NSQD_ADDR: nsqd:4150', 'QT_NSQD_ADDR: 127.0.0.1:4150'
    $composeContent = $composeContent -replace 'QT_NSQ_LOOKUPD_ADDR: nsqlookupd:4161', 'QT_NSQ_LOOKUPD_ADDR: 127.0.0.1:4161'
    $composeContent = $composeContent -replace 'QT_MINIO_ENDPOINT: minio:9000', 'QT_MINIO_ENDPOINT: 127.0.0.1:9000'
    $composeContent = $composeContent -replace 'QT_CONTROL_PLANE_URL: http://apiserver:8080', 'QT_CONTROL_PLANE_URL: http://127.0.0.1:8080'
    $composeContent = $composeContent -replace '- API_SERVER=apiserver', "- API_SERVER=127.0.0.1`n      - NGINX_PORT=$Port"
    
    $tempCompose = Join-Path $env:TEMP "docker-compose-host.yml"
    Set-Content -Path $tempCompose -Value $composeContent -Encoding UTF8 -NoNewline
    
    pscp -pw $Password $tempCompose "$User@$HostIP`:$RemotePath/docker-compose.yml"
    Remove-Item $tempCompose -Force
}

Write-Host "✓ docker-compose 配置完成" -ForegroundColor Green

Write-Host ""
Write-Host "[5/6] 停止现有容器并清理..." -ForegroundColor Yellow

Execute-SSHCommand -Command "cd $RemotePath && echo '$SudoPassword' | sudo -S docker compose down 2>/dev/null || true" | Out-Null
Execute-SSHCommand -Command "echo '$SudoPassword' | sudo -S docker rm -f `$(docker ps -aq --filter name=qingtian 2>/dev/null) 2>/dev/null || true" | Out-Null

Write-Host "✓ 清理完成" -ForegroundColor Green

Write-Host ""
Write-Host "[6/6] 构建并启动服务（这可能需要几分钟）..." -ForegroundColor Yellow

Execute-SSHCommand -Command "cd $RemotePath && echo '$SudoPassword' | sudo -S docker compose up -d --build"

Write-Host ""
Write-Host "等待服务启动..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

Write-Host ""
Write-Host "检查服务状态..." -ForegroundColor Yellow
Execute-SSHCommand -Command "echo '$SudoPassword' | sudo -S docker ps --filter name=qingtian"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  部署完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "前端访问地址: http://$HostIP`:$Port" -ForegroundColor Green
Write-Host ""
Write-Host "查看日志命令:" -ForegroundColor Yellow
Write-Host "  ssh $User@$HostIP" -ForegroundColor Gray
Write-Host "  cd $RemotePath" -ForegroundColor Gray
Write-Host "  sudo docker compose logs -f" -ForegroundColor Gray
Write-Host ""

Remove-Item $tempZip -Force -ErrorAction SilentlyContinue
