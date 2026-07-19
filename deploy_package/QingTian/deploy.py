#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import tarfile
import io
import time
from pathlib import Path

def run_remote_command(ssh_client, command, sudo_password=None):
    stdin, stdout, stderr = ssh_client.exec_command(command)
    if sudo_password:
        stdin.write(sudo_password + '\n')
        stdin.flush()
    exit_status = stdout.channel.recv_exit_status()
    output = stdout.read().decode('utf-8')
    error = stderr.read().decode('utf-8')
    return exit_status, output, error

def check_docker_installed(ssh_client, sudo_password):
    print("检查 Docker 是否安装...")
    status, output, error = run_remote_command(ssh_client, "docker --version")
    if status == 0:
        print(f"✓ Docker 已安装: {output.strip()}")
        return True
    return False

def install_docker(ssh_client, sudo_password):
    print("开始安装 Docker...")
    commands = [
        f"echo '{sudo_password}' | sudo -S apt-get update",
        f"echo '{sudo_password}' | sudo -S apt-get install -y ca-certificates curl gnupg lsb-release",
        f"echo '{sudo_password}' | sudo -S mkdir -p /etc/apt/keyrings",
        f"curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg",
        f"echo 'deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable' | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null",
        f"echo '{sudo_password}' | sudo -S apt-get update",
        f"echo '{sudo_password}' | sudo -S apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin",
        f"echo '{sudo_password}' | sudo -S systemctl enable docker",
        f"echo '{sudo_password}' | sudo -S systemctl start docker"
    ]
    
    for cmd in commands:
        status, output, error = run_remote_command(ssh_client, cmd)
        if status != 0:
            print(f"✗ 命令执行失败: {cmd}")
            print(f"错误: {error}")
            return False
    
    print("✓ Docker 安装完成")
    return True

def create_project_tar(project_path):
    print("创建项目压缩包...")
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
        for root, dirs, files in os.walk(project_path):
            for file in files:
                if file.endswith(('.pyc', '.pyo')) or '.git' in root or 'node_modules' in root:
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, project_path)
                tar.add(file_path, arcname=arcname)
    tar_buffer.seek(0)
    print("✓ 项目压缩包创建完成")
    return tar_buffer

def upload_and_extract(ssh_client, tar_buffer, remote_path):
    print(f"上传项目到远程服务器: {remote_path}")
    
    sftp = ssh_client.open_sftp()
    
    remote_tar = f"{remote_path}/qingtian_project.tar.gz"
    
    status, output, error = run_remote_command(ssh_client, f"mkdir -p {remote_path}")
    if status != 0:
        print(f"✗ 创建远程目录失败: {error}")
        return False
    
    with sftp.open(remote_tar, 'wb') as f:
        f.write(tar_buffer.read())
    
    status, output, error = run_remote_command(ssh_client, f"cd {remote_path} && tar -xzf qingtian_project.tar.gz")
    if status != 0:
        print(f"✗ 解压失败: {error}")
        return False
    
    run_remote_command(ssh_client, f"rm -f {remote_tar}")
    print("✓ 项目上传并解压完成")
    return True

def update_docker_compose_for_host(ssh_client, remote_path, frontend_port, network_mode):
    print(f"更新 docker-compose.yml (网络模式: {network_mode}, 前端端口: {frontend_port})...")
    
    sftp = ssh_client.open_sftp()
    compose_path = f"{remote_path}/docker-compose.yml"
    
    with sftp.open(compose_path, 'r') as f:
        content = f.read().decode('utf-8')
    
    if network_mode == 'host':
        content = content.replace('    networks:\n      - qingtian-network', '    network_mode: host')
        content = content.replace('\nnetworks:\n  qingtian-network:\n    driver: bridge', '')
        content = content.replace('    ports:\n      - "${FRONTEND_PORT:-80}:80"\n', '')
        
        content = content.replace('QT_MONGO_URI: mongodb://mongodb:27017', 'QT_MONGO_URI: mongodb://127.0.0.1:27017')
        content = content.replace('QT_REDIS_ADDR: redis:6379', 'QT_REDIS_ADDR: 127.0.0.1:6379')
        content = content.replace('QT_NSQD_ADDR: nsqd:4150', 'QT_NSQD_ADDR: 127.0.0.1:4150')
        content = content.replace('QT_NSQ_LOOKUPD_ADDR: nsqlookupd:4161', 'QT_NSQ_LOOKUPD_ADDR: 127.0.0.1:4161')
        content = content.replace('QT_MINIO_ENDPOINT: minio:9000', 'QT_MINIO_ENDPOINT: 127.0.0.1:9000')
        content = content.replace('QT_CONTROL_PLANE_URL: http://apiserver:8080', 'QT_CONTROL_PLANE_URL: http://127.0.0.1:8080')
        content = content.replace('- API_SERVER=apiserver', f'- API_SERVER=127.0.0.1\n      - NGINX_PORT={frontend_port}')
        
        env_content = f"FRONTEND_PORT={frontend_port}\n"
        sftp_env = ssh_client.open_sftp()
        with sftp_env.open(f"{remote_path}/.env", 'w') as f:
            f.write(env_content)
    
    sftp = ssh_client.open_sftp()
    with sftp.open(compose_path, 'w') as f:
        f.write(content)
    
    print("✓ docker-compose.yml 更新完成")
    return True

def stop_existing_containers(ssh_client, remote_path, sudo_password):
    print("停止并清理现有容器...")
    run_remote_command(ssh_client, f"cd {remote_path} && echo '{sudo_password}' | sudo -S docker compose down 2>/dev/null || true")
    run_remote_command(ssh_client, f"echo '{sudo_password}' | sudo -S docker rm -f $(docker ps -aq --filter name=qingtian) 2>/dev/null || true")
    print("✓ 现有容器清理完成")

def build_and_start(ssh_client, remote_path, sudo_password):
    print("开始构建并启动服务...")
    
    cmd = f"cd {remote_path} && echo '{sudo_password}' | sudo -S docker compose up -d --build"
    status, output, error = run_remote_command(ssh_client, cmd)
    
    if status != 0:
        print(f"✗ 构建失败: {error}")
        return False
    
    print("✓ 服务启动完成")
    return True

def wait_for_services(ssh_client, sudo_password):
    print("等待服务启动...")
    time.sleep(10)
    
    status, output, error = run_remote_command(ssh_client, f"echo '{sudo_password}' | sudo -S docker ps --filter name=qingtian")
    print(output)
    
    return True

def main():
    parser = argparse.ArgumentParser(description='QingTian 项目一键部署脚本')
    parser.add_argument('--host', default='192.168.10.82', help='目标服务器 IP')
    parser.add_argument('--port', type=int, default=80, help='前端访问端口')
    parser.add_argument('--user', default='sen', help='SSH 用户名')
    parser.add_argument('--password', default='123', help='SSH 密码')
    parser.add_argument('--sudo-password', default='123', help='sudo 密码')
    parser.add_argument('--network', default='host', choices=['host', 'bridge'], help='Docker 网络模式')
    parser.add_argument('--remote-path', default='/opt/qingtian', help='远程部署路径')
    
    args = parser.parse_args()
    
    project_path = Path(__file__).parent
    print(f"项目路径: {project_path}")
    print(f"目标服务器: {args.user}@{args.host}:{args.port}")
    print(f"网络模式: {args.network}")
    print("-" * 50)
    
    try:
        import paramiko
    except ImportError:
        print("安装 paramiko...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "paramiko"])
        import paramiko
    
    print("连接到远程服务器...")
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh_client.connect(args.host, username=args.user, password=args.password, timeout=30)
        print("✓ SSH 连接成功")
    except Exception as e:
        print(f"✗ SSH 连接失败: {e}")
        return 1
    
    try:
        if not check_docker_installed(ssh_client, args.sudo_password):
            if not install_docker(ssh_client, args.sudo_password):
                print("✗ Docker 安装失败")
                return 1
        
        tar_buffer = create_project_tar(project_path)
        if not upload_and_extract(ssh_client, tar_buffer, args.remote_path):
            return 1
        
        stop_existing_containers(ssh_client, args.remote_path, args.sudo_password)
        
        if not update_docker_compose_for_host(ssh_client, args.remote_path, args.port, args.network):
            return 1
        
        if not build_and_start(ssh_client, args.remote_path, args.sudo_password):
            return 1
        
        wait_for_services(ssh_client, args.sudo_password)
        
        print("-" * 50)
        print("部署完成！")
        print(f"前端访问地址: http://{args.host}:{args.port}")
        print("-" * 50)
        
    finally:
        ssh_client.close()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
