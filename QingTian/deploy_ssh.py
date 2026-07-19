#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import tarfile
import io
import time
from pathlib import Path
import tempfile

def run_remote_cmd(host, user, password, cmd, sudo=False):
    if sudo:
        full_cmd = f"sshpass -p '{password}' ssh -o StrictHostKeyChecking=no {user}@{host} \"echo '{password}' | sudo -S {cmd}\""
    else:
        full_cmd = f"sshpass -p '{password}' ssh -o StrictHostKeyChecking=no {user}@{host} \"{cmd}\""
    
    try:
        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def check_sshpass():
    try:
        result = subprocess.run("sshpass -V", shell=True, capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def check_docker_installed(host, user, password):
    print("检查 Docker 是否安装...")
    status, output, error = run_remote_cmd(host, user, password, "docker --version")
    if status == 0:
        print(f"✓ Docker 已安装: {output.strip()}")
        return True
    return False

def install_docker(host, user, password):
    print("开始安装 Docker...")
    commands = [
        "apt-get update",
        "apt-get install -y ca-certificates curl gnupg lsb-release",
        "mkdir -p /etc/apt/keyrings",
        "curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg",
        "echo 'deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable' | tee /etc/apt/sources.list.d/docker.list > /dev/null",
        "apt-get update",
        "apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin",
        "systemctl enable docker",
        "systemctl start docker"
    ]
    
    for cmd in commands:
        status, output, error = run_remote_cmd(host, user, password, cmd, sudo=True)
        if status != 0:
            print(f"✗ 命令执行失败: {cmd[:50]}...")
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
                if '__pycache__' in root:
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, project_path)
                tar.add(file_path, arcname=arcname)
    tar_buffer.seek(0)
    print("✓ 项目压缩包创建完成")
    return tar_buffer

def upload_and_extract(host, user, password, tar_buffer, remote_path):
    print(f"上传项目到远程服务器: {remote_path}")
    
    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
        tmp.write(tar_buffer.read())
        tmp_path = tmp.name
    
    try:
        run_remote_cmd(host, user, password, f"mkdir -p {remote_path}")
        
        scp_cmd = f"sshpass -p '{password}' scp -o StrictHostKeyChecking=no {tmp_path} {user}@{host}:{remote_path}/qingtian_project.tar.gz"
        result = subprocess.run(scp_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"✗ 上传失败: {result.stderr}")
            return False
        
        status, output, error = run_remote_cmd(host, user, password, f"cd {remote_path} && tar -xzf qingtian_project.tar.gz")
        if status != 0:
            print(f"✗ 解压失败: {error}")
            return False
        
        run_remote_cmd(host, user