#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QingTian 项目 Docker 自动部署脚本
支持 Windows 环境，无需额外依赖
"""

import subprocess
import sys
import os
import tarfile
import io
import time
import socket
from pathlib import Path
import tempfile
import getpass

# 配置信息
SERVER = "192.168.10.82"
USER = "sen"
PASSWORD = "123"
LOCAL_PROJECT_PATH = r"F:\code\cuda_learning\QingTian"
REMOTE_PATH = "/opt/qingtian"

def print_step(step, total, title):
    """打印步骤标题"""
    print("\n" + "=" * 60)
    print(f"  [步骤 {step}/{total}] {title}")
    print("=" * 60)

def print_success(msg):
    print(f"✓ {msg}")

def print_error(msg):
    print(f"✗ {msg}")

def print_info(msg):
    print(f"  {msg}")

def run_remote(host, user, password, command, sudo=False, timeout=30):
    """
    使用 subprocess 和管道执行远程命令
    注意：这个方法在 Windows 上需要 sshpass 或其他工具
    """
    # 尝试使用 plink (PuTTY)
    try:
        if sudo:
            cmd = f'plink -ssh {user}@{host} -pw {password} "echo {password} | sudo -S {command}"'
        else:
            cmd = f'plink -ssh {user}@{host} -pw {password} "{command}"'
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def check_ssh_available():
    """检查 SSH 相关工具是否可用"""
    tools = {}
    
    # 检查 ssh
    try:
        result = subprocess.run("ssh -V", shell=True, capture_output=True, text=True)
        tools['ssh'] = result.returncode == 0
    except:
        tools['ssh'] = False
    
    # 检查 scp
    try:
        result = subprocess.run("scp", shell=True, capture_output=True, text=True)
        tools['scp'] = True
    except:
        tools['scp'] = False
    
    # 检查 plink
    try:
        result = subprocess.run("plink -V", shell=True, capture_output=True, text=True)
        tools['plink'] = result.returncode == 0
    except:
        tools['plink'] = False
    
    return tools

def create_tar_archive(source_dir, output_tar):
    """创建 tar 压缩包"""
    print_info(f"正在创建项目压缩包: {output_tar}")
    
    exclude_patterns = ['.git', 'node_modules', '__pycache__', '.pyc', '.pyo', '.tar.gz', '.zip']
    
    with tarfile.open(output_tar, "w:gz") as tar:
        for root, dirs, files in os.walk(source_dir):
            # 排除不需要的目录
            dirs[:] = [d for d in dirs if d not in exclude_patterns and not d.startswith('.')]
            
            for file in files:
                # 排除不需要的文件
                if any(pattern in file for pattern in exclude_patterns):
                    continue
                
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                tar.add(file_path, arcname=arcname)
    
    print_success("项目压缩包创建完成")
    return True

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║           QingTian 项目 Docker 部署工具                       ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 检查工具
    print_step(1, 6, "检查系统环境和 SSH 工具")
    tools = check_ssh_available()
    print_info(f"SSH 工具状态: ssh={tools['ssh']}, scp={tools['scp']}, plink={tools['plink']}")
    
    if not tools['ssh'] or not tools['scp']:
        print_error("SSH 或 SCP 工具不可用")
        print_info("请确保已安装 OpenSSH 客户端")
        print_info("Windows 10/11 可通过设置 -> 应用 -> 可选功能安装 OpenSSH 客户端")
        return False
    
    print_success("系统环境检查完成")
    
    # 准备本地文件
    print_step(2, 6, "准备本地项目文件")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        print_info(f"临时目录: {temp_dir}")
        
        # 复制项目文件
        import shutil
        print_info("复制项目文件...")
        shutil.copytree(LOCAL_PROJECT_PATH, os.path.join(temp_dir, "QingTian"), 
                       ignore=shutil.ignore_patterns('.git', 'node_modules', '__pycache__', '*.pyc'))
        
        # 使用 host 网络模式的 docker-compose.yml
        host_compose = os.path.join(LOCAL_PROJECT_PATH, "docker-compose-host.yml")
        target_compose = os.path.join(temp_dir, "QingTian", "docker-compose.yml")
        shutil.copy2(host_compose, target_compose)
        print_success("已替换 docker-compose.yml 为 host 网络模式")
        
        # 创建压缩包
        tar_path = os.path.join(temp_dir, "qingtian.tar.gz")
        create_tar_archive(os.path.join(temp_dir, "QingTian"), tar_path)
        
        file_size = os.path.getsize(tar_path) / (1024 * 1024)
        print_info(f"压缩包大小: {file_size:.2f} MB")
        
        print_success("本地文件准备完成")
        
        print("\n" + "=" * 60)
        print("  接下来需要手动执行以下操作")
        print("=" * 60)
        
        print(f"""
请按照以下步骤手动完成部署：

1. 上传文件到服务器:
   scp "{tar_path}" {USER}@{SERVER}:{REMOTE_PATH}/

2. SSH 登录服务器:
   ssh {USER}@{SERVER}
   密码: {PASSWORD}

3. 在服务器上执行:
   cd {REMOTE_PATH}
   mkdir -p {REMOTE_PATH}
   tar -xzf qingtian.tar.gz
   echo '{PASSWORD}' | sudo -S docker compose up -d --build

4. 等待构建完成后，检查服务状态:
   echo '{PASSWORD}' | sudo -S docker compose ps

5. 访问前端:
   http://{SERVER}

注意:
- 如果服务器上没有 Docker，请先安装 Docker
- 构建过程可能需要 5-15 分钟
- 所有服务使用 host 网络模式，无需端口映射
""")
        
        # 保存部署说明
        deploy_instructions = os.path.join(LOCAL_PROJECT_PATH, "部署说明.txt")
        with open(deploy_instructions, 'w', encoding='utf-8') as f:
            f.write(f"""
QingTian 项目部署说明
{'='*60}

服务器信息:
- 地址: {SERVER}
- 用户: {USER}
- 密码: {PASSWORD}
- 部署目录: {REMOTE_PATH}

快速部署命令:
{'='*60}

# 1. 上传文件
scp "{tar_path}" {USER}@{SERVER}:{REMOTE_PATH}/

# 2. SSH 登录
ssh {USER}@{SERVER}

# 3. 解压并部署（在服务器上执行）
cd {REMOTE_PATH}
mkdir -p {REMOTE_PATH}
tar -xzf qingtian.tar.gz

# 4. 启动服务（需要 sudo）
echo '{PASSWORD}' | sudo -S docker compose up -d --build

# 5. 检查服务状态
echo '{PASSWORD}' | sudo -S docker compose ps

# 6. 查看日志
echo '{PASSWORD}' | sudo -S docker compose logs -f

访问地址
{'='*60}
前端地址: http://{SERVER}

默认账号
{'='*60}
用户名: admin
密码: admin123
""")
        
        print_success(f"部署说明已保存到: {deploy_instructions}")
        
        # 尝试打开文件所在目录
        try:
            os.startfile(os.path.dirname(deploy_instructions))
        except:
            pass
    
    print("\n" + "=" * 60)
    print("  部署准备完成！")
    print("=" * 60)
    print("\n请按照上述说明完成部署操作。")
    print(f"\n前端访问地址: http://{SERVER}")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n用户中断部署")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n部署过程发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)