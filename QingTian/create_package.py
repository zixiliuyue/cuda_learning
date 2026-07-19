#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建持久的项目压缩包用于部署
"""

import os
import tarfile
import shutil
import tempfile

# 配置
LOCAL_PROJECT_PATH = r"F:\code\cuda_learning\QingTian"
OUTPUT_DIR = r"F:\code\cuda_learning\deploy_package"
SERVER = "192.168.10.82"
USER = "sen"
PASSWORD = "123"
REMOTE_PATH = "/opt/qingtian"

def main():
    print("创建 QingTian 部署包...")
    
    # 创建输出目录
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    # 复制项目文件
    print("复制项目文件...")
    dest_dir = os.path.join(OUTPUT_DIR, "QingTian")
    shutil.copytree(LOCAL_PROJECT_PATH, dest_dir,
                   ignore=shutil.ignore_patterns('.git', 'node_modules', '__pycache__', 
                                                 '*.pyc', 'deploy_*.py', '*.ps1', '*.vbs'))
    
    # 替换 docker-compose.yml
    host_compose = os.path.join(LOCAL_PROJECT_PATH, "docker-compose-host.yml")
    target_compose = os.path.join(dest_dir, "docker-compose.yml")
    shutil.copy2(host_compose, target_compose)
    print("✓ 已替换 docker-compose.yml 为 host 网络模式")
    
    # 创建压缩包
    tar_path = os.path.join(OUTPUT_DIR, "qingtian.tar.gz")
    print(f"创建压缩包: {tar_path}")
    
    with tarfile.open(tar_path, "w:gz") as tar:
        for root, dirs, files in os.walk(dest_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, dest_dir)
                tar.add(file_path, arcname=arcname)
    
    file_size = os.path.getsize(tar_path) / (1024 * 1024)
    print(f"✓ 压缩包创建完成，大小: {file_size:.2f} MB")
    
    # 创建部署脚本
    deploy_script = os.path.join(OUTPUT_DIR, "deploy.sh")
    with open(deploy_script, 'w', encoding='utf-8') as f:
        f.write(f"""#!/bin/bash
# QingTian 自动部署脚本

REMOTE_PATH="{REMOTE_PATH}"
PASSWORD="{PASSWORD}"

echo "开始部署 QingTian 项目..."

# 创建目录
mkdir -p $REMOTE_PATH

# 解压
echo "解压文件..."
tar -xzf qingtian.tar.gz -C $REMOTE_PATH

# 进入目录
cd $REMOTE_PATH

# 启动服务
echo "启动 Docker Compose..."
echo $PASSWORD | sudo -S docker compose up -d --build

echo "等待服务启动..."
sleep 10

# 检查状态
echo "服务状态:"
echo $PASSWORD | sudo -S docker compose ps

echo ""
echo "部署完成!"
echo "前端地址: http://{SERVER}"
""")
    
    print(f"✓ 部署脚本已创建")
    
    # 创建 Windows 批处理文件
    bat_file = os.path.join(OUTPUT_DIR, "上传文件.bat")
    with open(bat_file, 'w', encoding='gbk') as f:
        f.write(f"""@echo off
chcp 65001 >nul
echo ========================================
echo   QingTian 项目文件上传工具
echo ========================================
echo.

echo [1/3] 连接服务器并创建目录...
ssh -o StrictHostKeyChecking=no {USER}@{SERVER} "mkdir -p {REMOTE_PATH}"

echo.
echo [2/3] 上传压缩包...
scp qingtian.tar.gz {USER}@{SERVER}:{REMOTE_PATH}/

echo.
echo [3/3] 上传部署脚本...
scp deploy.sh {USER}@{SERVER}:{REMOTE_PATH}/

echo.
echo ========================================
echo   文件上传完成！
echo ========================================
echo.
echo 接下来请 SSH 登录服务器执行部署:
echo   ssh {USER}@{SERVER}
echo   cd {REMOTE_PATH}
echo   chmod +x deploy.sh
echo   bash deploy.sh
echo.
echo 服务器密码: {PASSWORD}
echo.
pause
""")
    
    print(f"✓ 上传脚本已创建")
    
    # 创建部署说明
    readme = os.path.join(OUTPUT_DIR, "README.txt")
    with open(readme, 'w', encoding='utf-8') as f:
        f.write(f"""
QingTian 项目部署说明
{'='*60}

部署包位置: {OUTPUT_DIR}

快速开始
{'='*60}

方法一: 使用批处理文件（推荐）
1. 双击 "上传文件.bat"
2. 输入密码: {PASSWORD}
3. SSH 登录服务器执行部署

方法二: 手动执行
1. 上传文件:
   scp qingtian.tar.gz {USER}@{SERVER}:{REMOTE_PATH}/
   
2. SSH 登录:
   ssh {USER}@{SERVER}
   密码: {PASSWORD}
   
3. 执行部署:
   cd {REMOTE_PATH}
   tar -xzf qingtian.tar.gz
   echo '{PASSWORD}' | sudo -S docker compose up -d --build

服务器信息
{'='*60}
地址: {SERVER}
用户: {USER}
密码: {PASSWORD}
部署目录: {REMOTE_PATH}

访问地址
{'='*60}
前端: http://{SERVER}

默认账号
{'='*60}
用户名: admin
密码: admin123

Docker 常用命令
{'='*60}
# 查看服务状态
sudo docker compose ps

# 查看日志
sudo docker compose logs -f

# 重启服务
sudo docker compose restart

# 停止服务
sudo docker compose down
""")
    
    print(f"✓ 部署说明已创建")
    
    print("\n" + "=" * 60)
    print("  部署包创建成功！")
    print("=" * 60)
    print(f"\n部署包位置: {OUTPUT_DIR}")
    print("\n文件列表:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"  - {f}")
    
    print(f"\n接下来请执行:")
    print(f"1. 打开目录: {OUTPUT_DIR}")
    print(f"2. 双击 '上传文件.bat'")
    print(f"3. 按照提示完成文件上传")
    print(f"4. SSH 登录服务器执行部署")
    
    # 打开目录
    try:
        os.startfile(OUTPUT_DIR)
    except:
        pass

if __name__ == "__main__":
    main()