#!/usr/bin/env python3
"""
QingTian 项目 Docker 自动化部署脚本

功能特性：
1. 支持本地部署和远程部署
2. 支持多种 Docker 网络模式（host、bridge、自定义网络）
3. SSH 密码认证和密钥认证（自动搜索默认密钥）
4. 开发模式（前台运行查看日志）
5. 查看服务日志
6. 重启指定服务
7. 清理部署
8. nginx.conf 挂载动态配置

命令：
  local    - 在本地部署项目
  remote   - 部署到远程服务器
  logs     - 查看服务日志
  restart  - 重启服务
  clean    - 清理部署
"""

import argparse
import subprocess
import sys
import os
import io
import time
import json
from pathlib import Path


class DeployLogger:
    @staticmethod
    def info(msg):
        print(f"[INFO] {msg}")

    @staticmethod
    def success(msg):
        print(f"\033[92m[SUCCESS] {msg}\033[0m")

    @staticmethod
    def warning(msg):
        print(f"\033[93m[WARNING] {msg}\033[0m")

    @staticmethod
    def error(msg):
        print(f"\033[91m[ERROR] {msg}\033[0m")

    @staticmethod
    def step(step_num, total, msg):
        print(f"\n\033[94m[STEP {step_num}/{total}] {msg}\033[0m")
        print("-" * 60)


def run_command(cmd, cwd=None):
    """执行本地 shell 命令"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=3600
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "命令执行超时"
    except Exception as e:
        return -1, "", str(e)


def get_default_ssh_keys():
    """获取默认的 SSH 密钥文件列表"""
    home = Path.home()
    ssh_dir = home / '.ssh'
    key_candidates = [
        ssh_dir / 'id_rsa',
        ssh_dir / 'id_ed25519',
        ssh_dir / 'id_ecdsa',
    ]
    existing_keys = [str(key) for key in key_candidates if key.exists() and key.is_file()]
    return existing_keys


def get_nginx_conf_content(listen_port=80, api_server='apiserver'):
    """生成 nginx.conf 内容"""
    return f"""server {{
    listen {listen_port};
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml+rss text/javascript;

    location / {{
        try_files $uri $uri/ /index.html;
    }}

    location /api/ {{
        proxy_pass http://{api_server}:8080/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}
}}
"""


def get_docker_compose_base():
    """获取 docker-compose.yml 基础内容"""
    return """version: '3.8'

services:
  mongodb:
    image: mongo:6
    container_name: qingtian-mongodb
    restart: unless-stopped
    volumes:
      - mongodb_data:/data/db
    networks:
      - qingtian-network

  redis:
    image: redis:7-alpine
    container_name: qingtian-redis
    restart: unless-stopped
    volumes:
      - redis_data:/data
    networks:
      - qingtian-network

  nsqlookupd:
    image: nsqio/nsq:v1.2.1
    container_name: qingtian-nsqlookupd
    restart: unless-stopped
    command: /nsqlookupd
    networks:
      - qingtian-network

  nsqd:
    image: nsqio/nsq:v1.2.1
    container_name: qingtian-nsqd
    command: /nsqd --lookupd-tcp-address=nsqlookupd:4160
    depends_on:
      - nsqlookupd
    networks:
      - qingtian-network

  minio:
    image: minio/minio:latest
    container_name: qingtian-minio
    restart: unless-stopped
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/data
    networks:
      - qingtian-network

  apiserver:
    build:
      context: .
      dockerfile: Dockerfile.backend
    container_name: qingtian-apiserver
    restart: unless-stopped
    command: ./apiserver -config configs/apiserver.yaml
    environment:
      QT_MONGO_URI: mongodb://mongodb:27017
      QT_REDIS_ADDR: redis:6379
      QT_NSQD_ADDR: nsqd:4150
      QT_NSQ_LOOKUPD_ADDR: nsqlookupd:4161
      QT_MINIO_ENDPOINT: minio:9000
      QT_JWT_SECRET: qingtian-secret-key
    depends_on:
      - mongodb
      - redis
      - nsqd
      - minio
    networks:
      - qingtian-network

  scheduler:
    build:
      context: .
      dockerfile: Dockerfile.backend
    container_name: qingtian-scheduler
    restart: unless-stopped
    command: ./scheduler -config configs/scheduler.yaml
    environment:
      QT_MONGO_URI: mongodb://mongodb:27017
      QT_REDIS_ADDR: redis:6379
      QT_NSQD_ADDR: nsqd:4150
      QT_NSQ_LOOKUPD_ADDR: nsqlookupd:4161
      QT_MINIO_ENDPOINT: minio:9000
    depends_on:
      - mongodb
      - redis
      - nsqd
      - minio
    networks:
      - qingtian-network

  agent:
    build:
      context: .
      dockerfile: Dockerfile.backend
    container_name: qingtian-agent
    restart: unless-stopped
    command: ./agent -config configs/agent.yaml
    environment:
      QT_MONGO_URI: mongodb://mongodb:27017
      QT_REDIS_ADDR: redis:6379
      QT_NSQD_ADDR: nsqd:4150
      QT_NSQ_LOOKUPD_ADDR: nsqlookupd:4161
      QT_MINIO_ENDPOINT: minio:9000
      QT_CONTROL_PLANE_URL: http://apiserver:8080
    depends_on:
      - mongodb
      - redis
      - nsqd
      - minio
    networks:
      - qingtian-network

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    container_name: qingtian-frontend
    restart: unless-stopped
    volumes:
      - ./web/nginx.conf:/etc/nginx/conf.d/default.conf:ro
    ports:
      - "${FRONTEND_PORT:-80}:80"
    depends_on:
      - apiserver
    networks:
      - qingtian-network

volumes:
  mongodb_data:
  redis_data:
  minio_data:

networks:
  qingtian-network:
    driver: bridge
"""


def configure_for_host_network(content, frontend_port):
    """配置 docker-compose.yml 为 host 网络模式"""
    content = content.replace('    networks:\n      - qingtian-network', '    network_mode: host')
    content = content.replace('\nnetworks:\n  qingtian-network:\n    driver: bridge\n', '')
    content = content.replace('    ports:\n      - "${FRONTEND_PORT:-80}:80"\n', '')

    content = content.replace('QT_MONGO_URI: mongodb://mongodb:27017', 'QT_MONGO_URI: mongodb://127.0.0.1:27017')
    content = content.replace('QT_REDIS_ADDR: redis:6379', 'QT_REDIS_ADDR: 127.0.0.1:6379')
    content = content.replace('QT_NSQD_ADDR: nsqd:4150', 'QT_NSQD_ADDR: 127.0.0.1:4150')
    content = content.replace('QT_NSQ_LOOKUPD_ADDR: nsqlookupd:4161', 'QT_NSQ_LOOKUPD_ADDR: 127.0.0.1:4161')
    content = content.replace('QT_MINIO_ENDPOINT: minio:9000', 'QT_MINIO_ENDPOINT: 127.0.0.1:9000')
    content = content.replace('QT_CONTROL_PLANE_URL: http://apiserver:8080', 'QT_CONTROL_PLANE_URL: http://127.0.0.1:8080')

    return content


def configure_for_dev_mode(content):
    """配置开发模式：前端使用 vite dev 服务，挂载源码支持热更新"""
    # 修改 frontend 配置为开发模式，用实际端口替换变量
    port = os.getenv('FRONTEND_PORT', '80')
    dev_frontend_config = f'''  frontend_dev:
    build:
      context: ./web
      dockerfile: Dockerfile.dev
    container_name: qingtian-frontend-dev
    restart: unless-stopped
    volumes:
      - ./web:/app
      - /app/node_modules
    ports:
      - {port}:5173
    environment:
      - VITE_API_HOST=http://apiserver:8080
    depends_on:
      - apiserver
    networks:
      - qingtian-network
'''
    # 替换原 frontend 配置
    lines = content.split('\n')
    new_lines = []
    in_frontend = False
    found_frontend = False
    for line in lines:
        if 'frontend:' in line and not found_frontend:
            in_frontend = True
            found_frontend = True
            new_lines.append(dev_frontend_config.rstrip())
            continue
        if in_frontend and line.strip() == '':
            in_frontend = False
            new_lines.append(line)
        if not in_frontend:
            new_lines.append(line)
    content = '\n'.join(new_lines)
    return content


def check_local_docker():
    """检查本地 Docker 是否安装并运行"""
    DeployLogger.info("检查本地 Docker 环境...")
    code, stdout, _ = run_command("docker --version")
    if code != 0:
        DeployLogger.error("本地 Docker 未安装")
        return False
    DeployLogger.success(f"Docker 版本: {stdout.strip()}")

    code, _, _ = run_command("docker info")
    if code != 0:
        DeployLogger.error("Docker 守护进程未运行")
        return False
    DeployLogger.success("本地 Docker 环境正常")
    return True


def deploy_local(args):
    """本地部署"""
    project_path = str(Path(__file__).parent)
    total_steps = 6 if args.build else 5
    current_step = 1

    DeployLogger.step(current_step, total_steps, "检查本地 Docker 环境")
    current_step += 1
    if not check_local_docker():
        return 1

    DeployLogger.step(current_step, total_steps, "停止现有容器")
    current_step += 1
    run_command("docker compose down 2>/dev/null || true", cwd=project_path)

    DeployLogger.step(current_step, total_steps, "重置配置文件")
    current_step += 1
    # 将前端端口导出为环境变量，供 docker-compose 使用（替换 ${FRONTEND_PORT:-80}）
    os.environ['FRONTEND_PORT'] = str(args.port)

    compose_content = get_docker_compose_base()
    nginx_content = get_nginx_conf_content(80, 'apiserver')

    if args.network == 'host':
        compose_content = configure_for_host_network(compose_content, args.port)
        nginx_content = get_nginx_conf_content(args.port, '127.0.0.1')

    # --dev 参数启用前端开发模式（vite 热重载）
    if args.dev:
        compose_content = configure_for_dev_mode(compose_content)

    try:
        with open(compose_path, 'w') as f:
            f.write(compose_content)
        with open(nginx_path, 'w') as f:
            f.write(nginx_content)
    except Exception as e:
        DeployLogger.error(f"写入配置文件失败: {e}")
        return 1

    if args.build:
        DeployLogger.step(current_step, total_steps, "构建 Docker 镜像")
        current_step += 1
        code, _, stderr = run_command("docker compose build", cwd=project_path)
        if code != 0:
            DeployLogger.error(f"构建失败: {stderr}")
            return 1
        DeployLogger.success("Docker 镜像构建完成")

    DeployLogger.step(current_step, total_steps, "启动服务")
    current_step += 1

    if args.dev:
        DeployLogger.success("开发模式：按 Ctrl+C 可停止服务")
        print("\n" + "=" * 60)
        print("服务日志输出：")
        print("=" * 60)
        os.system(f"cd {project_path} && docker compose up")
    else:
        code, _, stderr = run_command("docker compose up -d", cwd=project_path)
        if code != 0:
            DeployLogger.error(f"启动失败: {stderr}")
            return 1
        DeployLogger.success("本地部署完成！")
        DeployLogger.info(f"前端访问地址: http://localhost:{args.port}")
        DeployLogger.info("常用命令：")
        DeployLogger.info(f"  查看日志: python deploy_docker.py logs local")
        DeployLogger.info(f"  重启服务: python deploy_docker.py restart local")
        DeployLogger.info(f"  停止服务: python deploy_docker.py clean local")

    return 0


def logs_local(args):
    """查看本地日志"""
    project_path = str(Path(__file__).parent)
    cmd = "docker compose logs"
    if args.follow:
        cmd += " -f"
    if args.tail:
        cmd += f" --tail={args.tail}"
    if args.service:
        cmd += f" {args.service}"
    os.system(f"cd {project_path} && {cmd}")
    return 0


def restart_local(args):
    """重启本地服务"""
    project_path = str(Path(__file__).parent)
    if args.service:
        DeployLogger.info(f"重启服务: {args.service}")
        code, _, stderr = run_command(f"docker compose restart {args.service}", cwd=project_path)
    else:
        DeployLogger.info("重启所有服务")
        code, _, stderr = run_command("docker compose restart", cwd=project_path)
    if code != 0:
        DeployLogger.error(f"重启失败: {stderr}")
        return 1
    DeployLogger.success("服务重启完成")
    return 0


def clean_local(args):
    """清理本地部署"""
    project_path = str(Path(__file__).parent)
    DeployLogger.info("停止并清理容器...")
    cmd = "docker compose down"
    if args.volumes:
        cmd += " -v"
    if args.images:
        cmd += " --rmi all"
    code, _, stderr = run_command(cmd, cwd=project_path)
    if code != 0:
        DeployLogger.warning(f"清理时出现警告: {stderr[:200]}")
    DeployLogger.success("清理完成")
    return 0


def run_remote_command(ssh_client, command, sudo_password=None):
    """在远程服务器执行命令"""
    if sudo_password:
        command = f"echo '{sudo_password}' | sudo -S {command}"
    stdin, stdout, stderr = ssh_client.exec_command(command)
    exit_status = stdout.channel.recv_exit_status()
    output = stdout.read().decode('utf-8')
    error = stderr.read().decode('utf-8')
    return exit_status, output, error


def check_remote_docker(ssh_client, sudo_password):
    """检查远程 Docker 是否安装"""
    DeployLogger.info("检查远程服务器 Docker...")
    status, output, _ = run_remote_command(ssh_client, "docker --version", sudo_password)
    if status == 0:
        DeployLogger.success(f"远程 Docker 已安装: {output.strip()}")
        return True
    return False


def deploy_remote(args):
    """远程部署"""
    project_path = str(Path(__file__).parent)
    total_steps = 10 if args.local_build else 9
    current_step = 1

    try:
        import paramiko
    except ImportError:
        DeployLogger.info("安装 paramiko 依赖...")
        code, _, _ = run_command(f"{sys.executable} -m pip install paramiko")
        if code != 0:
            DeployLogger.error("安装 paramiko 失败，请手动安装: pip install paramiko")
            return 1
        import paramiko

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    connect_kwargs = {
        'hostname': args.host,
        'port': args.ssh_port,
        'username': args.user,
        'timeout': 30,
    }

    if args.password:
        connect_kwargs['password'] = args.password
    elif args.key_file:
        connect_kwargs['key_filename'] = args.key_file
        if args.key_passphrase:
            connect_kwargs['passphrase'] = args.key_passphrase
    else:
        default_keys = get_default_ssh_keys()
        if default_keys:
            DeployLogger.info(f"找到默认密钥: {', '.join(default_keys)}")
            connect_kwargs['key_filename'] = default_keys
        else:
            DeployLogger.error("未提供密码或密钥文件，且未找到默认 SSH 密钥")
            return 1

    try:
        ssh_client.connect(**connect_kwargs)
        DeployLogger.success(f"SSH 连接成功: {args.user}@{args.host}:{args.ssh_port}")
    except Exception as e:
        DeployLogger.error(f"SSH 连接失败: {e}")
        return 1

    try:
        DeployLogger.step(current_step, total_steps, "检查远程 Docker 环境")
        current_step += 1
        if not check_remote_docker(ssh_client, args.sudo_password):
            if args.install_docker:
                DeployLogger.info("正在远程安装 Docker...")
                commands = [
                    "apt-get update -y",
                    "apt-get install -y ca-certificates curl gnupg lsb-release",
                    "mkdir -p /etc/apt/keyrings",
                    "curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg",
                    "echo 'deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable' | tee /etc/apt/sources.list.d/docker.list > /dev/null",
                    "apt-get update -y",
                    "apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin",
                    "systemctl enable docker",
                    "systemctl start docker",
                ]
                for cmd in commands:
                    status, _, error = run_remote_command(ssh_client, cmd, args.sudo_password)
                    if status != 0:
                        DeployLogger.error(f"安装 Docker 失败: {error}")
                        return 1
                DeployLogger.success("Docker 安装完成")
            else:
                DeployLogger.error("远程 Docker 未安装，使用 --install-docker 参数自动安装")
                return 1

        DeployLogger.step(current_step, total_steps, "重置远程配置文件")
        current_step += 1
        compose_content = get_docker_compose_base()
        nginx_content = get_nginx_conf_content(80, 'apiserver')

        if args.network == 'host':
            compose_content = configure_for_host_network(compose_content, args.port)
            nginx_content = get_nginx_conf_content(args.port, '127.0.0.1')

        # 将前端端口导出为环境变量，供 docker-compose 使用
        os.environ['FRONTEND_PORT'] = str(args.port)

        run_remote_command(ssh_client, f"mkdir -p {args.remote_path}/web", args.sudo_password)

        sftp = ssh_client.open_sftp()
        try:
            with sftp.open(f"{args.remote_path}/docker-compose.yml", 'w') as f:
                f.write(compose_content)
            with sftp.open(f"{args.remote_path}/web/nginx.conf", 'w') as f:
                f.write(nginx_content)
        except Exception as e:
            DeployLogger.error(f"写入远程配置文件失败: {e}")
            return 1

        DeployLogger.step(current_step, total_steps, "创建项目压缩包")
        current_step += 1
        import tarfile
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
            for root, dirs, files in os.walk(project_path):
                for file in files:
                    if file.endswith(('.pyc', '.pyo')) or '.git' in root or 'node_modules' in root or file.endswith('.tar.gz'):
                        continue
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, project_path)
                    tar.add(file_path, arcname=arcname)
        tar_buffer.seek(0)

        DeployLogger.step(current_step, total_steps, "上传项目到远程服务器")
        current_step += 1
        remote_tar = f"{args.remote_path}/qingtian_project.tar.gz"
        try:
            with sftp.open(remote_tar, 'wb') as f:
                f.write(tar_buffer.read())
        except Exception as e:
            DeployLogger.error(f"上传文件失败: {e}")
            return 1

        status, _, error = run_remote_command(ssh_client, f"cd {args.remote_path} && tar -xzf qingtian_project.tar.gz", args.sudo_password)
        if status != 0:
            DeployLogger.error(f"解压失败: {error}")
            return 1
        run_remote_command(ssh_client, f"rm -f {remote_tar}", args.sudo_password)
        DeployLogger.success("项目上传并解压完成")

        DeployLogger.step(current_step, total_steps, "停止现有容器")
        current_step += 1
        run_remote_command(ssh_client, f"cd {args.remote_path} && docker compose down 2>/dev/null || true", args.sudo_password)

        if args.build:
            DeployLogger.step(current_step, total_steps, "构建 Docker 镜像")
            current_step += 1
            DeployLogger.info("正在远程构建镜像，这可能需要几分钟...")
            status, _, error = run_remote_command(ssh_client, f"cd {args.remote_path} && docker compose build", args.sudo_password)
            if status != 0:
                DeployLogger.error(f"构建失败: {error}")
                return 1
            DeployLogger.success("Docker 镜像构建完成")

        DeployLogger.step(current_step, total_steps, "启动服务")
        current_step += 1
        status, _, error = run_remote_command(ssh_client, f"cd {args.remote_path} && docker compose up -d", args.sudo_password)
        if status != 0:
            DeployLogger.error(f"启动失败: {error}")
            return 1

        DeployLogger.success("远程部署完成！")
        DeployLogger.info(f"前端访问地址: http://{args.host}:{args.port}")

    finally:
        ssh_client.close()

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='QingTian 项目 Docker 自动化部署脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：

  本地部署：
    python deploy_docker.py local
    python deploy_docker.py local --network host --port 8080
    python deploy_docker.py local --dev --no-build --port 8580 # 开发模式，前台运行

  远程部署：
    python deploy_docker.py remote --host 192.168.1.100 --user root --password yourpass
    python deploy_docker.py remote --host 192.168.1.100 --user root  # 使用默认密钥

  查看日志：
    python deploy_docker.py logs local
    python deploy_docker.py logs local --service apiserver --follow

  重启：
    python deploy_docker.py restart local
    python deploy_docker.py restart local --service frontend

  清理部署：
    python deploy_docker.py clean local
    python deploy_docker.py clean local --volumes --images
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='部署命令')

    # local 命令
    local_parser = subparsers.add_parser('local', help='在本地部署项目')
    local_parser.add_argument('--network', default='bridge', choices=['host', 'bridge'], help='Docker 网络模式')
    local_parser.add_argument('--port', type=int, default=80, help='前端访问端口')
    local_parser.add_argument('--no-build', action='store_false', dest='build', help='不构建镜像，直接启动')
    local_parser.add_argument('--dev', '--foreground', action='store_true', help='开发模式：前台运行，查看实时日志')

    # remote 命令
    remote_parser = subparsers.add_parser('remote', help='部署到远程服务器')
    remote_parser.add_argument('--host', required=True, help='目标服务器 IP 地址')
    remote_parser.add_argument('--ssh-port', type=int, default=22, help='SSH 端口')
    remote_parser.add_argument('--user', required=True, help='SSH 用户名')
    remote_parser.add_argument('--password', help='SSH 密码')
    remote_parser.add_argument('--key-file', help='SSH 私钥文件路径')
    remote_parser.add_argument('--key-passphrase', help='SSH 密钥密码')
    remote_parser.add_argument('--sudo-password', help='sudo 密码')
    remote_parser.add_argument('--network', default='bridge', choices=['host', 'bridge'], help='Docker 网络模式')
    remote_parser.add_argument('--port', type=int, default=80, help='前端访问端口')
    remote_parser.add_argument('--remote-path', default='/opt/qingtian', help='远程部署路径')
    remote_parser.add_argument('--no-build', action='store_false', dest='build', help='不构建镜像，直接启动')
    remote_parser.add_argument('--local-build', action='store_true', help='本地构建后上传（暂未实现）')
    remote_parser.add_argument('--install-docker', action='store_true', help='自动安装 Docker')

    # logs 命令
    logs_parser = subparsers.add_parser('logs', help='查看服务日志')
    logs_subparsers = logs_parser.add_subparsers(dest='target', required=True)

    logs_local_parser = logs_subparsers.add_parser('local', help='查看本地日志')
    logs_local_parser.add_argument('-f', '--follow', action='store_true', help='实时跟踪日志')
    logs_local_parser.add_argument('--tail', type=int, default=100, help='显示最后 N 行')
    logs_local_parser.add_argument('--service', help='指定服务名')

    # restart 命令
    restart_parser = subparsers.add_parser('restart', help='重启服务')
    restart_subparsers = restart_parser.add_subparsers(dest='target', required=True)

    restart_local_parser = restart_subparsers.add_parser('local', help='重启本地服务')
    restart_local_parser.add_argument('--service', help='指定服务名')

    # clean 命令
    clean_parser = subparsers.add_parser('clean', help='清理部署')
    clean_subparsers = clean_parser.add_subparsers(dest='target', required=True)

    clean_local_parser = clean_subparsers.add_parser('local', help='清理本地部署')
    clean_local_parser.add_argument('--volumes', action='store_true', help='删除数据卷')
    clean_local_parser.add_argument('--images', action='store_true', help='删除镜像')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == 'local':
        return deploy_local(args)
    elif args.command == 'remote':
        return deploy_remote(args)
    elif args.command == 'logs':
        if args.target == 'local':
            return logs_local(args)
    elif args.command == 'restart':
        if args.target == 'local':
            return restart_local(args)
    elif args.command == 'clean':
        if args.target == 'local':
            return clean_local(args)

    return 0


if __name__ == '__main__':
    sys.exit(main())
