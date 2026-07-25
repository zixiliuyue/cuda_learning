#!/usr/bin/env python3
"""
QingTian（擎天）算力调度平台 — 一键部署/更新脚本

功能：
  - 自动拉取最新代码（git pull）
  - 拉取所需 Docker 基础镜像（含 registry mirror 降级）
  - 构建后端 Go 服务与前端 Vue 应用
  - 启动/停止/重启/查看状态
  - 部署后自动创建 admin 账号并输出访问信息
  - 详细的日志记录与错误处理

用法：
  python deploy_qingtian.py deploy             首次部署
  python deploy_qingtian.py deploy --no-build  仅启动（不重新构建）
  python deploy_qingtian.py update             重新拉取代码并部署
  python deploy_qingtian.py start              启动已有服务
  python deploy_qingtian.py stop               停止服务
  python deploy_qingtian.py restart            重启服务
  python deploy_qingtian.py status             查看服务状态
  python deploy_qingtian.py logs [service]     查看日志
  python deploy_qingtian.py info               查看部署信息
  python deploy_qingtian.py clean              清理所有数据

环境变量：
  QT_ADMIN_USER    管理员用户名（默认: admin）
  QT_ADMIN_PASS    管理员密码（默认: 随机生成 16 位）
  QT_JWT_SECRET    JWT 签名密钥（默认: 自动生成）
  QT_FRONTEND_PORT 前端端口（默认: 80）
  QT_API_PORT      API 端口（默认: 8080）
  QT_MIRROR        Docker registry mirror（默认: docker.m.daocloud.io）
"""

import argparse
import json
import os
import secrets
import string
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
CREDENTIALS_FILE = PROJECT_ROOT / ".deploy_credentials.json"
OVERRIDE_FILE = PROJECT_ROOT / "docker-compose.override.yml"
LOG_DIR = PROJECT_ROOT / ".deploy_logs"

# 需要预拉取的 Docker 镜像（含 registry mirror 映射）
REQUIRED_IMAGES = {
    "mongo:6": "docker.m.daocloud.io/library/mongo:6",
    "redis:7-alpine": "docker.m.daocloud.io/library/redis:7-alpine",
    "nsqio/nsq:v1.2.1": "docker.m.daocloud.io/nsqio/nsq:v1.2.1",
    "minio/minio:latest": "docker.m.daocloud.io/minio/minio:latest",
    "golang:1.26-alpine": "docker.m.daocloud.io/library/golang:1.26-alpine",
    "alpine:latest": "docker.m.daocloud.io/library/alpine:latest",
    "node:20-alpine": "docker.m.daocloud.io/library/node:20-alpine",
    "nginx:alpine": "docker.m.daocloud.io/library/nginx:alpine",
}

# 健康检查超时（秒）
HEALTH_CHECK_TIMEOUT = 120
HEALTH_CHECK_INTERVAL = 5

# 生产部署使用的服务列表（排除仅开发用的 frontend_dev，避免端口冲突）
PROD_SERVICES = [
    "mongodb", "redis", "nsqlookupd", "nsqd", "minio",
    "apiserver", "scheduler", "agent", "frontend",
]


# ---------------------------------------------------------------------------
# 日志工具
# ---------------------------------------------------------------------------
class Log:
    """双输出日志：控制台（彩色）+ 文件（带时间戳）。"""

    _file = None
    _enabled = True

    @classmethod
    def _ensure_file(cls):
        if cls._file is None:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = LOG_DIR / f"deploy_{ts}.log"
            cls._file = open(path, "a", encoding="utf-8")
        return cls._file

    @classmethod
    def _write(cls, color: str, label: str, msg: str):
        if not cls._enabled:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        # 控制台（彩色）
        print(f"{color}[{label}]{cls._dim(ts)} {msg}\033[0m")
        # 文件
        f = cls._ensure_file()
        print(f"[{datetime.now().isoformat()}][{label}] {msg}", file=f)
        f.flush()

    @staticmethod
    def _dim(s: str) -> str:
        return f"\033[2m {s}\033[22m"

    @classmethod
    def info(cls, msg: str):    cls._write("\033[94m", "INFO", msg)      # noqa: E704
    @classmethod
    def ok(cls, msg: str):      cls._write("\033[92m", " OK ", msg)      # noqa: E704
    @classmethod
    def warn(cls, msg: str):    cls._write("\033[93m", "WARN", msg)      # noqa: E704
    @classmethod
    def err(cls, msg: str):     cls._write("\033[91m", "ERR ", msg)      # noqa: E704
    @classmethod
    def step(cls, i: int, n: int, msg: str):
        cls._write("\033[96m", f"STEP{i}/{n}", msg)

    @classmethod
    def close(cls):
        if cls._file:
            cls._file.close()
            cls._file = None


# ---------------------------------------------------------------------------
# Shell 执行
# ---------------------------------------------------------------------------
def run(cmd: str, cwd=None, timeout: int = 1800,
        check: bool = False) -> Tuple[int, str, str]:
    """执行 shell 命令，返回 (exit_code, stdout, stderr)。"""
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd or PROJECT_ROOT,
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"命令失败 (exit={result.returncode}): {cmd}\n"
                f"stderr: {(result.stderr or '')[:500]}"
            )
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        return result.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        return -1, "", "执行超时"
    except Exception as e:
        return -1, "", str(e)


def run_ok(cmd: str, **kwargs) -> bool:
    code, _, _ = run(cmd, **kwargs)
    return code == 0


def compose_down():
    """停止并移除 compose 容器（忽略「无容器」等非致命错误）。"""
    # 不使用 shell 的 `|| true`，改为 Python 忽略返回码，保证跨平台
    run("docker compose down")


# ---------------------------------------------------------------------------
# Docker 镜像预拉取（含 mirror 降级）
# ---------------------------------------------------------------------------
def _mirror_name(image: str) -> str:
    """将官方镜像名映射到 daocloud mirror。"""
    mapping = REQUIRED_IMAGES
    return mapping.get(image, image)


def pull_image(image: str) -> bool:
    """拉取镜像：若配置了 mirror 则优先用 mirror，否则直连。

    直连 Docker Hub 在国内常被墙，故对已知镜像直接走 mirror 并重打标签，
    避免每次都先等一次直连超时。
    """
    mirror = _mirror_name(image)
    # 无 mirror 映射：只能直连
    if mirror == image:
        code, _, err = run(f"docker pull {image}")
        if code != 0:
            Log.err(f"拉取 {image} 失败: {err[:200]}")
            return False
        return True
    # 有 mirror 映射：优先 mirror
    code, _, err = run(f"docker pull {mirror}")
    if code != 0:
        Log.warn(f"mirror 拉取失败，尝试直连 {image}")
        code, _, err = run(f"docker pull {image}")
        if code != 0:
            Log.err(f"拉取 {image} 失败: {err[:200]}")
            return False
        return True
    # 重新打标签为官方名，供 compose 使用
    run(f"docker tag {mirror} {image}")
    return True


def ensure_images() -> bool:
    """确保所有必需的 Docker 镜像已拉取。"""
    needed = list(REQUIRED_IMAGES.keys())
    # 检查本地已有（用双引号包裹 format，兼容 Windows cmd.exe 与 POSIX shell）
    code, out, _ = run('docker images --format "{{.Repository}}:{{.Tag}}"')
    local = set(out.splitlines()) if code == 0 else set()
    missing = [img for img in needed if img not in local]
    if not missing:
        Log.ok("所有 Docker 基础镜像已就绪")
        return True
    Log.info(f"需要拉取 {len(missing)} 个镜像: {', '.join(missing)}")
    for img in missing:
        Log.info(f"拉取 {img} ...")
        if not pull_image(img):
            return False
        Log.ok(f"{img} 就绪")
    return True


# ---------------------------------------------------------------------------
# 密码 / 密钥生成
# ---------------------------------------------------------------------------
def generate_password(length: int = 16) -> str:
    chars = string.ascii_letters + string.digits + "!@#$%^&*"
    return "".join(secrets.choice(chars) for _ in range(length))


def generate_jwt_secret() -> str:
    return secrets.token_hex(32)


# ---------------------------------------------------------------------------
# 凭证文件管理
# ---------------------------------------------------------------------------
def load_credentials() -> dict:
    if CREDENTIALS_FILE.exists():
        try:
            return json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_credentials(data: dict):
    CREDENTIALS_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # 确保 .gitignore 包含此文件
    gitignore = PROJECT_ROOT / ".gitignore"
    entry = ".deploy_credentials.json"
    if gitignore.exists():
        content = gitignore.read_text(encoding="utf-8")
        if entry not in content:
            gitignore.write_text(content.rstrip() + "\n" + entry + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Compose override 管理
# ---------------------------------------------------------------------------
def generate_override(frontend_port: int, api_port: int, jwt_secret: str) -> str:
    """生成 docker-compose.override.yml，不修改原始 docker-compose.yml。"""
    return f"""# 自动生成 — 请勿手动编辑
# 由 deploy_qingtian.py 管理，deploy/update 时自动生成

services:
  apiserver:
    ports:
      - "{api_port}:8080"
    environment:
      QT_JWT_SECRET: "{jwt_secret}"

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    container_name: qingtian-frontend
    restart: unless-stopped
    ports:
      - "{frontend_port}:80"
    depends_on:
      - apiserver
    networks:
      - qingtian-network
"""


def write_override(frontend_port: int, api_port: int, jwt_secret: str):
    content = generate_override(frontend_port, api_port, jwt_secret)
    OVERRIDE_FILE.write_text(content, encoding="utf-8")
    Log.info(f"已生成 {OVERRIDE_FILE.name}")


def remove_override():
    if OVERRIDE_FILE.exists():
        OVERRIDE_FILE.unlink()
        Log.info("已移除 override 文件")


# ---------------------------------------------------------------------------
# 服务健康检查
# ---------------------------------------------------------------------------
def wait_for_health(api_port: int, timeout: int = HEALTH_CHECK_TIMEOUT) -> bool:
    """等待 API 服务健康检查通过。"""
    url = f"http://localhost:{api_port}/healthz"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            if resp.status == 200:
                return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass
        time.sleep(HEALTH_CHECK_INTERVAL)
    return False


# ---------------------------------------------------------------------------
# 管理员账号创建
# ---------------------------------------------------------------------------
def ensure_admin(api_port: int, username: str, password: str) -> bool:
    """通过 API 注册管理员账号。若已存在则尝试登录验证。"""
    base = f"http://localhost:{api_port}/api/v1/auth"

    # 尝试注册
    body = json.dumps({
        "username": username, "password": password, "role": "admin"
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/register", data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read().decode("utf-8"))
        if resp.status == 201 or data.get("code") == "OK":
            Log.ok(f"管理员账号 '{username}' 创建成功")
            return True
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        # 409 / 用户名已存在 — 尝试登录验证
        if e.code in (409, 400) and "已存在" in body:
            Log.info("管理员账号已存在，验证密码...")
            return verify_admin_password(api_port, username, password)
        Log.err(f"注册失败 (HTTP {e.code}): {body[:200]}")
        return False
    except Exception as e:
        Log.err(f"注册请求异常: {e}")
        return False
    return False


def verify_admin_password(api_port: int, username: str, password: str) -> bool:
    """尝试登录验证密码是否正确。"""
    body = json.dumps({"username": username, "password": password}).encode("utf-8")
    req = urllib.request.Request(
        f"http://localhost:{api_port}/api/v1/auth/login",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        if resp.status == 200:
            Log.ok("密码验证通过")
            return True
    except urllib.error.HTTPError:
        pass
    Log.err("密码验证失败 — 密码可能已被修改，请检查 credentials 文件")
    return False


# ---------------------------------------------------------------------------
# 部署状态
# ---------------------------------------------------------------------------
def get_deploy_status() -> dict:
    """获取当前部署状态。"""
    code, out, _ = run("docker compose ps --format json", timeout=30)
    services = {}
    if code == 0:
        for line in out.splitlines():
            try:
                item = json.loads(line)
                services[item.get("Name", "?")] = {
                    "status": item.get("Status", "?"),
                    "state": item.get("State", "?"),
                    "ports": item.get("Ports", ""),
                }
            except json.JSONDecodeError:
                pass
    creds = load_credentials()
    return {"services": services, "credentials": creds}


# ---------------------------------------------------------------------------
# 部署步骤
# ---------------------------------------------------------------------------
def step_git_pull() -> bool:
    """拉取最新代码。"""
    Log.info("检查 git 仓库状态...")
    code, out, _ = run("git status --porcelain")
    if code != 0:
        Log.warn("不是 git 仓库或 git 不可用，跳过代码拉取")
        return True
    if out:
        Log.warn("工作区有未提交的修改，跳过 git pull（请手动处理）")
        Log.info("未提交文件:\n" + out[:500])
        return True
    code, out, err = run("git pull --ff-only")
    if code != 0:
        Log.err(f"git pull 失败: {err[:200]}")
        return False
    Log.ok(f"代码已更新: {out[:200]}")
    return True


def step_build() -> bool:
    """构建后端 + 前端镜像。前端构建失败不影响后端部署，仅告警。"""
    Log.info("构建 Docker 镜像（Go 后端）...")
    backend_svcs = [s for s in PROD_SERVICES if s != "frontend"]
    svc_list = " ".join(backend_svcs)
    code, _, err = run(f"docker compose build {svc_list}")
    if code != 0:
        Log.err(f"后端构建失败: {err[:500]}")
        return False
    Log.ok("后端镜像构建完成")

    # 构建前端（Vue SPA）
    Log.info("构建前端镜像...")
    code, _, err = run("docker compose build frontend")
    if code != 0:
        Log.warn("前端构建失败，不影响后端运行")
        Log.warn("可执行 npm run build 查看详细错误")
    else:
        Log.ok("前端镜像构建完成")
    return True


def step_start() -> bool:
    """启动服务（仅后端，前端可选）。"""
    # 先停止旧容器
    compose_down()
    Log.info("启动后端服务...")
    backend_svcs = [s for s in PROD_SERVICES if s != "frontend"]
    svc_list = " ".join(backend_svcs)
    code, _, err = run(f"docker compose up -d {svc_list}")
    if code != 0:
        Log.err(f"启动失败: {err[:500]}")
        return False
    Log.ok("后端服务已启动")

    # 尝试启动前端（镜像可能不存在）
    code, _, err = run("docker compose up -d frontend")
    if code != 0:
        Log.warn("前端容器未启动（镜像未构建），仅后端可用")
    else:
        Log.ok("前端服务已启动")
    return True


def step_health_check(api_port: int) -> bool:
    """等待并检查 API 健康状态。"""
    Log.info("等待 API 服务就绪...")
    if wait_for_health(api_port):
        Log.ok("API 服务已就绪")
        return True
    Log.err("API 服务未在超时时间内就绪，请检查日志")
    return False


def step_create_admin(api_port: int, username: str, password: str) -> bool:
    """创建或验证管理员账号。"""
    return ensure_admin(api_port, username, password)


# ---------------------------------------------------------------------------
# 命令实现
# ---------------------------------------------------------------------------
def cmd_deploy(args):
    """首次部署。"""
    total_steps = 7
    Log.step(1, total_steps, "环境检查")
    if not run_ok("docker info"):
        Log.err("Docker 未运行，请先启动 Docker Desktop / Rancher Desktop")
        return 1
    Log.ok("Docker 环境正常")

    Log.step(2, total_steps, "拉取基础镜像")
    if not ensure_images():
        Log.err("基础镜像拉取失败，请检查网络")
        return 1

    if args.git_pull:
        Log.step(3, total_steps, "拉取最新代码")
        if not step_git_pull():
            Log.warn("git pull 失败，继续使用当前代码")

    Log.step(4, total_steps, "生成配置")
    frontend_port = args.frontend_port
    api_port = args.api_port
    # 复用已有凭证：避免每次更新都重置 JWT 密钥和管理员密码
    existing = load_credentials()
    jwt_secret = (os.environ.get("QT_JWT_SECRET")
                  or existing.get("jwt_secret")
                  or generate_jwt_secret())
    write_override(frontend_port, api_port, jwt_secret)

    if args.build:
        Log.step(5, total_steps, "构建镜像")
        if not step_build():
            return 1
    else:
        Log.step(5, total_steps, "跳过构建")

    Log.step(6, total_steps, "启动服务")
    if not step_start():
        return 1

    Log.step(7, total_steps, "健康检查 & 初始化")
    if not step_health_check(api_port):
        return 1

    # 创建管理员
    admin_user = args.admin_user
    # 复用已验证的历史密码，保证「显示的密码真实可用」
    admin_pass = (os.environ.get("QT_ADMIN_PASS")
                  or existing.get("admin_password")
                  or generate_password())
    admin_ok = step_create_admin(api_port, admin_user, admin_pass)
    if not admin_ok:
        Log.warn("管理员密码无法验证：账号可能已用其它密码创建")
        Log.warn("如需重置，请执行 clean 后重新部署，或手动修改数据库")

    # 保存凭证
    creds = {
        "admin_user": admin_user,
        "admin_password": admin_pass if admin_ok else "<未验证-见告警>",
        "admin_verified": admin_ok,
        "jwt_secret": jwt_secret,
        "frontend_port": frontend_port,
        "api_port": api_port,
        "frontend_url": f"http://localhost:{frontend_port}",
        "api_url": f"http://localhost:{api_port}",
        "deployed_at": datetime.now().isoformat(),
    }
    save_credentials(creds)

    Log.info("=" * 55)
    Log.ok("部署完成！")
    Log.info("=" * 55)
    Log.info(f"前端地址:    \033[4mhttp://localhost:{frontend_port}\033[0m")
    Log.info(f"API 地址:    \033[4mhttp://localhost:{api_port}\033[0m")
    Log.info(f"管理员账号:  {admin_user}")
    if admin_ok:
        Log.info(f"管理员密码:  \033[93m{admin_pass}\033[0m  (已验证可登录)")
    else:
        Log.warn("管理员密码:  未能验证，请见上方告警")
    Log.info("=" * 55)
    Log.info(f"凭证已保存至: {CREDENTIALS_FILE}")
    Log.info("可用命令: python deploy_qingtian.py status | logs | info")
    return 0


def cmd_update(args):
    """更新部署：拉取代码 + 重新构建 + 滚动重启。"""
    args.git_pull = True
    args.build = True
    return cmd_deploy(args)


def cmd_start(args):
    """启动已有服务。"""
    if not OVERRIDE_FILE.exists():
        Log.warn("未找到 override 文件，请先执行 deploy 或 info 命令初始化")
        return 1
    if not step_start():
        return 1
    # 获取端口信息
    creds = load_credentials()
    api_port = creds.get("api_port", 8080)
    if not step_health_check(api_port):
        return 1
    Log.ok("服务已启动")
    return 0


def cmd_stop(args):
    """停止服务。"""
    Log.info("停止服务...")
    code, _, err = run("docker compose down")
    if code != 0:
        Log.err(f"停止失败: {err[:200]}")
        return 1
    Log.ok("服务已停止")
    return 0


def cmd_restart(args):
    """重启服务。"""
    Log.info("重启服务...")
    code, _, err = run("docker compose restart")
    if code != 0:
        Log.err(f"重启失败: {err[:200]}")
        return 1
    Log.ok("服务已重启")
    return 0


def cmd_status(args):
    """查看服务状态。"""
    status = get_deploy_status()
    if not status["services"]:
        Log.warn("没有运行中的服务（可能未部署）")
        return 1
    Log.info("服务状态:")
    print("-" * 70)
    print(f"  {'名称':<30} {'状态':<20} {'端口':<20}")
    print("-" * 70)
    for name, info in status["services"].items():
        print(f"  {name:<30} {info['state']:<20} {info['ports']:<20}")
    print("-" * 70)
    return 0


def cmd_logs(args):
    """查看服务日志。"""
    svc = args.service or ""
    follow = "-f" if args.follow else ""
    tail = f"--tail={args.tail}"
    cmd = f"docker compose logs {follow} {tail} {svc}"
    # 用 subprocess 在项目目录执行，直接透传输出到终端（跨平台）
    subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)
    return 0


def cmd_info(args):
    """查看部署信息。"""
    creds = load_credentials()
    if not creds:
        Log.warn("未找到部署记录，请先执行 deploy 命令")
        return 1
    print()
    print("=" * 55)
    print("  QingTian 部署信息")
    print("=" * 55)
    print(f"  前端地址:     {creds.get('frontend_url', '?')}")
    print(f"  API 地址:     {creds.get('api_url', '?')}")
    print(f"  管理员账号:   {creds.get('admin_user', '?')}")
    print(f"  管理员密码:   {creds.get('admin_password', '?')}")
    print(f"  部署时间:     {creds.get('deployed_at', '?')}")
    print("=" * 55)
    print()
    return 0


def cmd_clean(args):
    """清理所有数据。"""
    if args.yes is False:
        confirm = input("确定要删除所有容器、数据卷和镜像？(yes/no): ")
        if confirm.lower() not in ("yes", "y"):
            Log.info("已取消")
            return 0
    Log.info("停止并删除容器...")
    run("docker compose down -v")
    remove_override()
    if args.images:
        Log.info("删除 QingTian 镜像...")
        run("docker rmi qingtian-apiserver qingtian-scheduler qingtian-agent qingtian-frontend")
    if args.credentials:
        if CREDENTIALS_FILE.exists():
            CREDENTIALS_FILE.unlink()
    Log.ok("清理完成")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="QingTian 一键部署/更新脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # deploy
    p = sub.add_parser("deploy", help="首次部署")
    p.add_argument("--no-build", action="store_false", dest="build", help="跳过 Docker 构建")
    p.add_argument("--no-git-pull", action="store_false", dest="git_pull", help="跳过 git pull")
    p.add_argument("--frontend-port", type=int, default=80, help="前端端口（默认 80）")
    p.add_argument("--api-port", type=int, default=8080, help="API 端口（默认 8080）")
    p.add_argument("--admin-user", default="admin", help="管理员用户名（默认 admin）")
    p.set_defaults(func=cmd_deploy)

    # update
    p = sub.add_parser("update", help="更新部署（git pull + 构建 + 重启）")
    p.add_argument("--frontend-port", type=int, default=80)
    p.add_argument("--api-port", type=int, default=8080)
    p.add_argument("--admin-user", default="admin")
    p.set_defaults(func=cmd_update)

    # start / stop / restart
    for name in ("start", "stop", "restart"):
        p = sub.add_parser(name, help=f"{name} 服务")
        p.set_defaults(func=globals()[f"cmd_{name}"])

    # status
    sub.add_parser("status", help="查看服务状态").set_defaults(func=cmd_status)

    # logs
    p = sub.add_parser("logs", help="查看日志")
    p.add_argument("service", nargs="?", default="", help="服务名，如 apiserver")
    p.add_argument("-f", "--follow", action="store_true", help="实时跟踪")
    p.add_argument("--tail", type=int, default=100, help="行数（默认 100）")
    p.set_defaults(func=cmd_logs)

    # info
    sub.add_parser("info", help="查看部署信息").set_defaults(func=cmd_info)

    # clean
    p = sub.add_parser("clean", help="清理所有数据")
    p.add_argument("--yes", "-y", action="store_true", help="跳过确认")
    p.add_argument("--images", action="store_true", help="同时删除镜像")
    p.add_argument("--credentials", action="store_true", help="同时删除凭证")
    p.set_defaults(func=cmd_clean)

    args = parser.parse_args()

    try:
        exit_code = args.func(args)
        sys.exit(exit_code or 0)
    except KeyboardInterrupt:
        Log.info("用户中断")
        sys.exit(1)
    except Exception as e:
        Log.err(f"未预期异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        Log.close()


if __name__ == "__main__":
    main()