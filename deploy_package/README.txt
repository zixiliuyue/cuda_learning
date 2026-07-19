
QingTian 项目部署说明
============================================================

部署包位置: F:\code\cuda_learning\deploy_package

快速开始
============================================================

方法一: 使用批处理文件（推荐）
1. 双击 "上传文件.bat"
2. 输入密码: 123
3. SSH 登录服务器执行部署

方法二: 手动执行
1. 上传文件:
   scp qingtian.tar.gz sen@192.168.10.82:/opt/qingtian/
   
2. SSH 登录:
   ssh sen@192.168.10.82
   密码: 123
   
3. 执行部署:
   cd /opt/qingtian
   tar -xzf qingtian.tar.gz
   echo '123' | sudo -S docker compose up -d --build

服务器信息
============================================================
地址: 192.168.10.82
用户: sen
密码: 123
部署目录: /opt/qingtian

访问地址
============================================================
前端: http://192.168.10.82

默认账号
============================================================
用户名: admin
密码: admin123

Docker 常用命令
============================================================
# 查看服务状态
sudo docker compose ps

# 查看日志
sudo docker compose logs -f

# 重启服务
sudo docker compose restart

# 停止服务
sudo docker compose down
