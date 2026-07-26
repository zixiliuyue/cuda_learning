# GitHub 开源合规性与质量评估报告

- **评估对象**：`f:\code\cuda_learning\`（CUDA 40/90 天学习项目 + QingTian 算力调度平台）
- **远端仓库**：`github.com/zixiliuyue/cuda_learning`（公开）
- **评估依据**：字节跳动《开源项目管理规范》《代码安全标准》《编码规范》及开源社区最佳实践
- **评估结论**：**通过（有条件）** —— 工作区问题已全部整改并验证；唯一遗留项为 Git 历史凭证清理，需仓库所有者手动执行（见第七节）。

---

## 一、检查范围与方法

| 维度 | 方法 | 工具/手段 |
|------|------|-----------|
| 敏感信息 | 全量逐行模式扫描 + 人工复核 | Grep 正则（密钥/令牌/IP/凭证/PII）、check-secrets.sh |
| 依赖合规 | 清单梳理 + 许可核对 | go.mod/go.sum、package.json、requirements.txt |
| 代码安全 | 数据流追踪（source→sink）+ 人工审计 | 子代理专项审计、go vet |
| 代码质量 | 静态检查 + 单元测试 | go vet、go test -race |
| 文档规范 | 完整性/准确性核对 | README/LICENSE/注释审查 |
| 第三方资源 | 资产盘点 | 图片/字体/音视频排查 |

---

## 二、敏感信息审计

### 发现（严重）
公开仓库中硬编码并已提交真实内网凭证：

| 类别 | 内容 | 分布 |
|------|------|------|
| 内网服务器 IP | `192.168.10.82` | deploy_package/ 约 40 处、根目录 3 个 SSH 脚本 |
| SSH 用户名 | `sen` | 同上 |
| SSH 明文密码 | `123` | 20+ 部署脚本 |
| 应用默认密码 | `admin123` | 部署报告/说明文档 |
| PII 本地路径 | `C:\Users\sen\AppData\...` | 部署说明.txt |

### 整改（已完成）
- ✅ 删除整个 `deploy_package/`（151 文件，含全套明文凭证 + 项目副本）
- ✅ 删除 `test_conn.ps1`、`test_ssh.ps1`、`ssh_test.vbs`
- ✅ 全树复扫 `192.168.10.82`/`sen@`/`admin123` → **零残留**
- ✅ 深度扫描私钥/AWS Key/GitHub Token/Slack/JWT/个人邮箱 → **未发现**
- ✅ 配置文件仅含 `localhost`、`minioadmin`（开发占位）、`change-me-in-production`（启动即拒绝）

### 长效机制（已建立）
- ✅ 根 `.gitignore` 新增 `.env*`、`*.pem/*.key/*.p12`、`*_credentials.json`、`deploy_package/` 等过滤规则
- ✅ CI（`.github/workflows/ci.yml`）集成 `check-secrets.sh --all` + `gitleaks` 双重扫描
- ✅ QingTian 已提供 `scripts/check-secrets.sh` pre-commit 钩子

---

## 三、依赖项合规性

详见 [DEPENDENCIES.md](./DEPENDENCIES.md)。摘要：

- **Go 后端**：18 个直接依赖 + ~60 间接依赖，全部 MIT/Apache-2.0/BSD。
- **前端**：Vue3/Arco/axios/pinia 等全部 MIT。
- **Python**：numba/numpy/matplotlib/pytest 全部 BSD/PSF/MIT。
- **结论**：🟢 无 GPL 等 Copyleft 传染性依赖，与项目 MIT 许可兼容，无商用限制。
- **供应链安全**：CI 集成 `go mod verify` / `govulncheck` / `npm audit` 持续监控 CVE。

---

## 四、代码质量与安全评估（QingTian 平台）

| 编号 | 级别 | 问题 | 整改 |
|------|------|------|------|
| H1 | 高 | 注册接口垂直越权：信任请求体 `role`，匿名可注册 admin | ✅ 服务端裁决角色（首用户 admin，其余强制 developer） |
| H2 | 高 | 任务执行器命令注入：`sh -c` 直跑用户命令无隔离 | ✅ 启动安全告警 + README 要求容器隔离（根治需运维层容器化） |
| M1 | 中 | CORS `Access-Control-Allow-Origin: *` | ✅ 改为可信来源白名单工厂 |
| M2 | 中 | 定时计划列表越权（IDOR，代码内 TODO） | ✅ repo/service/handler 贯通 ownerID 过滤 |
| M3 | 中 | 模板/工作流/执行历史/执行实例/计划 Get 越权 | ✅ 补齐 owner 归属校验 |
| M4 | 中 | MinIO 默认凭证 `minioadmin` 无启动强校验 | ✅ validateSecurity 增加强校验 |
| M5/M6 | 中 | Agent token 空值放行、心跳无归属绑定 | ⚠️ 已有告警；归属绑定列入后续（见第七节） |

### 值得肯定的既有实践
JWT 强制签名算法校验（防 alg=none）、bcrypt 口令哈希、SSO state 防 CSRF（crypto/rand + ConstantTimeCompare）、MongoDB 全 bson.M 无注入、文件名 sanitize 防穿越、Mongo cursor 均 defer Close。

### 验证结果
```
go build ./...    → 通过
go vet ./...      → 通过
go test ./...     → 12 个包全部 ok（含 -race）
```

---

## 五、文档与注释规范

| 项 | 状态 |
|----|------|
| 根 README（概述/结构/安装/使用/路线图/FAQ） | 🟢 完整 |
| QingTian README（架构/构建/运行/安全说明/贡献指南/许可） | 🟢 完整 |
| LICENSE（MIT） | 🟢 存在 |
| 代码注释（类/函数/参数/复杂逻辑，中文规范） | 🟢 良好 |
| 依赖清单 DEPENDENCIES.md | 🟢 本次新增 |
| 架构文档（QingTian/plan.md） | 🟢 存在 |

> 建议（低优先）：QingTian 后端可补充 OpenAPI/Swagger 接口文档以进一步提升开源友好度。

---

## 六、第三方资源与知识产权

- 静态资源仅 `web/public/favicon.svg`（自绘 SVG），无第三方图片/字体/音视频。
- 未发现未授权的开源片段或商业代码嵌入。
- 🟢 无知识产权风险。

---

## 七、遗留事项与整改计划

| 事项 | 责任方 | 说明 | 验收标准 |
|------|--------|------|----------|
| **Git 历史凭证清理** | 仓库所有者 | 凭证已 push 到公开仓库，删除工作区文件无法移除历史。需 `git filter-repo`/BFG 重写历史后 force push | GitHub 历史中检索不到 `192.168.10.82`/明文密码 |
| **凭证轮换** | 运维 | 立即修改服务器 `192.168.10.82` 口令、QingTian `admin123` 账号 | 旧凭证失效 |
| **任务执行容器隔离** | 平台团队 | H2 根治需接入 nvidia-container-runtime | 任务在容器内执行，宿主机隔离 |
| **Agent 令牌与节点绑定** | 平台团队 | M5/M6：令牌与节点身份绑定，禁止空令牌放行 | 单一节点令牌不可操作他人节点 |

---

## 八、本次变更清单

**删除**：`deploy_package/`（151 文件）、`test_conn.ps1`、`test_ssh.ps1`、`ssh_test.vbs`

**修改（QingTian）**：
- `internal/api/server.go`（H1 + 5 处 IDOR owner 校验 + schedule 过滤）
- `internal/api/middleware.go`（CORS 白名单）
- `internal/config/config.go`（MinIO 凭证强校验）
- `internal/repository/workflow.go`、`memory.go`、`internal/service/schedule.go`（schedule ownerID 贯通）
- `cmd/agent/main.go`（执行隔离告警）

**新增**：`.github/workflows/ci.yml`、`DEPENDENCIES.md`、本报告；根 `.gitignore` 加固。

---

*报告生成：合规性评估自动化流程 · 覆盖敏感信息/依赖/代码安全/文档/知识产权五大维度*
