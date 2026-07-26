# 依赖清单与开源许可合规说明

> 生成范围：`f:\code\cuda_learning\`。
> 结论：**全部依赖均为宽松许可（MIT / Apache-2.0 / BSD / PSF），无 GPL 等 Copyleft 传染性条款，与项目 MIT 许可完全兼容，符合开源政策。**

## 一、后端 Go 直接依赖（QingTian/go.mod，module `qingtian`，Go 1.26）

| 依赖 | 版本 | 许可类型 | 来源 |
|------|------|----------|------|
| github.com/coreos/go-oidc/v3 | v3.20.0 | Apache-2.0 | https://github.com/coreos/go-oidc |
| github.com/go-resty/resty/v2 | v2.17.2 | MIT | https://github.com/go-resty/resty |
| github.com/golang-jwt/jwt/v5 | v5.3.1 | MIT | https://github.com/golang-jwt/jwt |
| github.com/google/uuid | v1.6.0 | BSD-3-Clause | https://github.com/google/uuid |
| github.com/minio/minio-go/v7 | v7.2.1 | Apache-2.0 | https://github.com/minio/minio-go |
| github.com/nsqio/go-nsq | v1.1.0 | MIT | https://github.com/nsqio/go-nsq |
| github.com/prometheus/client_golang | v1.24.0 | Apache-2.0 | https://github.com/prometheus/client_golang |
| github.com/redis/go-redis/v9 | v9.21.0 | BSD-2-Clause | https://github.com/redis/go-redis |
| github.com/robfig/cron/v3 | v3.0.1 | MIT | https://github.com/robfig/cron |
| go.mongodb.org/mongo-driver | v1.17.9 | Apache-2.0 | https://github.com/mongodb/mongo-go-driver |
| golang.org/x/crypto | v0.54.0 | BSD-3-Clause | https://pkg.go.dev/golang.org/x/crypto |
| golang.org/x/oauth2 | v0.36.0 | BSD-3-Clause | https://pkg.go.dev/golang.org/x/oauth2 |
| gopkg.in/yaml.v3 | v3.0.1 | MIT / Apache-2.0 | https://github.com/go-yaml/yaml |
| k8s.io/api | v0.36.2 | Apache-2.0 | https://github.com/kubernetes/api |
| k8s.io/apimachinery | v0.36.2 | Apache-2.0 | https://github.com/kubernetes/apimachinery |
| k8s.io/client-go | v0.36.0 | Apache-2.0 | https://github.com/kubernetes/client-go |
| sigs.k8s.io/controller-runtime | v0.24.1 | Apache-2.0 | https://github.com/kubernetes-sigs/controller-runtime |
| volcano.sh/apis | v1.19.6 | Apache-2.0 | https://github.com/volcano-sh/apis |

> 间接依赖（`// indirect`，约 60 项）均由上述直接依赖引入，主要来自
> Kubernetes、Prometheus、MinIO、MongoDB 生态，许可同为 Apache-2.0 / BSD / MIT。
> 完整锁定版本见 `QingTian/go.sum`；CI 中通过 `go mod verify` 校验完整性、
> `govulncheck` 扫描已知 CVE。

## 二、前端 npm 依赖（QingTian/web/package.json）

### 生产依赖
| 依赖 | 版本约束 | 许可类型 | 来源 |
|------|----------|----------|------|
| @arco-design/web-vue | ^2.57.0 | MIT | https://github.com/arco-design/arco-design-vue |
| @vue-flow/core | ^1.48.2 | MIT | https://github.com/bcakmakoglu/vue-flow |
| axios | ^1.7.7 | MIT | https://github.com/axios/axios |
| pinia | ^2.2.4 | MIT | https://github.com/vuejs/pinia |
| vue | ^3.5.12 | MIT | https://github.com/vuejs/core |
| vue-i18n | ^9.14.0 | MIT | https://github.com/intlify/vue-i18n |
| vue-router | ^4.4.5 | MIT | https://github.com/vuejs/router |

### 开发依赖
| 依赖 | 版本约束 | 许可类型 | 来源 |
|------|----------|----------|------|
| @vitejs/plugin-vue | ^5.1.4 | MIT | https://github.com/vitejs/vite-plugin-vue |
| vite | ^5.4.10 | MIT | https://github.com/vitejs/vite |

> 精确锁定版本见 `QingTian/web/package-lock.json`；CI 中通过 `npm audit
> --audit-level=high` 审计传递依赖漏洞。

## 三、Python 依赖（requirements.txt，CUDA 学习项目）

| 依赖 | 版本约束 | 许可类型 | 来源 |
|------|----------|----------|------|
| numba | >=0.59.0 | BSD-2-Clause | https://github.com/numba/numba |
| numpy | >=1.24.0 | BSD-3-Clause | https://github.com/numpy/numpy |
| matplotlib | >=3.7.0 | Matplotlib(PSF-based, BSD 兼容) | https://github.com/matplotlib/matplotlib |
| pytest | >=7.0.0 | MIT | https://github.com/pytest-dev/pytest |

> torch / opencv-python / triton / transformers 在 requirements.txt 中以注释形式
> 提供（需按 CUDA 环境单独安装），非强制分发依赖：
> PyTorch(BSD-3)、OpenCV(Apache-2.0)、Triton(MIT)、Transformers(Apache-2.0)，均为宽松许可。

## 四、运行时/系统级依赖（非分发）

| 组件 | 用途 | 说明 |
|------|------|------|
| NVIDIA CUDA Toolkit 12.8~13.x | CUDA 编译/运行 | 专有 EULA，作为运行时环境依赖，不随源码分发，不影响仓库开源许可 |
| Docker 镜像（mongo/redis/nsq/minio/nginx/golang/node/alpine） | 部署编排 | 均为官方公开镜像，各自许可（多为 Apache-2.0/MIT/BSD），以容器形式引用非源码嵌入 |

## 五、合规结论

1. **许可兼容性**：所有依赖为 MIT/Apache-2.0/BSD/PSF 宽松许可，可与本项目 `MIT License` 共存与再分发。
2. **无 Copyleft 风险**：未发现 GPL/LGPL/AGPL 等具有传染性的 Copyleft 依赖，商业使用无限制。
3. **供应链安全**：已在 CI 中集成 `go mod verify`、`govulncheck`、`npm audit`，持续监控 CVE 与完整性。
4. **版本锁定**：go.sum 与 package-lock.json 已锁定精确版本，保证可复现构建。
