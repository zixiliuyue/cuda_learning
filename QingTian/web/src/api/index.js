// 业务 API 封装：将后端接口聚合为按资源分组的调用方法。
import request from './request'

// 认证相关。
export const authApi = {
  // 登录，返回 {token, user}
  login: (username, password) => request.post('/auth/login', { username, password }),
  // 注册
  register: (username, password) => request.post('/auth/register', { username, password, role: 'developer' })
}

// 节点相关。
export const nodeApi = {
  // 查询节点列表
  list: () => request.get('/nodes'),
  // 获取空闲节点列表
  listIdle: () => request.get('/nodes/idle')
}

// 任务相关。
export const taskApi = {
  // 查询任务列表，可按状态过滤
  list: (status) => request.get('/tasks', { params: status ? { status } : {} }),
  // 提交任务
  submit: (payload) => request.post('/tasks', payload),
  // 查询任务详情
  get: (id) => request.get(`/tasks/${id}`),
  // 取消任务
  cancel: (id) => request.post(`/tasks/${id}/cancel`),
  // 恢复已取消/失败任务
  resume: (id) => request.post(`/tasks/${id}/resume`),
  // 获取任务历史日志
  logs: (id) => request.get(`/tasks/${id}/logs`),
  // 获取结果文件下载预签名 URL
  resultURL: (id) => request.get(`/tasks/${id}/result`)
}

// 数据集存储相关。
export const datasetApi = {
  // 获取上传预签名 URL
  presignUpload: () => request.get('/datasets/presign-upload')
}

// 计量计费相关。
export const meteringApi = {
  // 获取计量看板（全局统计 + 用户用量列表）
  dashboard: () => request.get('/metering/dashboard')
}

// 配额管理相关。
export const quotaApi = {
  // 获取配额信息
  get: (subject) => request.get(`/quotas/${subject}`)
}

// 项目管理相关。
export const projectApi = {
  // 列出我参与的项目
  list: () => request.get('/projects'),
  // 创建新项目
  create: (name) => request.post('/projects', { name }),
  // 添加项目成员
  addMember: (projectId, username) => request.post(`/projects/${projectId}/members`, { username }),
  // 删除项目（仅所有者）
  delete: (projectId) => request.delete(`/projects/${projectId}`),
  // 重命名项目（仅所有者）
  rename: (projectId, name) => request.put(`/projects/${projectId}`, { name })
}
export { workflowApi, scheduleApi } from './workflow'
