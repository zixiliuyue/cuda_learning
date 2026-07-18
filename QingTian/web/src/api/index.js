// 业务 API 封装：将后端接口聚合为按资源分组的调用方法。
import request from './request'

// 认证相关。
export const authApi = {
  // 登录，返回 {token, user}
  login: (username, password) => request.post('/auth/login', { username, password }),
  // 注册
  register: (username, password, role) => request.post('/auth/register', { username, password, role })
}

// 节点相关。
export const nodeApi = {
  // 查询节点列表
  list: () => request.get('/nodes')
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
  cancel: (id) => request.post(`/tasks/${id}/cancel`)
}
