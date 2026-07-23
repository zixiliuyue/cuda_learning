import request from './request'

// 工作流定义 API
export const workflowApi = {
  // 获取工作流列表
  list: (params) => {
    return request.get('/workflows', { params })
  },
  // 获取工作流详情
  get: (id) => {
    return request.get(`/workflows/${id}`)
  },
  // 创建工作流
  create: (data) => {
    return request.post('/workflows', data)
  },
  // 更新工作流
  update: (id, data) => {
    return request.put(`/workflows/${id}`, data)
  },
  // 删除工作流
  delete: (id) => {
    return request.delete(`/workflows/${id}`)
  },
  // 手动触发工作流执行
  run: (id, data) => {
    return request.post(`/workflows/${id}/run`, data)
  },
  // 获取工作流执行历史
  listRuns: (workflowId, params) => {
    return request.get(`/workflows/${workflowId}/runs`, { params })
  },
  // 获取工作流执行详情
  getRun: (id) => {
    return request.get(`/workflow-runs/${id}`)
  },
}

// 定时调度 API
export const scheduleApi = {
  // 获取调度列表
  list: (params) => {
    return request.get('/schedules', { params })
  },
  // 获取调度详情
  get: (id) => {
    return request.get(`/schedules/${id}`)
  },
  // 创建调度
  create: (data) => {
    return request.post('/schedules', data)
  },
  // 更新调度
  update: (id, data) => {
    return request.put(`/schedules/${id}`, data)
  },
  // 删除调度
  delete: (id) => {
    return request.delete(`/schedules/${id}`)
  },
  // 手动触发一次执行
  trigger: (id, data) => {
    return request.post(`/schedules/${id}/trigger`, data)
  },
}
