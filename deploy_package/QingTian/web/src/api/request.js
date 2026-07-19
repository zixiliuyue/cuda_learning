// Axios 封装：统一 baseURL、自动携带 JWT 令牌、统一响应/错误处理。
import axios from 'axios'
import { Message } from '@arco-design/web-vue'

// 统一的本地存储键与后端约定的成功业务码。
const TOKEN_KEY = 'qt_token'
const SUCCESS_CODE = 'OK'

// baseURL 优先取构建期环境变量（VITE_API_BASE），缺省回退到 /api/v1。
// 便于在不同部署环境（本地/测试/生产）无需改代码即可切换后端地址。
const baseURL = import.meta.env.VITE_API_BASE || '/api/v1'

// 创建 axios 实例。
const request = axios.create({
  baseURL,
  timeout: 15000
})

// 请求拦截器：从 localStorage 读取令牌并加入 Authorization 头。
request.interceptors.request.use((config) => {
  const token = localStorage.getItem(TOKEN_KEY)
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// 响应拦截器：解包统一响应结构 {code,message,data}，并统一错误提示。
request.interceptors.response.use(
  (resp) => {
    const body = resp.data
    // 后端统一结构：{ code, message, data }。
    // 兼容极少数直接返回裸数据的场景（无 code 字段时原样返回）。
    if (body && typeof body === 'object' && 'code' in body) {
      if (body.code !== SUCCESS_CODE) {
        // HTTP 200 但业务码非 OK，同样视为失败。
        const err = new Error(body.message || '请求失败')
        Message.error(err.message)
        return Promise.reject(err)
      }
      return body.data
    }
    return body
  },
  (error) => {
    // 请求被取消时静默处理，避免误报。
    if (axios.isCancel?.(error)) {
      return Promise.reject(error)
    }
    const status = error.response?.status
    const msg = error.response?.data?.message || error.message || '请求失败'
    Message.error(msg)
    // 401 未授权：清除令牌并跳转登录。
    if (status === 401) {
      localStorage.removeItem(TOKEN_KEY)
      localStorage.removeItem('qt_user')
      if (location.hash !== '#/login') location.hash = '#/login'
    }
    return Promise.reject(error)
  }
)

export default request
