// Axios 封装：统一 baseURL、自动携带 JWT 令牌、统一响应/错误处理。
import axios from 'axios'
import { Message } from '@arco-design/web-vue'

// 创建 axios 实例，所有请求以 /api/v1 为前缀。
const request = axios.create({
  baseURL: '/api/v1',
  timeout: 10000
})

// 请求拦截器：从 localStorage 读取令牌并加入 Authorization 头。
request.interceptors.request.use((config) => {
  const token = localStorage.getItem('qt_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// 响应拦截器：解包统一响应结构 {code,message,data}，并统一错误提示。
request.interceptors.response.use(
  (resp) => resp.data.data,
  (error) => {
    const msg = error.response?.data?.message || error.message || '请求失败'
    Message.error(msg)
    // 401 未授权：清除令牌并跳转登录。
    if (error.response?.status === 401) {
      localStorage.removeItem('qt_token')
      if (location.hash !== '#/login') location.hash = '#/login'
    }
    return Promise.reject(error)
  }
)

export default request
