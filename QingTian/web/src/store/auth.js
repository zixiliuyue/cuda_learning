// 认证状态集中管理：令牌与用户信息统一由此 store 维护，
// 避免各处直接读写 localStorage 造成状态分散、易出错、难维护。
import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

const TOKEN_KEY = 'qt_token'
const USER_KEY = 'qt_user'

// 安全解析本地存储的用户信息，异常时回退为空对象。
function readUser() {
  try {
    return JSON.parse(localStorage.getItem(USER_KEY) || '{}') || {}
  } catch {
    return {}
  }
}

export const useAuthStore = defineStore('auth', () => {
  const token = ref(localStorage.getItem(TOKEN_KEY) || '')
  const user = ref(readUser())
  // 平台信息：edition 与功能开启状态
  const platformEdition = ref('community')
  const projectEnabled = ref(false)
  const meteringEnabled = ref(false)

  // 是否已登录。
  const isAuthenticated = computed(() => Boolean(token.value))
  // 当前用户名，缺省显示「用户」。
  const username = computed(() => user.value?.username || '用户')

  // 获取平台信息（edition 和功能开启状态）
  async function fetchPlatformInfo() {
    try {
      const res = await fetch('/api/v1/info')
      const data = await res.json()
      if (data.edition) {
        platformEdition.value = data.edition
      }
      if (typeof data.project_enabled === 'boolean') {
        projectEnabled.value = data.project_enabled
      }
      if (typeof data.metering_enabled === 'boolean') {
        meteringEnabled.value = data.metering_enabled
      }
    } catch (e) {
      console.warn('Failed to fetch platform info, using defaults', e)
      // 默认社区版，不开启企业版功能
      platformEdition.value = 'community'
      projectEnabled.value = false
      meteringEnabled.value = false
    }
  }

  // 登录成功后写入令牌与用户信息，并同步持久化。
  function setAuth(nextToken, nextUser) {
    token.value = nextToken || ''
    user.value = nextUser || {}
    if (nextToken) {
      localStorage.setItem(TOKEN_KEY, nextToken)
    } else {
      localStorage.removeItem(TOKEN_KEY)
    }
    localStorage.setItem(USER_KEY, JSON.stringify(user.value))
    // 获取平台信息
    fetchPlatformInfo()
  }

  // 清除登录态（退出登录或令牌失效时调用）。
  function clear() {
    token.value = ''
    user.value = {}
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(USER_KEY)
  }

  return {
    token,
    user,
    isAuthenticated,
    username,
    platformEdition,
    projectEnabled,
    meteringEnabled,
    fetchPlatformInfo,
    setAuth,
    clear
  }
})
