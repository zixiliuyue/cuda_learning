<script setup>
// 登录页：用户输入用户名密码登录，成功后保存令牌并跳转主页。
import { ref, onMounted, computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { Message } from '@arco-design/web-vue'
import { authApi } from '../api'
import { useAuthStore } from '../store/auth'
import { useUiStore } from '../store/ui'
import LocaleThemeSwitch from '../components/LocaleThemeSwitch.vue'

const { t } = useI18n()
const router = useRouter()
const route = useRoute()
const auth = useAuthStore()
const ui = useUiStore()
const form = ref({ username: '', password: '' })
const loading = ref(false)
// 是否记住账号密码，勾选后凭据会保存在浏览器本地。
const remember = ref(false)

// 登录页背景渐变根据主题和主题色变化
import { THEME_COLORS } from '../store/ui'
const loginBackground = computed(() => {
  const primaryColor = THEME_COLORS[ui.themeColor]?.primary || THEME_COLORS.blue.primary
  const lightColor = adjustBrightness(primaryColor, 80)
  return ui.theme === 'dark'
    ? `linear-gradient(135deg, #1d2129 0%, ${primaryColor} 100%)`
    : `linear-gradient(135deg, ${lightColor} 0%, ${primaryColor} 100%)`
})

// 调整 hex 颜色亮度，生成浅色调色板
function adjustBrightness(hex, percent) {
  hex = hex.replace(/^#/, '')
  let r = parseInt(hex.substring(0, 2), 16)
  let g = parseInt(hex.substring(2, 4), 16)
  let b = parseInt(hex.substring(4, 6), 16)
  r = Math.min(255, r + (255 - r) * percent / 100)
  g = Math.min(255, g + (255 - g) * percent / 100)
  b = Math.min(255, b + (255 - b) * percent / 100)
  return '#' + ((r << 16) | (g << 8) | b).toString(16).padStart(6, '0')
}

// localStorage 中保存记住密码凭据的键名。
const REMEMBER_KEY = 'qingtian_remember_cred'

// 读取本地记住的凭据并回填表单。
// 注意：btoa 只是简单编码，避免明文直接肉眼可见，并非加密，本地仍可被解码。
function loadRememberedCred() {
  try {
    const raw = localStorage.getItem(REMEMBER_KEY)
    if (!raw) return
    const { username, password } = JSON.parse(atob(raw))
    if (username) {
      form.value.username = username
      form.value.password = password || ''
      remember.value = true
    }
  } catch (e) {
    // 解析失败（数据被篡改等）时清理掉，避免影响后续登录。
    localStorage.removeItem(REMEMBER_KEY)
  }
}

// 根据是否勾选记住密码，保存或清除本地凭据。
function saveOrClearRememberedCred() {
  if (remember.value) {
    const payload = btoa(JSON.stringify({
      username: form.value.username,
      password: form.value.password,
    }))
    localStorage.setItem(REMEMBER_KEY, payload)
  } else {
    localStorage.removeItem(REMEMBER_KEY)
  }
}

// 页面加载时尝试回填已记住的账号密码。
onMounted(loadRememberedCred)

// 处理登录提交。
async function handleLogin() {
  if (loading.value) return
  if (!form.value.username || !form.value.password) {
    Message.warning(t('login.needInput'))
    return
  }
  loading.value = true
  try {
    const res = await authApi.login(form.value.username, form.value.password)
    auth.setAuth(res.token, res.user)
    // 登录成功后再持久化凭据，避免记住错误的密码。
    saveOrClearRememberedCred()
    Message.success(t('login.success'))
    // 支持登录后回跳到原访问页面。
    const redirect = typeof route.query.redirect === 'string' ? route.query.redirect : '/dashboard'
    router.push(redirect)
  } catch (e) {
    // 错误已由拦截器统一提示。
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="login-wrap" :style="{ background: loginBackground }">
    <a-card class="login-card" :bordered="false">
      <!-- 右上角语言/主题切换 -->
      <div class="switch-bar">
        <locale-theme-switch />
      </div>
      <div class="brand">
        <h1>{{ t('login.brand') }}</h1>
        <p>{{ t('login.subtitle') }}</p>
      </div>
      <a-form :model="form" layout="vertical" @submit-success="handleLogin">
        <a-form-item field="username" :label="t('login.username')">
          <a-input v-model="form.username" :placeholder="t('login.usernamePlaceholder')" allow-clear />
        </a-form-item>
        <a-form-item field="password" :label="t('login.password')">
          <a-input-password v-model="form.password" :placeholder="t('login.passwordPlaceholder')" allow-clear />
        </a-form-item>
        <a-form-item hide-label>
          <a-checkbox v-model="remember">{{ t('login.remember') }}</a-checkbox>
        </a-form-item>
        <a-button type="primary" long :loading="loading" @click="handleLogin">{{ t('login.loginBtn') }}</a-button>
        <div class="to-register">
          {{ t('login.toRegister') }} <a-router-link to="/register">{{ t('register.registerBtn') }}</a-router-link>
        </div>
      </a-form>
    </a-card>
  </div>
</template>

<style scoped>
.login-wrap {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.3s ease;
}
.login-card {
  position: relative;
  width: 360px;
  border-radius: 12px;
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
  /* 用 Arco 容器背景变量，暗色模式下卡片同步变深 */
  background: var(--color-bg-2);
}
/* 右上角切换控件 */
.switch-bar {
  position: absolute;
  top: 12px;
  right: 12px;
}
.brand {
  text-align: center;
  margin-bottom: 24px;
}
.brand h1 {
  margin: 0;
  color: var(--color-primary);
}
.brand p {
  margin: 4px 0 0;
  /* 次要文字色，跟随主题 */
  color: var(--color-text-3);
}
.to-register {
  margin-top: 16px;
  text-align: center;
  color: var(--color-text-3);
}
.to-register a {
  color: var(--color-primary);
  text-decoration: none;
}
</style>
