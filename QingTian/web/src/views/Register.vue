<script setup>
// 用户注册页：新用户填写用户名密码注册，成功跳转登录。
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { Message } from '@arco-design/web-vue'
import { authApi } from '../api'
import LocaleThemeSwitch from '../components/LocaleThemeSwitch.vue'

const { t } = useI18n()
const router = useRouter()

const form = ref({
  username: '',
  password: '',
  confirmPassword: '',
})
const loading = ref(false)

// 处理注册提交。
async function handleRegister() {
  if (loading.value) return
  if (!form.value.username || !form.value.password || !form.value.confirmPassword) {
    Message.warning(t('register.username') + ', ' + t('register.password') + ', ' + t('register.confirmPassword') + ' ' + t('common.required'))
    return
  }
  if (form.value.password !== form.value.confirmPassword) {
    Message.warning(t('register.passwordMismatch'))
    return
  }
  loading.value = true
  try {
    await authApi.register(form.value.username, form.value.password)
    Message.success(t('register.success'))
    router.push('/login')
  } catch (e) {
    // 错误已由拦截器统一提示。
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="register-wrap">
    <a-card class="register-card" :bordered="false">
      <!-- 右上角语言/主题切换 -->
      <div class="switch-bar">
        <locale-theme-switch />
      </div>
      <div class="brand">
        <h1>{{ t('login.brand') }}</h1>
        <p>{{ t('register.title') }}</p>
      </div>
      <a-form :model="form" layout="vertical" @submit-success="handleRegister">
        <a-form-item field="username" :label="t('register.username')" required>
          <a-input v-model="form.username" :placeholder="t('register.usernamePlaceholder')" allow-clear />
        </a-form-item>
        <a-form-item field="password" :label="t('register.password')" required>
          <a-input-password v-model="form.password" :placeholder="t('register.passwordPlaceholder')" allow-clear />
        </a-form-item>
        <a-form-item field="confirmPassword" :label="t('register.confirmPassword')" required>
          <a-input-password v-model="form.confirmPassword" :placeholder="t('register.confirmPasswordPlaceholder')" allow-clear />
        </a-form-item>
        <a-button type="primary" long :loading="loading" @click="handleRegister">{{ t('register.registerBtn') }}</a-button>
        <div class="to-login">
          {{ t('register.toLogin') }} <a-router-link to="/login">{{ t('login.loginBtn') }}</a-router-link>
        </div>
      </a-form>
    </a-card>
  </div>
</template>

<style scoped>
.register-wrap {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #1d2129 0%, #165dff 100%);
}
.register-card {
  position: relative;
  width: 380px;
  border-radius: 12px;
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
  background: var(--color-bg-2);
}
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
  color: #165dff;
}
.brand p {
  margin: 4px 0 0;
  color: var(--color-text-3);
}
.to-login {
  margin-top: 16px;
  text-align: center;
  color: var(--color-text-3);
}
.to-login a {
  color: #165dff;
  text-decoration: none;
}
</style>
