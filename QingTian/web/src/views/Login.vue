<script setup>
// 登录页：用户输入用户名密码登录，成功后保存令牌并跳转主页。
import { ref } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { Message } from '@arco-design/web-vue'
import { authApi } from '../api'
import { useAuthStore } from '../store/auth'

const router = useRouter()
const route = useRoute()
const auth = useAuthStore()
const form = ref({ username: '', password: '' })
const loading = ref(false)

// 处理登录提交。
async function handleLogin() {
  if (loading.value) return
  if (!form.value.username || !form.value.password) {
    Message.warning('请输入用户名和密码')
    return
  }
  loading.value = true
  try {
    const res = await authApi.login(form.value.username, form.value.password)
    auth.setAuth(res.token, res.user)
    Message.success('登录成功')
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
  <div class="login-wrap">
    <a-card class="login-card" :bordered="false">
      <div class="brand">
        <h1>QingTian 擎天</h1>
        <p>算力调度平台</p>
      </div>
      <a-form :model="form" layout="vertical" @submit-success="handleLogin">
        <a-form-item field="username" label="用户名">
          <a-input v-model="form.username" placeholder="请输入用户名" allow-clear />
        </a-form-item>
        <a-form-item field="password" label="密码">
          <a-input-password v-model="form.password" placeholder="请输入密码" allow-clear />
        </a-form-item>
        <a-button type="primary" long :loading="loading" @click="handleLogin">登录</a-button>
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
  background: linear-gradient(135deg, #1d2129 0%, #165dff 100%);
}
.login-card {
  width: 360px;
  border-radius: 12px;
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
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
  color: #86909c;
}
</style>
