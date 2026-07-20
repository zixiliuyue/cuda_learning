<script setup>
// 主框架：左侧菜单 + 顶部栏 + 内容区。
import { computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useAuthStore } from '../store/auth'
import LocaleThemeSwitch from '../components/LocaleThemeSwitch.vue'

const { t } = useI18n()
const router = useRouter()
const route = useRoute()
const auth = useAuthStore()

// 当前选中的菜单项（取路由末段）。
const selectedKey = computed(() => [route.path.split('/').pop() || 'dashboard'])

// 当前登录用户名。
const username = computed(() => auth.username)

// 菜单点击跳转。
function onMenuClick(key) {
  router.push('/' + key)
}

// 退出登录。
function logout() {
  auth.clear()
  router.push('/login')
}
</script>

<template>
  <a-layout class="layout">
    <a-layout-sider :width="220" theme="dark">
      <div class="logo">{{ t('login.brand') }}</div>
      <a-menu theme="dark" :selected-keys="selectedKey" @menu-item-click="onMenuClick">
        <a-menu-item key="dashboard">{{ t('menu.dashboard') }}</a-menu-item>
        <a-menu-item key="nodes">{{ t('menu.nodes') }}</a-menu-item>
        <a-menu-item key="tasks">{{ t('menu.tasks') }}</a-menu-item>
        <a-menu-item key="metering">{{ t('menu.metering') }}</a-menu-item>
        <a-menu-item key="projects">{{ t('menu.projects') }}</a-menu-item>
      </a-menu>
    </a-layout-sider>
    <a-layout>
      <a-layout-header class="header">
        <span class="title">{{ t('login.subtitle') }}</span>
        <a-space :size="16">
          <locale-theme-switch />
          <span class="username">{{ username }}</span>
          <a-button type="text" @click="logout">{{ t('common.logout') }}</a-button>
        </a-space>
      </a-layout-header>
      <a-layout-content class="content">
        <router-view />
      </a-layout-content>
    </a-layout>
  </a-layout>
</template>

<style scoped>
.layout {
  height: 100%;
}
.logo {
  height: 60px;
  line-height: 60px;
  text-align: center;
  color: #fff;
  font-size: 18px;
  font-weight: bold;
}
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  /* 用 Arco 变量，暗色模式下 header 同步变深 */
  background: var(--color-bg-2);
  border-bottom: 1px solid var(--color-border);
}
.title {
  font-size: 16px;
  font-weight: 500;
  color: var(--color-text-1);
}
.username {
  color: var(--color-text-1);
}
.content {
  padding: 24px;
  /* 内容区背景跟随主题 */
  background: var(--color-fill-2);
  overflow: auto;
}
</style>
