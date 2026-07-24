<script setup>
// 主框架：左侧菜单 + 顶部栏 + 内容区。
import { computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useAuthStore } from '../store/auth'
import { useUiStore } from '../store/ui'
import LocaleThemeSwitch from '../components/LocaleThemeSwitch.vue'

const { t } = useI18n()
const router = useRouter()
const route = useRoute()
const auth = useAuthStore()
const ui = useUiStore()

// 当前选中的菜单项（取路由末段）。
const selectedKey = computed(() => [route.path.split('/').pop() || 'dashboard'])

// 当前登录用户名。
const username = computed(() => auth.username)

// 侧边栏和菜单主题跟随全局主题切换。
const siderTheme = computed(() => ui.theme === 'dark' ? 'dark' : 'light')
const menuTheme = computed(() => ui.theme === 'dark' ? 'dark' : 'light')

// 始终显示所有菜单，admin 需要完整功能
const showMetering = true
const showProjects = true

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
    <a-layout-sider :width="220" :theme="siderTheme">
      <div class="logo">{{ t('login.brand') }}</div>
      <a-menu :theme="menuTheme" :selected-keys="selectedKey" @menu-item-click="onMenuClick">
        <a-menu-item key="dashboard">
          <icon-dashboard />
          <span>{{ t('menu.dashboard') }}</span>
        </a-menu-item>
        <a-menu-item key="nodes">
          <icon-cloud />
          <span>{{ t('menu.nodes') }}</span>
        </a-menu-item>
        <a-menu-item key="tasks">
          <icon-list />
          <span>{{ t('menu.tasks') }}</span>
        </a-menu-item>
        <a-menu-item key="metering">
          <icon-bar-chart />
          <span>{{ t('menu.metering') }}</span>
        </a-menu-item>
        <a-menu-item key="projects">
          <icon-folder />
          <span>{{ t('menu.projects') }}</span>
        </a-menu-item>
        <a-divider />
        <a-menu-item key="workflows">
          <icon-list />
          <span>{{ t('menu.workflows') }}</span>
        </a-menu-item>
        <a-menu-item key="schedules">
          <icon-cloud />
          <span>{{ t('menu.schedules') }}</span>
        </a-menu-item>
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
  color: var(--color-text-1);
  font-size: 18px;
  font-weight: bold;
  border-bottom: 1px solid var(--color-border);
}
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  height: 64px;
  /* 用 Arco 变量，暗色模式下 header 同步变深 */
  background: var(--color-bg-2);
  border-bottom: 1px solid var(--color-border);
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
}
.title {
  font-size: 18px;
  font-weight: 600;
  color: var(--color-text-1);
  letter-spacing: 0.2px;
}
.username {
  color: var(--color-text-1);
  font-size: 16px;
}
.content {
  padding: 20px 24px;
  /* 内容区背景跟随主题 */
  background: var(--color-fill-2);
  overflow: auto;
  min-height: calc(100vh - 64px);
}
</style>
