<script setup>
// 主框架：左侧菜单 + 顶部栏 + 内容区。
import { computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useAuthStore } from '../store/auth'

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
      <div class="logo">QingTian 擎天</div>
      <a-menu theme="dark" :selected-keys="selectedKey" @menu-item-click="onMenuClick">
        <a-menu-item key="dashboard">概览</a-menu-item>
        <a-menu-item key="nodes">节点管理</a-menu-item>
        <a-menu-item key="tasks">任务管理</a-menu-item>
      </a-menu>
    </a-layout-sider>
    <a-layout>
      <a-layout-header class="header">
        <span class="title">算力调度控制台</span>
        <a-space>
          <span>{{ username }}</span>
          <a-button type="text" @click="logout">退出</a-button>
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
  background: #fff;
  border-bottom: 1px solid #f0f0f0;
}
.title {
  font-size: 16px;
  font-weight: 500;
}
.content {
  padding: 24px;
  background: #f7f8fa;
  overflow: auto;
}
</style>
