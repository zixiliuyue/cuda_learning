// 路由配置：登录页 + 需要认证的主框架（含节点、任务子页面）。
import { createRouter, createWebHashHistory } from 'vue-router'
import Login from '../views/Login.vue'
import Layout from '../views/Layout.vue'
import Dashboard from '../views/Dashboard.vue'
import Nodes from '../views/Nodes.vue'
import Tasks from '../views/Tasks.vue'

const routes = [
  { path: '/login', component: Login },
  {
    path: '/',
    component: Layout,
    redirect: '/dashboard',
    children: [
      { path: 'dashboard', component: Dashboard, meta: { title: '概览' } },
      { path: 'nodes', component: Nodes, meta: { title: '节点管理' } },
      { path: 'tasks', component: Tasks, meta: { title: '任务管理' } }
    ]
  }
]

const router = createRouter({
  history: createWebHashHistory(),
  routes
})

// 全局前置守卫：未登录访问受保护页面时跳转登录页。
router.beforeEach((to, _from, next) => {
  const token = localStorage.getItem('qt_token')
  if (to.path !== '/login' && !token) {
    next('/login')
  } else {
    next()
  }
})

export default router
