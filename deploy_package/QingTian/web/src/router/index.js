// 路由配置：登录页 + 需要认证的主框架（含节点、任务子页面）。
// 视图组件全部改为动态 import，实现按路由代码分割（懒加载），
// 首屏仅加载登录页所需代码，其余页面按需拉取，显著降低首包体积。
import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
  {
    path: '/login',
    component: () => import('../views/Login.vue'),
    meta: { title: '登录', public: true }
  },
  {
    path: '/',
    component: () => import('../views/Layout.vue'),
    redirect: '/dashboard',
    children: [
      { path: 'dashboard', component: () => import('../views/Dashboard.vue'), meta: { title: '概览' } },
      { path: 'nodes', component: () => import('../views/Nodes.vue'), meta: { title: '节点管理' } },
      { path: 'tasks', component: () => import('../views/Tasks.vue'), meta: { title: '任务管理' } }
    ]
  },
  // 兜底：未匹配路由重定向到首页。
  { path: '/:pathMatch(.*)*', redirect: '/' }
]

const router = createRouter({
  history: createWebHashHistory(),
  routes
})

const BASE_TITLE = 'QingTian 算力调度平台'

// 全局前置守卫：未登录访问受保护页面时跳转登录页。
router.beforeEach((to, _from, next) => {
  const token = localStorage.getItem('qt_token')
  if (!to.meta?.public && !token) {
    next({ path: '/login', query: to.fullPath !== '/' ? { redirect: to.fullPath } : undefined })
  } else if (to.path === '/login' && token) {
    // 已登录再访问登录页，直接回到首页。
    next('/dashboard')
  } else {
    next()
  }
})

// 后置守卫：根据路由 meta.title 更新文档标题。
router.afterEach((to) => {
  document.title = to.meta?.title ? `${to.meta.title} · ${BASE_TITLE}` : BASE_TITLE
})

export default router
