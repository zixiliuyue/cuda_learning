// 路由配置：登录页 + 需要认证的主框架（含节点、任务子页面）。
// 视图组件全部改为动态 import，实现按路由代码分割（懒加载），
// 首屏仅加载登录页所需代码，其余页面按需拉取，显著降低首包体积。
import { createRouter, createWebHashHistory } from 'vue-router'
import i18n from '../i18n'

// meta.titleKey 为 i18n 词条键，标题在守卫中按当前语言实时翻译。
const routes = [
  {
    path: '/login',
    component: () => import('../views/Login.vue'),
    meta: { titleKey: 'login.loginBtn', public: true }
  },
  {
    path: '/register',
    component: () => import('../views/Register.vue'),
    meta: { titleKey: 'register.title', public: true }
  },
  {
    path: '/',
    component: () => import('../views/Layout.vue'),
    redirect: '/dashboard',
    children: [
      { path: 'dashboard', component: () => import('../views/Dashboard.vue'), meta: { titleKey: 'menu.dashboard' } },
      { path: 'nodes', component: () => import('../views/Nodes.vue'), meta: { titleKey: 'menu.nodes' } },
      { path: 'tasks', component: () => import('../views/Tasks.vue'), meta: { titleKey: 'menu.tasks' } },
      { path: 'metering', component: () => import('../views/Metering.vue'), meta: { titleKey: 'menu.metering' } },
      { path: 'projects', component: () => import('../views/Projects.vue'), meta: { titleKey: 'menu.projects' } },
      { path: 'workflows', component: () => import('../views/WorkflowList.vue'), meta: { titleKey: 'menu.workflows' } },
      { path: 'workflows/editor/:id', component: () => import('../views/WorkflowEditor.vue'), meta: { titleKey: 'workflow.title' } },
      { path: 'workflows/runs/:id', component: () => import('../views/WorkflowRunDetail.vue'), meta: { titleKey: 'workflowRun.runTitle' } },
      { path: 'schedules', component: () => import('../views/ScheduleManagement.vue'), meta: { titleKey: 'menu.schedules' } },
    ]
  },
  // 兜底：未匹配路由重定向到首页。
  { path: '/:pathMatch(.*)*', redirect: '/' }
]

const router = createRouter({
  history: createWebHashHistory(),
  routes
})

const BASE_TITLE = 'QingTian'

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

// 后置守卫：根据路由 meta.titleKey 按当前语言更新文档标题。
router.afterEach((to) => {
  const t = i18n.global.t
  document.title = to.meta?.titleKey ? `${t(to.meta.titleKey)} · ${BASE_TITLE}` : BASE_TITLE
})

export default router
