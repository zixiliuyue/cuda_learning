// 前端入口：创建 Vue 应用，挂载 Arco Design、路由、状态管理与国际化。
import { createApp } from 'vue'
import ArcoVue from '@arco-design/web-vue'
import '@arco-design/web-vue/dist/arco.css'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'
import i18n from './i18n'
import { useUiStore } from './store/ui'

const app = createApp(App)
app.use(ArcoVue)   // 注册 Arco Design 组件库
app.use(createPinia())
app.use(router)
app.use(i18n)      // 注册 vue-i18n

// DOM 加载后应用已持久化的主题，避免首屏闪烁(先亮后暗)。
// 需确保 document.body 存在，挂载后立即执行不影响视觉。
app.mount('#app')
useUiStore().initTheme()

// 为 html 添加暗色模式背景，防止滚动区域露出底色不匹配
if (document.documentElement.getAttribute('arco-theme') === null) {
  document.documentElement.classList.toggle('arco-dark', useUiStore().theme === 'dark')
}
