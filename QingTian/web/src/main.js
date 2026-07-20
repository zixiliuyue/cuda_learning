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

// 挂载前应用已持久化的主题，避免首屏闪烁(先亮后暗)。
// 需在 pinia 注册之后调用，否则 store 不可用。
useUiStore().initTheme()

app.mount('#app')
