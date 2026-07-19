// 前端入口：创建 Vue 应用，挂载 Arco Design、路由与状态管理。
import { createApp } from 'vue'
import ArcoVue from '@arco-design/web-vue'
import '@arco-design/web-vue/dist/arco.css'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'

const app = createApp(App)
app.use(ArcoVue)   // 注册 Arco Design 组件库
app.use(createPinia())
app.use(router)
app.mount('#app')
