import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// Vite 配置：注册 Vue 插件，并将 /api 代理到后端控制面。
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      // 开发环境下将前端 /api 请求代理到 Go 后端，规避跨域。
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true
      }
    }
  }
})
