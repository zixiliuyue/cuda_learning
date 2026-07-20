import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// Vite 配置：注册 Vue 插件，将 /api 代理到后端控制面，并做生产分包优化。
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
  },
  build: {
    // 将体积较大的第三方依赖拆分为独立 chunk，利用浏览器缓存、
    // 避免业务代码更新导致整包（含庞大 UI 库）失效重下。
    chunkSizeWarningLimit: 800,
    rollupOptions: {
      output: {
        manualChunks: {
          // Vue 全家桶单独成包。
          vue: ['vue', 'vue-router', 'pinia', 'vue-i18n'],
          // Arco Design UI 库体积最大，单独拆分以最大化缓存命中。
          arco: ['@arco-design/web-vue'],
          // 网络库。
          axios: ['axios']
        }
      }
    }
  }
})
