// vue-i18n 实例：Composition API 模式(legacy: false)。
// 初始语言从 localStorage 读取，缺省简体中文；语言切换由 ui store 统一驱动。
import { createI18n } from 'vue-i18n'
import zhCN from './locales/zh-CN'
import enUS from './locales/en-US'

// 本地存储的语言键名，与 store/ui.js 保持一致。
export const LOCALE_KEY = 'qt_locale'

// 支持的语言列表，供切换控件渲染选项。
export const SUPPORTED_LOCALES = [
  { value: 'zh-CN', label: '简体中文' },
  { value: 'en-US', label: 'English' },
]

// 读取初始语言：优先本地存储，否则默认中文。
function getInitialLocale() {
  const saved = localStorage.getItem(LOCALE_KEY)
  return saved === 'en-US' || saved === 'zh-CN' ? saved : 'zh-CN'
}

const i18n = createI18n({
  legacy: false, // 使用 Composition API，配合 <script setup> 的 useI18n()
  locale: getInitialLocale(),
  fallbackLocale: 'zh-CN',
  messages: {
    'zh-CN': zhCN,
    'en-US': enUS,
  },
})

export default i18n
