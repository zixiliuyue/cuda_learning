// UI 状态集中管理：语言与主题。
// 集中读写 localStorage 与副作用(切 i18n locale、切 arco 暗黑属性)，
// 避免逻辑散落在各组件，保证切换行为一致、可持久化。
import { defineStore } from 'pinia'
import { ref } from 'vue'
import i18n, { LOCALE_KEY } from '../i18n'

// 主题持久化键名。
const THEME_KEY = 'qt_theme'

// 读取初始主题：优先本地存储，否则亮色。
function getInitialTheme() {
  const saved = localStorage.getItem(THEME_KEY)
  return saved === 'dark' ? 'dark' : 'light'
}

// 将暗黑属性应用到 body：Arco 官方约定 body[arco-theme=dark] 启用暗色变量。
function applyThemeToDom(theme) {
  if (theme === 'dark') {
    document.body.setAttribute('arco-theme', 'dark')
  } else {
    document.body.removeAttribute('arco-theme')
  }
}

export const useUiStore = defineStore('ui', () => {
  // 当前语言，初值与 i18n 实例一致(i18n 已从 localStorage 读过)。
  const locale = ref(i18n.global.locale.value)
  // 当前主题：'light' | 'dark'。
  const theme = ref(getInitialTheme())

  // 切换语言：同步 i18n 实例 + 持久化。Arco 组件文案由 App.vue 的
  // ConfigProvider 依据本值切换，无需在此处理。
  function setLocale(next) {
    if (next !== 'zh-CN' && next !== 'en-US') return
    locale.value = next
    i18n.global.locale.value = next
    localStorage.setItem(LOCALE_KEY, next)
  }

  // 设置主题：更新状态 + DOM 属性 + 持久化。
  function setTheme(next) {
    theme.value = next === 'dark' ? 'dark' : 'light'
    applyThemeToDom(theme.value)
    localStorage.setItem(THEME_KEY, theme.value)
  }

  // 在亮/暗之间切换。
  function toggleTheme() {
    setTheme(theme.value === 'dark' ? 'light' : 'dark')
  }

  // 应用启动时调用一次，把已存主题落到 DOM。
  function initTheme() {
    applyThemeToDom(theme.value)
  }

  return { locale, theme, setLocale, setTheme, toggleTheme, initTheme }
})
