// UI 状态集中管理：语言与主题。
// 集中读写 localStorage 与副作用(切 i18n locale、切 arco 暗黑属性)，
// 避免逻辑散落在各组件，保证切换行为一致、可持久化。
import { defineStore } from 'pinia'
import { ref } from 'vue'
import i18n, { LOCALE_KEY } from '../i18n'

// 主题持久化键名。
const THEME_KEY = 'qt_theme'
const THEME_COLOR_KEY = 'qt_theme_color'

// 预置主题配置
// primary 对应 Arco 的主题色--color-primary，mode 是亮暗模式
export const THEME_COLORS = {
  blue_light: {
    name: '主题蓝',
    primary: '#165dff',
    mode: 'light',
  },
  blue_dark: {
    name: '主题蓝(暗)',
    primary: '#165dff',
    mode: 'dark',
  },
  green_light: {
    name: '清新绿',
    primary: '#00b42a',
    mode: 'light',
  },
  green_dark: {
    name: '清新绿(暗)',
    primary: '#00b42a',
    mode: 'dark',
  },
  purple_light: {
    name: '优雅紫',
    primary: '#722ed1',
    mode: 'light',
  },
  purple_dark: {
    name: '优雅紫(暗)',
    primary: '#722ed1',
    mode: 'dark',
  },
  orange_light: {
    name: '活力橙',
    primary: '#ff7d00',
    mode: 'light',
  },
  orange_dark: {
    name: '活力橙(暗)',
    primary: '#ff7d00',
    mode: 'dark',
  },
  red_light: {
    name: '热情红',
    primary: '#f53f3f',
    mode: 'light',
  },
  red_dark: {
    name: '热情红(暗)',
    primary: '#f53f3f',
    mode: 'dark',
  },
  cyan_light: {
    name: '青春青',
    primary: '#14c9c9',
    mode: 'light',
  },
  cyan_dark: {
    name: '青春青(暗)',
    primary: '#14c9c9',
    mode: 'dark',
  },
}

// 数据迁移：旧格式转新格式
function migrateOldTheme() {
  const oldThemeColor = localStorage.getItem(THEME_COLOR_KEY)
  const oldThemeMode = localStorage.getItem(THEME_KEY)
  // 如果已是新格式，直接返回
  if (oldThemeColor && THEME_COLORS[oldThemeColor]) {
    return oldThemeColor
  }
  // 旧格式 color + mode 转为新格式 key
  const color = oldThemeColor && THEME_COLORS[oldThemeColor + '_light'] ? oldThemeColor : 'blue'
  const mode = oldThemeMode === 'dark' ? 'dark' : 'light'
  const newKey = `${color}_${mode}`
  localStorage.setItem(THEME_COLOR_KEY, newKey)
  // 清除旧键，避免下次再迁移
  localStorage.removeItem(THEME_KEY)
  return newKey
}

// 读取初始主题：从存储的主题key中解析mode，否则默认亮色。
function getInitialTheme() {
  const savedTheme = localStorage.getItem(THEME_COLOR_KEY)
  if (savedTheme && THEME_COLORS[savedTheme]) {
    return THEME_COLORS[savedTheme].mode
  }
  return 'light'
}

// 读取初始主题：优先本地存储，否则默认蓝色亮色。自动迁移旧数据。
function getInitialThemeColor() {
  const saved = localStorage.getItem(THEME_COLOR_KEY)
  if (saved && THEME_COLORS[saved]) {
    return saved
  }
  // 尝试迁移旧格式
  const migrated = migrateOldTheme()
  if (migrated && THEME_COLORS[migrated]) {
    return migrated
  }
  return 'blue_light'
}

// 将暗黑属性应用到 body：Arco 官方约定 body[arco-theme=dark] 启用暗色变量。
function applyThemeToDom(theme) {
  if (theme === 'dark') {
    document.body.setAttribute('arco-theme', 'dark')
    document.documentElement.classList.add('arco-dark')
    document.documentElement.style.backgroundColor = '#1d2129'
  } else {
    document.body.removeAttribute('arco-theme')
    document.documentElement.classList.remove('arco-dark')
    document.documentElement.style.backgroundColor = '#ffffff'
  }
}

// 将主题色应用到 html 根元素的 CSS 变量。
function applyThemeColorToDom(colorKey) {
  const color = THEME_COLORS[colorKey]?.primary || THEME_COLORS.blue.primary
  document.documentElement.style.setProperty('--color-primary', color)
  // Arco Design 也需要更新 primary 相关变量
  document.documentElement.style.setProperty('--color-primary-light-1', adjustBrightness(color, 5))
  document.documentElement.style.setProperty('--color-primary-light-2', adjustBrightness(color, 10))
  document.documentElement.style.setProperty('--color-primary-light-3', adjustBrightness(color, 15))
  document.documentElement.style.setProperty('--color-primary-light-4', adjustBrightness(color, 20))
  document.documentElement.style.setProperty('--color-primary-light-5', adjustBrightness(color, 25))
  document.documentElement.style.setProperty('--color-primary-light-6', adjustBrightness(color, 30))
  document.documentElement.style.setProperty('--color-primary-light-7', adjustBrightness(color, 35))
  document.documentElement.style.setProperty('--color-primary-light-8', adjustBrightness(color, 40))
  document.documentElement.style.setProperty('--color-primary-light-9', adjustBrightness(color, 45))
  document.documentElement.style.setProperty('--color-primary-hover', adjustBrightness(color, 5))
  document.documentElement.style.setProperty('--color-primary-active', adjustBrightness(color, -5))
}

// 调整 hex 颜色亮度，生成浅色调色板
function adjustBrightness(hex, percent) {
  // 去掉 # 号
  hex = hex.replace(/^#/, '')
  // 转为 r g b
  let r = parseInt(hex.substring(0, 2), 16)
  let g = parseInt(hex.substring(2, 4), 16)
  let b = parseInt(hex.substring(4, 6), 16)

  // 调亮（增加亮度）
  r = Math.min(255, r + (255 - r) * percent / 100)
  g = Math.min(255, g + (255 - g) * percent / 100)
  b = Math.min(255, b + (255 - b) * percent / 100)

  // 转回 hex
  return '#' + ((r << 16) | (g << 8) | b).toString(16).padStart(6, '0')
}

export const useUiStore = defineStore('ui', () => {
  // 当前语言，初值与 i18n 实例一致(i18n 已从 localStorage 读过)。
  const locale = ref(i18n.global.locale.value)
  // 当前主题：'light' | 'dark'。
  const theme = ref(getInitialTheme())
  // 当前主题色：blue | green | purple | orange | red | cyan。
  const themeColor = ref(getInitialThemeColor())

  // 切换语言：同步 i18n 实例 + 持久化。Arco 组件文案由 App.vue 的
  // ConfigProvider 依据本值切换，无需在此处理。
  function setLocale(next) {
    if (next !== 'zh-CN' && next !== 'en-US') return
    locale.value = next
    i18n.global.locale.value = next
    localStorage.setItem(LOCALE_KEY, next)
  }

  // 设置主题（亮/暗）：更新状态 + DOM 属性 + 持久化。
  function setTheme(next) {
    theme.value = next === 'dark' ? 'dark' : 'light'
    applyThemeToDom(theme.value)
    localStorage.setItem(THEME_KEY, theme.value)
  }

  // 在亮/暗之间切换。
  function toggleTheme() {
    setTheme(theme.value === 'dark' ? 'light' : 'dark')
  }

  // 设置主题：更新颜色 + 亮暗模式 + CSS 变量 + 持久化。
  function setThemeColor(next) {
    if (!THEME_COLORS[next]) return
    themeColor.value = next
    const themeConfig = THEME_COLORS[next]
    setTheme(themeConfig.mode)
    applyThemeColorToDom(next)
    localStorage.setItem(THEME_COLOR_KEY, next)
  }

  // 应用启动时调用一次，把已存主题落到 DOM。
  function initTheme() {
    applyThemeToDom(theme.value)
    applyThemeColorToDom(themeColor.value)
  }

  // 获取所有可用主题列表（用于选择框）
  function getAvailableThemeColors() {
    return Object.entries(THEME_COLORS).map(([key, cfg]) => ({
      key,
      name: cfg.name,
      color: cfg.primary,
      mode: cfg.mode,
    }))
  }

  return {
    locale,
    theme,
    themeColor,
    setLocale,
    setTheme,
    toggleTheme,
    setThemeColor,
    getAvailableThemeColors,
    initTheme,
  }
})
