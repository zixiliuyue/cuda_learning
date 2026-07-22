<script setup>
// 语言 + 主题切换控件：登录页与主框架 header 复用，避免重复实现。
// 语言用下拉选择，主题包含颜色+亮暗在一个下拉选择。
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { IconLanguage, IconPalette } from '@arco-design/web-vue/es/icon'
import { useUiStore } from '../store/ui'
import { SUPPORTED_LOCALES } from '../i18n'

const { t } = useI18n()
const ui = useUiStore()

// 语言下拉当前值(双向)：读 store.locale，写走 setLocale。
const currentLocale = computed({
  get: () => ui.locale,
  set: (v) => ui.setLocale(v),
})

// 当前主题（包含颜色+亮暗）
const currentThemeColor = computed({
  get: () => ui.themeColor,
  set: (v) => ui.setThemeColor(v),
})

// 可用主题列表
const themeColorOptions = ui.getAvailableThemeColors()
</script>

<template>
  <a-space :size="12">
    <!-- 语言选择：图标 + 下拉 -->
    <a-select
      v-model="currentLocale"
      :bordered="false"
      size="small"
      :style="{ width: '120px' }"
      position="bl"
    >
      <template #prefix><icon-language /></template>
      <a-option v-for="l in SUPPORTED_LOCALES" :key="l.value" :value="l.value">
        {{ l.label }}
      </a-option>
    </a-select>

    <!-- 主题选择：调色板图标 + 下拉选择（颜色+亮暗合一） -->
    <a-tooltip content="主题">
      <a-select
        v-model="currentThemeColor"
        :bordered="false"
        size="small"
        :style="{ width: '180px' }"
        position="bl"
        :dropdown-style="{ minWidth: '180px' }"
      >
        <template #prefix>
          <icon-palette :style="{ color: themeColorOptions.find(c => c.key === currentThemeColor)?.color, fontSize: '16px' }" />
        </template>
        <a-option v-for="c in themeColorOptions" :key="c.key" :value="c.key">
          <a-space :size="8">
            <div
              style="width: 18px; height: 18px; border-radius: 50%; background: {{ c.color }}; border: 1px solid rgba(0, 0, 0, 0.1);"
            ></div>
            <span>{{ c.name }}</span>
          </a-space>
        </a-option>
      </a-select>
    </a-tooltip>
  </a-space>
</template>
