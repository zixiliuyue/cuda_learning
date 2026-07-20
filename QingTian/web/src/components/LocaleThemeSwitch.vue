<script setup>
// 语言 + 主题切换控件：登录页与主框架 header 复用，避免重复实现。
// 语言用下拉选择，主题用图标按钮在亮/暗间切换。
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { IconMoonFill, IconSunFill, IconLanguage } from '@arco-design/web-vue/es/icon'
import { useUiStore } from '../store/ui'
import { SUPPORTED_LOCALES } from '../i18n'

const { t } = useI18n()
const ui = useUiStore()

// 语言下拉当前值(双向)：读 store.locale，写走 setLocale。
const currentLocale = computed({
  get: () => ui.locale,
  set: (v) => ui.setLocale(v),
})

// 是否暗色，用于切换图标与提示文案。
const isDark = computed(() => ui.theme === 'dark')
</script>

<template>
  <a-space :size="8">
    <!-- 语言选择：图标 + 下拉 -->
    <a-select
      v-model="currentLocale"
      :bordered="false"
      size="small"
      :style="{ width: '120px' }"
    >
      <template #prefix><icon-language /></template>
      <a-option v-for="l in SUPPORTED_LOCALES" :key="l.value" :value="l.value">
        {{ l.label }}
      </a-option>
    </a-select>

    <!-- 主题切换：亮色显示月亮(点击转暗)，暗色显示太阳(点击转亮) -->
    <a-tooltip :content="isDark ? t('switcher.toLight') : t('switcher.toDark')">
      <a-button shape="circle" size="small" @click="ui.toggleTheme()">
        <template #icon>
          <icon-sun-fill v-if="isDark" />
          <icon-moon-fill v-else />
        </template>
      </a-button>
    </a-tooltip>
  </a-space>
</template>
