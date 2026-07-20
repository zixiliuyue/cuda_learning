<script setup>
// 根组件：承载路由出口，并通过 ConfigProvider 同步 Arco 组件的内置文案语言。
import { computed } from 'vue'
import zhCN from '@arco-design/web-vue/es/locale/lang/zh-cn'
import enUS from '@arco-design/web-vue/es/locale/lang/en-us'
import { useUiStore } from './store/ui'

const ui = useUiStore()

// 根据当前语言选择 Arco 组件库内置文案(分页、空状态、日期选择等)。
const arcoLocale = computed(() => (ui.locale === 'en-US' ? enUS : zhCN))
</script>

<template>
  <a-config-provider :locale="arcoLocale">
    <router-view />
  </a-config-provider>
</template>

<style>
html, body, #app {
  height: 100%;
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
}
</style>
