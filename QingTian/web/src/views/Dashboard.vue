<script setup>
// 概览页：展示节点数、在线节点、任务状态统计等关键指标。
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { nodeApi, taskApi } from '../api'

const { t } = useI18n()
const loading = ref(false)
const stats = ref({
  totalNodes: 0,
  onlineNodes: 0,
  totalGPUs: 0,
  runningTasks: 0,
  pendingTasks: 0
})

// 加载概览统计数据。单个接口失败降级为空，避免整页崩溃。
async function loadStats() {
  loading.value = true
  try {
    const [nodes, tasks] = await Promise.all([
      nodeApi.list().catch(() => []),
      taskApi.list().catch(() => [])
    ])
    const nodeList = Array.isArray(nodes) ? nodes : []
    const taskList = Array.isArray(tasks) ? tasks : []
    stats.value = {
      totalNodes: nodeList.length,
      onlineNodes: nodeList.filter((n) => n.status === 'online').length,
      totalGPUs: nodeList.reduce((sum, n) => sum + (n.gpus?.length || 0), 0),
      runningTasks: taskList.filter((t) => t.status === 'running').length,
      pendingTasks: taskList.filter((t) => t.status === 'pending').length
    }
  } finally {
    loading.value = false
  }
}

onMounted(loadStats)
</script>

<template>
  <a-spin :loading="loading" style="display: block">
    <a-grid :cols="{ xs: 1, sm: 2, md: 3, lg: 5 }" :col-gap="16" :row-gap="16">
      <a-grid-item>
        <a-card><a-statistic :title="t('dashboard.totalNodes')" :value="stats.totalNodes" /></a-card>
      </a-grid-item>
      <a-grid-item>
        <a-card><a-statistic :title="t('dashboard.onlineNodes')" :value="stats.onlineNodes" /></a-card>
      </a-grid-item>
      <a-grid-item>
        <a-card><a-statistic :title="t('dashboard.totalGPUs')" :value="stats.totalGPUs" /></a-card>
      </a-grid-item>
      <a-grid-item>
        <a-card><a-statistic :title="t('dashboard.runningTasks')" :value="stats.runningTasks" /></a-card>
      </a-grid-item>
      <a-grid-item>
        <a-card><a-statistic :title="t('dashboard.pendingTasks')" :value="stats.pendingTasks" /></a-card>
      </a-grid-item>
    </a-grid>
    <a-card style="margin-top: 16px" :title="t('dashboard.platformTitle')">
      <template #extra>
        <a-button size="small" @click="loadStats" :loading="loading">{{ t('common.refresh') }}</a-button>
      </template>
      <p>{{ t('dashboard.intro1') }}</p>
      <p>{{ t('dashboard.intro2') }}</p>
    </a-card>
  </a-spin>
</template>
