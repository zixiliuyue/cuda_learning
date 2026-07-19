<script setup>
// 概览页：展示节点数、在线节点、任务状态统计等关键指标。
import { ref, onMounted } from 'vue'
import { nodeApi, taskApi } from '../api'

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
        <a-card><a-statistic title="节点总数" :value="stats.totalNodes" /></a-card>
      </a-grid-item>
      <a-grid-item>
        <a-card><a-statistic title="在线节点" :value="stats.onlineNodes" /></a-card>
      </a-grid-item>
      <a-grid-item>
        <a-card><a-statistic title="GPU 总数" :value="stats.totalGPUs" /></a-card>
      </a-grid-item>
      <a-grid-item>
        <a-card><a-statistic title="运行中任务" :value="stats.runningTasks" /></a-card>
      </a-grid-item>
      <a-grid-item>
        <a-card><a-statistic title="待调度任务" :value="stats.pendingTasks" /></a-card>
      </a-grid-item>
    </a-grid>
    <a-card style="margin-top: 16px" title="平台说明">
      <template #extra>
        <a-button size="small" @click="loadStats" :loading="loading">刷新</a-button>
      </template>
      <p>QingTian（擎天）是基于 CUDA 的算力调度平台，提供 GPU 节点纳管、任务调度、资源隔离与可观测能力。</p>
      <p>请在「节点管理」查看已纳管的 GPU 节点，在「任务管理」提交与跟踪算力任务。</p>
    </a-card>
  </a-spin>
</template>
