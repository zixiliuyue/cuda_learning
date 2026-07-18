<script setup>
// 概览页：展示节点数、在线节点、任务状态统计等关键指标。
import { ref, onMounted } from 'vue'
import { nodeApi, taskApi } from '../api'

const stats = ref({
  totalNodes: 0,
  onlineNodes: 0,
  totalGPUs: 0,
  runningTasks: 0,
  pendingTasks: 0
})

// 加载概览统计数据。
async function loadStats() {
  const [nodes, tasks] = await Promise.all([
    nodeApi.list().catch(() => []),
    taskApi.list().catch(() => [])
  ])
  stats.value.totalNodes = nodes.length
  stats.value.onlineNodes = nodes.filter((n) => n.status === 'online').length
  stats.value.totalGPUs = nodes.reduce((sum, n) => sum + (n.gpus?.length || 0), 0)
  stats.value.runningTasks = tasks.filter((t) => t.status === 'running').length
  stats.value.pendingTasks = tasks.filter((t) => t.status === 'pending').length
}

onMounted(loadStats)
</script>

<template>
  <div>
    <a-grid :cols="5" :col-gap="16">
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
      <p>QingTian（擎天）是基于 CUDA 的算力调度平台，提供 GPU 节点纳管、任务调度、资源隔离与可观测能力。</p>
      <p>请在「节点管理」查看已纳管的 GPU 节点，在「任务管理」提交与跟踪算力任务。</p>
    </a-card>
  </div>
</template>
