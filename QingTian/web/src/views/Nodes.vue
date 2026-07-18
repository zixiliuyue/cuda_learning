<script setup>
// 节点管理页：展示所有 GPU 节点及其 GPU 明细、在线状态。
import { ref, onMounted } from 'vue'
import { nodeApi } from '../api'

const nodes = ref([])
const loading = ref(false)

const columns = [
  { title: '节点 ID', dataIndex: 'id' },
  { title: '主机名', dataIndex: 'hostname' },
  { title: 'IP', dataIndex: 'ip' },
  { title: '状态', slotName: 'status' },
  { title: 'GPU 数', slotName: 'gpuCount' },
  { title: '驱动/CUDA', slotName: 'versions' }
]

// 加载节点列表。
async function loadNodes() {
  loading.value = true
  try {
    nodes.value = await nodeApi.list()
  } catch {
    nodes.value = []
  } finally {
    loading.value = false
  }
}

onMounted(loadNodes)
</script>

<template>
  <a-card title="GPU 节点">
    <template #extra>
      <a-button type="primary" @click="loadNodes" :loading="loading">刷新</a-button>
    </template>
    <a-table :data="nodes" :columns="columns" :loading="loading" row-key="id" :pagination="false">
      <template #status="{ record }">
        <a-tag :color="record.status === 'online' ? 'green' : 'red'">
          {{ record.status === 'online' ? '在线' : '离线' }}
        </a-tag>
      </template>
      <template #gpuCount="{ record }">
        {{ record.gpus ? record.gpus.length : 0 }}
      </template>
      <template #versions="{ record }">
        {{ record.driver_version }} / {{ record.cuda_version }}
      </template>
    </a-table>
  </a-card>
</template>
