<script setup>
// 节点管理页：展示所有 GPU 节点及其 GPU 明细、在线状态。
import { ref, computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { nodeApi } from '../api'

const { t } = useI18n()
const nodes = ref([])
const loading = ref(false)

// 列定义用 computed，语言切换时表头随之更新。
const columns = computed(() => [
  { title: t('nodes.colId'), dataIndex: 'id' },
  { title: t('nodes.colHostname'), dataIndex: 'hostname' },
  { title: t('nodes.colIp'), dataIndex: 'ip' },
  { title: t('nodes.colStatus'), slotName: 'status' },
  { title: t('nodes.colGpuCount'), slotName: 'gpuCount' },
  { title: t('nodes.colVersions'), slotName: 'versions' }
])

// 加载节点列表。
async function loadNodes() {
  loading.value = true
  try {
    const data = await nodeApi.list()
    nodes.value = Array.isArray(data) ? data : []
  } catch {
    nodes.value = []
  } finally {
    loading.value = false
  }
}

onMounted(loadNodes)
</script>

<template>
  <a-card :title="t('nodes.title')">
    <template #extra>
      <a-button type="primary" @click="loadNodes" :loading="loading">{{ t('common.refresh') }}</a-button>
    </template>
    <a-table :data="nodes" :columns="columns" :loading="loading" row-key="id" :pagination="false">
      <template #status="{ record }">
        <a-tag :color="record.status === 'online' ? 'green' : 'red'">
          {{ record.status === 'online' ? t('nodes.online') : t('nodes.offline') }}
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
