<script setup>
import { ref, computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { Message, Modal } from '@arco-design/web-vue'
import { workflowApi } from '../api'
import IconDelete from '@arco-design/web-vue/es/icon/icon-delete'
import IconEdit from '@arco-design/web-vue/es/icon/icon-edit'
import IconPlayCircle from '@arco-design/web-vue/es/icon/icon-play-circle'
import IconList from '@arco-design/web-vue/es/icon/icon-list'

const { t } = useI18n()

const loading = ref(false)
const data = ref([])

async function loadData() {
  loading.value = true
  try {
    const res = await workflowApi.list()
    data.value = res.items || []
  } catch (e) {
    data.value = []
  } finally {
    loading.value = false
  }
}

function handleDelete(item) {
  Modal.confirm({
    title: t('workflow.confirmDelete'),
    content: '',
    onOk: async () => {
      await workflowApi.delete(item.id)
      Message.success(t('common.success'))
      loadData()
    },
  })
}

function goToEditor(id) {
  window.location.href = `#/workflows/editor/${id}`
}

function goToRuns(id) {
  window.location.href = `#/workflows/runs/${id}`
}

function handleRun(item) {
  Modal.open({
    title: item.name,
    content: '确认要立即执行这个工作流吗？',
    onOk: async () => {
      await workflowApi.run(item.id, {})
      Message.success(t('common.success'))
      goToRuns(item.id)
    },
  })
}

onMounted(() => {
  loadData()
})

const columns = computed(() => [
  {
    title: t('workflow.name'),
    dataIndex: 'name',
  },
  {
    title: t('workflow.description'),
    dataIndex: 'description',
  },
  {
    title: t('workflow.status'),
    dataIndex: 'status',
  },
  {
    title: t('workflow.creator'),
    dataIndex: 'createdBy',
  },
  {
    title: t('workflow.createTime'),
    dataIndex: 'createdAt',
  },
  {
    title: t('common.actions'),
    dataIndex: 'actions',
    fixed: 'right',
    width: 200,
  },
])
</script>

<template>
  <a-card :title="t('workflow.title')" class="g-card">
    <template #extra>
      <a-space>
        <a-button :loading="loading" @click="loadData">
          <template #icon><icon-refresh />
          </template>
          {{ t('common.refresh') }}
        </a-button>
        <a-button type="primary" @click="() => goToEditor('new')">
          <template #icon><icon-plus />
          </template>
          {{ t('workflow.createBtn') }}
        </a-button>
      </a-space>
    </template>
    <a-table
      :loading="loading"
      :data="data"
      :columns="columns"
      :pagination="{ pageSize: 10 }"
      row-key="id"
    >
      <template #actions="{ record }">
        <a-space>
          <a-button size="small" type="primary" @click="() => handleRun(record)">
            <template #icon><icon-play-circle />
            </template>
            {{ t('workflow.runBtn') }}
          </a-button>
          <a-button size="small" @click="() => goToRuns(record.id)">
            <template #icon><icon-list />
            </template>
            {{ t('workflow.viewRuns') }}
          </a-button>
          <a-button size="small" @click="() => goToEditor(record.id)">
            <template #icon><icon-edit />
            </template>
            {{ t('common.edit') }}
          </a-button>
          <a-button size="small" status="danger" @click="() => handleDelete(record)">
            <template #icon><icon-delete />
            </template>
            {{ t('common.delete') }}
          </a-button>
        </a-space>
      </template>
    </a-table>
  </a-card>
</template>

<style scoped>
.g-card {
  margin: 0;
}
</style>
