<script setup>
import { ref, computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { Message, Modal } from '@arco-design/web-vue'
import { scheduleApi, workflowApi } from '../api'

const { t } = useI18n()

const loading = ref(false)
const data = ref([])
const workflows = ref([])
const createModalVisible = ref(false)
const createForm = ref({
  name: '',
  description: '',
  workflowId: '',
  cronExpr: '0 0 * * *',
  timezone: 'Asia/Shanghai',
  parameters: {},
  enabled: true,
})

// 加载工作流列表（供新建调度选择）
async function loadWorkflows() {
  try {
    const res = await workflowApi.list()
    workflows.value = res.items || []
  } catch (e) {
    workflows.value = []
  }
}

async function loadData() {
  loading.value = true
  try {
    const res = await scheduleApi.list()
    data.value = res.items || []
  } catch (e) {
    data.value = []
  } finally {
    loading.value = false
  }
}

function openCreateModal() {
  createForm.value = { name: '', description: '', workflowId: '', cronExpr: '0 0 * * *', timezone: 'Asia/Shanghai', parameters: {}, enabled: true }
  createModalVisible.value = true
}

async function handleCreate() {
  if (!createForm.value.name.trim()) {
    Message.warning(t('common.required'))
    return
  }
  if (!createForm.value.workflowId) {
    Message.warning('请选择关联的工作流')
    return
  }
  try {
    await scheduleApi.create(createForm.value)
    Message.success(t('common.success'))
    createModalVisible.value = false
    loadData()
  } catch (e) {
    // Error handled by request interceptor
  }
}

function handleDelete(item) {
  Modal.confirm({
    title: t('schedule.confirmDelete'),
    content: '',
    onOk: async () => {
      await scheduleApi.delete(item.id)
      Message.success(t('common.success'))
      loadData()
    },
  })
}

function handleTrigger(item) {
  Modal.open({
    title: item.name,
    content: '确认要立即执行一次这个调度吗？',
    onOk: async () => {
      await scheduleApi.trigger(item.id, {})
      Message.success(t('common.success'))
    },
  })
}

onMounted(() => {
  loadData()
  loadWorkflows()
})

const columns = computed(() => [
  {
    title: t('schedule.name'),
    dataIndex: 'name',
  },
  {
    title: t('schedule.workflow'),
    dataIndex: 'workflowName',
  },
  {
    title: t('schedule.cronExpr'),
    dataIndex: 'cronExpr',
  },
  {
    title: t('schedule.status'),
    dataIndex: 'status',
    slotName: 'status',
  },
  {
    title: t('schedule.lastRunAt'),
    dataIndex: 'lastRunAt',
  },
  {
    title: t('schedule.nextRunAt'),
    dataIndex: 'nextRunAt',
  },
  {
    title: t('common.actions'),
    dataIndex: 'actions',
    slotName: 'actions',
    width: 200,
    fixed: 'right',
  },
])
</script>

<template>
  <a-card :title="t('schedule.title')" class="schedule-list-card">
    <template #extra>
      <a-space>
        <a-button :loading="loading" @click="loadData">
          <template #icon><icon-refresh />
          </template>
          {{ t('common.refresh') }}
        </a-button>
        <a-button type="primary" @click="openCreateModal">
          <template #icon><icon-plus />
          </template>
          {{ t('schedule.createBtn') }}
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
      <template #status="{ record }">
        <a-tag :color="record.enabled ? 'green' : 'gray'">
          {{ record.status ? t(`schedule.${record.status.toLowerCase()}`) : '-' }}
        </a-tag>
      </template>
      <template #actions="{ record }">
        <a-space>
          <a-button size="small" @click="() => handleTrigger(record)">
            <template #icon><icon-play-circle />
            </template>
            {{ t('schedule.triggerBtn') }}
          </a-button>
          <a-button size="small" status="danger" @click="() => handleDelete(record)">
            <template #icon><icon-delete />
            </template>
            {{ t('common.delete') }}
          </a-button>
        </a-space>
      </template>
    </a-table>

    <a-modal v-model:visible="createModalVisible" :title="t('schedule.createBtn')" @ok="handleCreate" width="520px">
      <a-form :model="createForm" layout="vertical">
        <a-form-item field="name" :label="t('schedule.name')" required>
          <a-input v-model="createForm.name" placeholder="请输入调度名称" />
        </a-form-item>
        <a-form-item field="workflowId" :label="t('schedule.workflow')" required>
          <a-select v-model="createForm.workflowId" placeholder="请选择关联的工作流">
            <a-option v-for="wf in workflows" :key="wf.id" :value="wf.id">{{ wf.name }}</a-option>
          </a-select>
        </a-form-item>
        <a-form-item field="cronExpr" :label="t('schedule.cronExpr')" required>
          <a-input v-model="createForm.cronExpr" placeholder="秒 分 时 日 月 周，如 0 0 2 * * * 表示每天2点" />
        </a-form-item>
        <a-form-item field="description" :label="t('schedule.description') || '描述'">
          <a-textarea v-model="createForm.description" :rows="2" placeholder="可选" />
        </a-form-item>
        <a-form-item field="enabled" label="启用">
          <a-switch v-model="createForm.enabled" />
        </a-form-item>
      </a-form>
    </a-modal>
  </a-card>
</template>

<style scoped>
.schedule-list-card {
  margin: 0;
}
</style>
