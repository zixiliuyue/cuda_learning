<script setup>
import { ref, computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { Message, Modal } from '@arco-design/web-vue'
import { scheduleApi } from '../api'
import IconDelete from '@arco-design/web-vue/es/icon/icon-delete'
import IconEdit from '@arco-design/web-vue/es/icon/icon-edit'
import IconPlayCircle from '@arco-design/web-vue/es/icon/icon-play-circle'

const { t } = useI18n()

const loading = ref(false)
const data = ref([])

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
        <a-button type="primary" @click="() => window.location.href = '#/workflows'">
          <template #icon><icon-add />
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
          {{ t(`schedule.${record.status.toLowerCase()}`) }}
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
  </a-card>
</template>

<style scoped>
.schedule-list-card {
  margin: 0;
}
</style>
