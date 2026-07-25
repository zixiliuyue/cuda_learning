<script setup>
import { ref, computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute } from 'vue-router'
import { workflowApi } from '../api'

const { t } = useI18n()
const route = useRoute()
const runId = route.params.id

const loading = ref(true)
const run = ref(null)

async function loadDetail() {
  loading.value = true
  try {
    run.value = await workflowApi.getRun(runId)
  } catch (e) {
    console.error(e)
  } finally {
    loading.value = false
  }
}

const statusBadge = (status) => {
  const map = {
    Pending: 'warning',
    Running: 'processing',
    Succeeded: 'success',
    Failed: 'danger',
    Canceled: 'default',
  }
  return map[status] || 'default'
}

onMounted(() => {
  loadDetail()
})
</script>

<template>
  <a-card v-if="!loading" :title="`${t('workflowRun.runTitle')} #${run.id}`" class="run-detail-card">
    <a-descriptions :column="2" border :data="run">
      <a-descriptions-item :label="t('workflow.name')" :content="run.name" />
      <a-descriptions-item :label="t('workflow.status')">
        <a-tag :color="statusBadge(run.status)">{{ t(`workflowRun.${run.status.toLowerCase()}`) }}</a-tag>
      </a-descriptions-item>
      <a-descriptions-item :label="t('workflowRun.triggerType')" :content="t(`workflowRun.${run.trigger.type.toLowerCase()}`)" />
      <a-descriptions-item :label="t('workflowRun.createdBy')" :content="run.createdBy" />
      <a-descriptions-item :label="t('workflowRun.startTime')" :content="run.startedAt" />
      <a-descriptions-item :label="t('workflowRun.endTime')" :content="run.finishedAt" />
      <a-descriptions-item :label="t('workflowRun.duration')" :content="run.duration" />
    </a-descriptions>

    <a-divider />
    <div class="task-runs">
      <a-card
        v-for="taskRun in run.taskRuns"
        :key="taskRun.id"
        :class="['task-run-card']"
        bordered
        size="small"
      >
        <a-row :gutter="12">
          <a-col :span="12">
            <div class="item-title">{{ t('workflowNode.name') }}</div>
            <div class="item-content">{{ taskRun.taskName }}</div>
          </a-col>
          <a-col :span="6">
            <div class="item-title">{{ t('workflowRun.status') }}</div>
            <div class="item-content">
              <a-tag :color="statusBadge(taskRun.status)">{{ t(`workflowRun.${taskRun.status.toLowerCase()}`) }}</a-tag>
            </div>
          </a-col>
          <a-col :span="6" v-if="taskRun.duration">
            <div class="item-title">{{ t('workflowRun.duration') }}</div>
            <div class="item-content">{{ taskRun.duration }}</div>
          </a-col>
        </a-row>
      </a-card>
    </div>
  </a-card>
</template>

<style scoped>
.run-detail-card {
  margin: 0;
}

.task-runs {
  margin-top: 16px;
  display: grid;
  grid-template-columns: repeat(1, 1fr);
  gap: 12px;
}

.task-run-card {
  .item-title {
  font-size: 12px;
  color: var(--color-text-3);
  margin-bottom: 4px;
}

.item-content {
  font-size: 14px;
  color: var(--color-text-1);
}
}
</style>
