<script setup>
// 计量计费页面：展示全局统计和当前用户的用量历史。
import { ref, computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { meteringApi, quotaApi } from '../api'
import { useAuthStore } from '../store/auth'

const { t } = useI18n()
const auth = useAuthStore()

const loading = ref(false)
const stats = ref({
  totalGPUHours: 0,
  totalCost: 0,
  totalTasks: 0,
})
const myUsage = ref([])
const myQuota = ref(null)

// 列定义使用 computed 而非 <a-table-column> 子组件，避免 Arco 组件注册问题。
const usageColumns = computed(() => [
  { title: t('metering.date'), dataIndex: 'date', width: 180 },
  { title: t('metering.taskCount'), dataIndex: 'taskCount', width: 120 },
  { title: t('metering.gpuHours'), dataIndex: 'gpuHours', width: 120 },
  { title: t('metering.cost'), dataIndex: 'cost', width: 120 },
])

// 加载计量看板和我的配额。
async function loadData() {
  loading.value = true
  try {
    const data = await meteringApi.dashboard()
    stats.value = {
      totalGPUHours: data.totalGPUHours || 0,
      totalCost: data.totalCost || 0,
      totalTasks: data.totalTasks || 0,
    }
    myUsage.value = data.myUsage || []
    // 加载我的配额
    if (auth.username) {
      myQuota.value = await quotaApi.get(auth.username)
    }
  } catch (e) {
    // 错误已拦截
  } finally {
    loading.value = false
  }
}

onMounted(loadData)
</script>

<template>
  <a-spin :loading="loading" style="display: block">
    <a-grid :cols="{ xs: 1, sm: 2, md: 3 }" :col-gap="16" :row-gap="16">
      <!-- 全局统计卡片 -->
      <a-grid-item>
        <a-card><a-statistic :title="t('metering.totalGPUHours')" :value="stats.totalGPUHours" :value-formatter="
          (val) => val.toFixed(2)
        " /></a-card>
      </a-grid-item>
      <a-grid-item>
        <a-card><a-statistic :title="t('metering.totalCost')" :value="stats.totalCost" :value-formatter="
          (val) => val.toFixed(2)
        " /></a-card>
      </a-grid-item>
      <a-grid-item>
        <a-card><a-statistic :title="t('metering.totalTasks')" :value="stats.totalTasks" /></a-card>
      </a-grid-item>
    </a-grid>

    <!-- 我的配额 -->
    <a-card :title="t('quota.title')" style="margin-top: 16px">
      <a-space direction="vertical" size="large" style="width: 100%">
        <a-row :gutter="16">
          <a-col :span="8">
            <a-statistic :title="t('quota.maxGPUHours')" :value="myQuota?.maxGPUHours ?? 0" :value-formatter="
              (val) => val <= 0 ? t('quota.unlimited') : val.toFixed(2)
            " />
          </a-col>
          <a-col :span="8">
            <a-statistic :title="t('quota.usedGPUHours')" :value="myQuota?.usedGPUHours ?? 0" :value-formatter="
              (val) => val.toFixed(2)
            " />
          </a-col>
          <a-col :span="8">
            <a-statistic :title="t('quota.remainingGPUHours')" :value-formatter="
              () => {
                if (!myQuota || myQuota.maxGPUHours <= 0) {
                  return t('quota.unlimited')
                }
                const remaining = myQuota.maxGPUHours - (myQuota.usedGPUHours ?? 0)
                return remaining.toFixed(2)
              }
            " />
          </a-col>
        </a-row>
        <a-row :gutter="16">
          <a-col :span="12">
            <a-statistic :title="t('quota.maxConcurrent')" :value="myQuota?.maxConcurrent ?? 0" :value-formatter="
              (val) => val <= 0 ? t('quota.unlimited') : String(val)
            " />
          </a-col>
          <a-col :span="12">
            <a-statistic :title="t('quota.currentConcurrent')" :value="myQuota?.currentConcurrent ?? 0" />
          </a-col>
        </a-row>
      </a-space>
    </a-card>

    <!-- 我的用量列表 -->
    <a-card :title="t('metering.myUsage')" style="margin-top: 16px">
      <a-table :data="myUsage" :columns="usageColumns" :loading="loading" row-key="date" :pagination="{ pageSize: 10 }" />
    </a-card>
  </a-spin>
</template>

<style scoped>
</style>
