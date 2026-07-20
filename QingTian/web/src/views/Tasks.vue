<script setup>
// 任务管理页：展示任务列表、支持提交新任务与取消任务。
import { ref, computed, reactive, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { Message, Modal, Drawer } from '@arco-design/web-vue'
import { taskApi, datasetApi, projectApi } from '../api'

const { t } = useI18n()
const tasks = ref([])
const loading = ref(false)
const submitting = ref(false)
const modalVisible = ref(false)

// 项目列表，用于提交任务时选择项目
const projects = ref([])
async function loadProjects() {
  try {
    projects.value = await projectApi.list()
  } catch (e) {
    projects.value = []
  }
}

// 任务详情抽屉
const detailDrawerVisible = ref(false)
const currentTask = ref(null)
const taskLogs = ref([])
let eventSource = null

// 数据集上传弹窗
const uploadModalVisible = ref(false)
const selectedFile = ref(null)
const uploading = ref(false)

// 新任务表单。
const form = reactive({
  name: '',
  image: '',
  command: '',
  priority: 0,
  gpu_count: 1,
  memory_mb: 8192,
  gpu_model: '',
  timeout_seconds: 0,
  preemptible: false,
  project_id: '',
})

// 列定义用 computed，语言切换时表头随之更新。
const columns = computed(() => [
  { title: t('tasks.colId'), dataIndex: 'id', ellipsis: true, tooltip: true, width: 200 },
  { title: t('tasks.colName'), dataIndex: 'name' },
  { title: t('tasks.colImage'), dataIndex: 'image' },
  { title: t('tasks.colStatus'), slotName: 'status' },
  { title: t('tasks.colNode'), dataIndex: 'node_id' },
  { title: t('tasks.colActions'), slotName: 'actions' },
])

// 状态到标签颜色的映射。
const statusColor = {
  pending: 'gray',
  scheduled: 'blue',
  running: 'arcoblue',
  succeeded: 'green',
  failed: 'red',
  canceled: 'orange',
}

// 打开数据集上传弹窗。
function openUploadModal() {
  selectedFile.value = null
  uploadModalVisible.value = true
}

// 选择文件变化。
function onFileChange(e) {
  selectedFile.value = e.target.files?.[0] || null
}

// 执行上传：先获取预签名 URL，再 PUT 直传到 MinIO。
async function handleUpload() {
  if (!selectedFile.value) {
    Message.warning(t('dataset.noFile'))
    return
  }
  uploading.value = true
  try {
    const { url } = await datasetApi.presignUpload()
    // PUT 上传
    await fetch(url, {
      method: 'PUT',
      body: selectedFile.value,
    })
    Message.success(t('dataset.uploadSuccess'))
    uploadModalVisible.value = false
    // 回填镜像路径：datasets/当前用户名/文件名
    const username = JSON.parse(localStorage.getItem('qt_user') || '{}').username || ''
    form.image = `datasets/${username}/${selectedFile.value.name}`
  } catch (e) {
    Message.error(t('dataset.uploadFailed'))
  } finally {
    uploading.value = false
  }
}

// 打开任务详情抽屉。
function openDetail(task) {
  currentTask.value = task
  taskLogs.value = []
  detailDrawerVisible.value = true
  // 先加载历史日志
  loadHistoryLogs(task.id)
  // 连接 SSE 实时日志流
  connectSSE(task.id)
}

// 加载历史日志。
async function loadHistoryLogs(taskId) {
  try {
    const logs = await taskApi.logs(taskId)
    taskLogs.value = logs || []
  } catch (e) {
    taskLogs.value = []
  }
}

// 连接 SSE 实时日志流。
function connectSSE(taskId) {
  // 先关闭之前的连接
  if (eventSource) {
    eventSource.close()
    eventSource = null
  }
  // SSE 需要带上 token 认证
  const token = localStorage.getItem('qt_token')
  const url = `/api/v1/tasks/${taskId}/logs/stream?token=${encodeURIComponent(token)}`
  const es = new EventSource(url)
  eventSource = es
  es.onmessage = (event) => {
    // 服务端逐行发送，每一行是一条日志
    if (event.data) {
      taskLogs.value.push(event.data)
      // 自动滚动到底部
      setTimeout(() => {
        const el = document.getElementById('task-log-container')
        if (el) {
          el.scrollTop = el.scrollHeight
        }
      }, 50)
    }
  }
  es.onerror = () => {
    // 连接失败/关闭，不再重试
    if (eventSource) {
      eventSource.close()
      eventSource = null
    }
  }
}

// 关闭详情抽屉时断开 SSE 连接。
function onDetailClose() {
  if (eventSource) {
    eventSource.close()
    eventSource = null
  }
  detailDrawerVisible.value = false
}

// 恢复已取消/失败任务。
function resumeTask(id) {
  Modal.confirm({
    title: t('taskDetail.resumeTask'),
    content: t('taskDetail.resumeConfirm'),
    onOk: async () => {
      await taskApi.resume(id)
      Message.success(t('common.success'))
      loadTasks()
    },
  })
}

// 下载任务结果文件。
async function downloadResult() {
  if (!currentTask.value?.id) return
  try {
    const { url } = await taskApi.resultURL(currentTask.value.id)
    // 直接跳转下载，后端预签名已经过期了也会重定向，前端不用处理签名
    window.open(url, '_blank')
  } catch (e) {
    // 错误已拦截
  }
}

// 加载任务列表。
async function loadTasks() {
  loading.value = true
  try {
    tasks.value = await taskApi.list()
  } catch {
    tasks.value = []
  } finally {
    loading.value = false
  }
}

// 打开提交任务弹窗。
function openModal() {
  Object.assign(form, { name: '', image: '', command: '', priority: 0, gpu_count: 1, memory_mb: 8192, gpu_model: '', timeout_seconds: 0, preemptible: false, project_id: '' })
  modalVisible.value = true
}

// 提交任务。返回 false 可阻止弹窗在校验失败/请求出错时关闭。
async function submitTask() {
  if (!form.name || !form.image) {
    Message.warning(t('tasks.needNameImage'))
    return false
  }
  submitting.value = true
  try {
    await taskApi.submit({
      name: form.name,
      image: form.image,
      command: form.command,
      priority: form.priority,
      timeout_seconds: form.timeout_seconds,
      preemptible: form.preemptible,
      project_id: form.project_id,
      resource: {
        gpu_count: form.gpu_count,
        memory_mb: form.memory_mb,
        gpu_model: form.gpu_model,
      },
    })
    Message.success(t('tasks.submitSuccess'))
    modalVisible.value = false
    loadTasks()
    return true
  } catch {
    // 错误已由拦截器提示，保持弹窗打开方便重试。
    return false
  } finally {
    submitting.value = false
  }
}

// 取消任务，先二次确认避免误操作。
function cancelTask(id) {
  Modal.warning({
    title: t('tasks.cancelTitle'),
    content: t('tasks.cancelContent'),
    hideCancel: false,
    onOk: async () => {
      try {
        await taskApi.cancel(id)
        Message.success(t('tasks.canceled'))
        loadTasks()
      } catch {
        // 已提示。
      }
    },
  })
}

onMounted(() => {
  loadTasks()
  loadProjects()
})
</script>

<template>
  <a-card :title="t('tasks.title')">
    <template #extra>
      <a-space>
        <a-button @click="loadTasks" :loading="loading">{{ t('common.refresh') }}</a-button>
        <a-button type="primary" @click="openUploadModal">{{ t('dataset.title') }}</a-button>
        <a-button type="primary" @click="openModal">{{ t('tasks.submitBtn') }}</a-button>
      </a-space>
    </template>

    <a-table :data="tasks" :columns="columns" :loading="loading" row-key="id" :pagination="{ pageSize: 10 }">
      <template #status="{ record }">
        <a-tag :color="statusColor[record.status] || 'gray'">{{ record.status }}</a-tag>
      </template>
      <template #actions="{ record }">
        <a-space>
          <a-button size="mini" @click="openDetail(record)">{{ t('taskDetail.title') }}</a-button>
          <a-button
            size="mini"
            status="danger"
            :disabled="['succeeded', 'failed', 'canceled'].includes(record.status)"
            @click="cancelTask(record.id)"
          >{{ t('common.cancel') }}</a-button>
        </a-space>
      </template>
    </a-table>

    <!-- 提交新任务弹窗 -->
    <a-modal
      v-model:visible="modalVisible"
      :title="t('tasks.modalTitle')"
      :on-before-ok="submitTask"
      :confirm-loading="submitting"
      :mask-closable="false"
      :ok-text="t('common.submit')"
      :cancel-text="t('common.cancel')"
    >
      <a-form :model="form" layout="vertical">
        <a-form-item field="name" :label="t('tasks.fieldName')" required>
          <a-input v-model="form.name" :placeholder="t('tasks.fieldNamePlaceholder')" />
        </a-form-item>
        <a-form-item field="image" :label="t('tasks.fieldImage')" required>
          <a-input v-model="form.image" :placeholder="t('tasks.fieldImagePlaceholder')" />
          <a-space style="margin-top: 8px">
            <a-button @click="openUploadModal">{{ t('dataset.upload') }}</a-button>
            <div v-if="form.image">{{ form.image }}</div>
          </a-space>
        </a-form-item>
        <a-form-item field="command" :label="t('tasks.fieldCommand')">
          <a-input v-model="form.command" :placeholder="t('tasks.fieldCommandPlaceholder')" />
        </a-form-item>
        <a-grid :cols="2" :col-gap="16">
          <a-grid-item>
            <a-form-item :label="t('tasks.fieldGpuCount')">
              <a-input-number v-model="form.gpu_count" :min="1" />
            </a-form-item>
          </a-grid-item>
          <a-grid-item>
            <a-form-item :label="t('tasks.fieldMemory')">
              <a-input-number v-model="form.memory_mb" :min="512" :step="512" />
            </a-form-item>
          </a-grid-item>
        </a-grid>
        <a-grid :cols="2" :col-gap="16">
          <a-grid-item>
            <a-form-item :label="t('tasks.fieldPriority')">
              <a-input-number v-model="form.priority" :min="0" />
            </a-form-item>
          </a-grid-item>
          <a-grid-item>
            <a-form-item :label="t('tasks.fieldGpuModel')">
              <a-input v-model="form.gpu_model" :placeholder="t('tasks.fieldGpuModelPlaceholder')" />
            </a-form-item>
          </a-grid-item>
        </a-grid>
        <a-grid :cols="2" :col-gap="16">
          <a-grid-item>
            <a-form-item :label="t('tasks.fieldTimeout')">
              <a-input-number v-model="form.timeout_seconds" :min="0" :step="60" />
            </a-form-item>
          </a-grid-item>
          <a-grid-item>
            <a-form-item :label="t('tasks.fieldPreemptible')">
              <a-switch v-model="form.preemptible" />
            </a-form-item>
          </a-grid-item>
        </a-grid>
      </a-form>
    </a-modal>

    <!-- 数据集上传弹窗 -->
    <a-modal
      v-model:visible="uploadModalVisible"
      :title="t('dataset.title')"
      @ok="handleUpload"
      :ok-text="t('common.upload')"
      :cancel-text="t('common.cancel')"
    >
      <a-form layout="vertical">
        <a-form-item :label="t('dataset.selectFile')">
          <input type="file" @change="onFileChange" />
        </a-form-item>
      </a-form>
    </a-modal>

    <!-- 任务详情抽屉 -->
    <a-drawer
      v-model:visible="detailDrawerVisible"
      :title="t('taskDetail.title')"
      width="60%"
      @close="onDetailClose"
    >
      <a-descriptions :column="2" bordered v-if="currentTask">
        <a-descriptions-item :label="t('taskDetail.id')">{{ currentTask.id }}</a-descriptions-item>
        <a-descriptions-item :label="t('taskDetail.name')">{{ currentTask.name }}</a-descriptions-item>
        <a-descriptions-item :label="t('taskDetail.image')">{{ currentTask.image }}</a-descriptions-item>
        <a-descriptions-item :label="t('taskDetail.command')">{{ currentTask.command }}</a-descriptions-item>
        <a-descriptions-item :label="t('taskDetail.status')">
          <a-tag :color="statusColor[currentTask.status] || 'gray'">{{ currentTask.status }}</a-tag>
        </a-descriptions-item>
        <a-descriptions-item :label="t('taskDetail.project')" v-if="currentTask.project_id">
          {{ currentTask.project_name || currentTask.project_id }}
        </a-descriptions-item>
        <a-descriptions-item :label="t('taskDetail.gpuCount')">{{ currentTask.gpu_count }}</a-descriptions-item>
        <a-descriptions-item :label="t('taskDetail.priority')">{{ currentTask.priority }}</a-descriptions-item>
        <a-descriptions-item :label="t('taskDetail.gpuHours')">{{ (currentTask.gpu_hours || 0).toFixed(2) }}</a-descriptions-item>
        <a-descriptions-item :label="t('taskDetail.cost')">{{ (currentTask.cost || 0).toFixed(2) }}</a-descriptions-item>
        <a-descriptions-item :label="t('taskDetail.createdAt')">{{ currentTask.created_at }}</a-descriptions-item>
      </a-descriptions>

      <a-card :title="t('taskDetail.logs')" style="margin-top: 16px">
        <div
          id="task-log-container"
          style="height: 300px; overflow-y: auto; background: var(--color-fill-2); border-radius: 4px; padding: 8px;"
        >
          <div v-if="taskLogs.length === 0" class="arco-typography color-text-3">{{ t('taskDetail.noLogs') }}</div>
          <div v-for="(log, idx) in taskLogs" :key="idx" style="font-family: monospace; line-height: 1.5;">
            {{ log }}
          </div>
        </div>
      </a-card>

      <div style="margin-top: 16px; text-align: right;">
        <a-space>
          <a-button v-if="currentTask && ['canceled', 'failed'].includes(currentTask.status)" @click="resumeTask(currentTask.id)">
            {{ t('taskDetail.resumeTask') }}
          </a-button>
          <a-button type="primary" @click="downloadResult">
            {{ t('taskDetail.downloadResult') }}
          </a-button>
        </a-space>
      </div>
    </a-drawer>
  </a-card>
</template>

<style scoped>
#task-log-container {
  /* 暗色模式下自动适配背景色 */
}
</style>
