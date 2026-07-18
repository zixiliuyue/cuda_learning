<script setup>
// 任务管理页：展示任务列表、支持提交新任务与取消任务。
import { ref, onMounted, reactive } from 'vue'
import { Message } from '@arco-design/web-vue'
import { taskApi } from '../api'

const tasks = ref([])
const loading = ref(false)
const modalVisible = ref(false)

// 新任务表单。
const form = reactive({
  name: '',
  image: '',
  command: '',
  priority: 0,
  gpu_count: 1,
  memory_mb: 8192,
  gpu_model: ''
})

const columns = [
  { title: '任务 ID', dataIndex: 'id', ellipsis: true, tooltip: true, width: 200 },
  { title: '名称', dataIndex: 'name' },
  { title: '镜像', dataIndex: 'image' },
  { title: '状态', slotName: 'status' },
  { title: '节点', dataIndex: 'node_id' },
  { title: '操作', slotName: 'actions' }
]

// 状态到标签颜色的映射。
const statusColor = {
  pending: 'gray',
  scheduled: 'blue',
  running: 'arcoblue',
  succeeded: 'green',
  failed: 'red',
  canceled: 'orange'
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
  Object.assign(form, { name: '', image: '', command: '', priority: 0, gpu_count: 1, memory_mb: 8192, gpu_model: '' })
  modalVisible.value = true
}

// 提交任务。
async function submitTask() {
  if (!form.name || !form.image) {
    Message.warning('请填写任务名称与镜像')
    return
  }
  try {
    await taskApi.submit({
      name: form.name,
      image: form.image,
      command: form.command,
      priority: form.priority,
      resource: {
        gpu_count: form.gpu_count,
        memory_mb: form.memory_mb,
        gpu_model: form.gpu_model
      }
    })
    Message.success('任务提交成功')
    modalVisible.value = false
    loadTasks()
  } catch {
    // 错误已由拦截器提示。
  }
}

// 取消任务。
async function cancelTask(id) {
  try {
    await taskApi.cancel(id)
    Message.success('已取消')
    loadTasks()
  } catch {
    // 已提示。
  }
}

onMounted(loadTasks)
</script>

<template>
  <a-card title="算力任务">
    <template #extra>
      <a-space>
        <a-button @click="loadTasks" :loading="loading">刷新</a-button>
        <a-button type="primary" @click="openModal">提交任务</a-button>
      </a-space>
    </template>

    <a-table :data="tasks" :columns="columns" :loading="loading" row-key="id" :pagination="{ pageSize: 10 }">
      <template #status="{ record }">
        <a-tag :color="statusColor[record.status] || 'gray'">{{ record.status }}</a-tag>
      </template>
      <template #actions="{ record }">
        <a-button
          size="mini"
          status="danger"
          :disabled="['succeeded', 'failed', 'canceled'].includes(record.status)"
          @click="cancelTask(record.id)"
        >取消</a-button>
      </template>
    </a-table>

    <a-modal v-model:visible="modalVisible" title="提交算力任务" @ok="submitTask" ok-text="提交" cancel-text="取消">
      <a-form :model="form" layout="vertical">
        <a-form-item label="任务名称" required>
          <a-input v-model="form.name" placeholder="例如 resnet-train" />
        </a-form-item>
        <a-form-item label="容器镜像" required>
          <a-input v-model="form.image" placeholder="例如 nvidia/cuda:13.1-runtime" />
        </a-form-item>
        <a-form-item label="执行命令">
          <a-input v-model="form.command" placeholder="例如 python train.py" />
        </a-form-item>
        <a-grid :cols="2" :col-gap="16">
          <a-grid-item>
            <a-form-item label="GPU 数量">
              <a-input-number v-model="form.gpu_count" :min="1" />
            </a-form-item>
          </a-grid-item>
          <a-grid-item>
            <a-form-item label="单卡显存 (MB)">
              <a-input-number v-model="form.memory_mb" :min="512" :step="512" />
            </a-form-item>
          </a-grid-item>
        </a-grid>
        <a-grid :cols="2" :col-gap="16">
          <a-grid-item>
            <a-form-item label="优先级">
              <a-input-number v-model="form.priority" :min="0" />
            </a-form-item>
          </a-grid-item>
          <a-grid-item>
            <a-form-item label="GPU 型号 (可选)">
              <a-input v-model="form.gpu_model" placeholder="留空不限" />
            </a-form-item>
          </a-grid-item>
        </a-grid>
      </a-form>
    </a-modal>
  </a-card>
</template>
