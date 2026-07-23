<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { Message } from '@arco-design/web-vue'
import { useRoute, useRouter } from 'vue-router'
import { VueFlow, useVueFlow } from '@vue-flow/core'
import IconArrowLeft from '@arco-design/web-vue/es/icon/icon-arrow-left'
import IconRefresh from '@arco-design/web-vue/es/icon/icon-refresh'
import IconPlus from '@arco-design/web-vue/es/icon/icon-plus'
import IconSave from '@arco-design/web-vue/es/icon/icon-save'
import TaskNodePanel from '../components/workflow/TaskNodePanel.vue'
import TaskConfigPanel from '../components/workflow/TaskConfigPanel.vue'
import { workflowApi } from '../api'

const { t } = useI18n()
const route = useRoute()
const router = useRouter()

const workflowId = route.params.id
const isNew = workflowId === 'new' || workflowId === 'new'

// VueFlow core
const {
  getNodes,
  getEdges,
  addNode,
  addEdge,
  onNodesChange,
  onEdgesChange,
  onConnect,
  setNodes,
  setEdges,
} = useVueFlow()

// 组件状态
const loading = ref(false)
const saving = ref(false)
const selectedNodeId = ref(null)
const reactFlowWrapper = ref(null)

// 工作流基本信息
const form = reactive({
  name: '',
  description: '',
  version: '1.0',
})

// 获取选中的节点
const selectedNode = computed(() => {
  const nodes = getNodes.value
  return nodes.find(n => n.id === selectedNodeId.value)
})

// 画布尺寸
const reactFlowStyle = computed(() => ({
  height: 'calc(100vh - 140px)',
  width: '100%',
}))

// 从后端加载工作流数据
async function loadWorkflow() {
  if (isNew) return
  loading.value = true
  try {
    const wf = await workflowApi.get(workflowId)
    form.name = wf.name
    form.description = wf.description
    form.version = wf.version || '1.0'
    setNodes.value = wf.tasks.map(t => ({
      id: t.id,
      type: 'default',
      position: t.position || { x: Math.random() * 500, y: Math.random() * 300 },
      data: {
        ...t,
        label: t.name,
      },
    }))
    setEdges.value = (wf.edges || []).map(e => ({
      id: `${e.from}-${e.to}`,
      source: e.from,
      target: e.to,
    }))
  } catch (e) {
    Message.error(e.message || t('common.loadFailed'))
  } finally {
    loading.value = false
  }
}

// 拖放新节点到画布
function onDrop(event) {
  event.preventDefault()
  const type = event.dataTransfer.getData('application/reactflow')
  if (!type) return

  // 获取鼠标位置
  const rect = reactFlowWrapper.value.getBoundingClientRect()
  const x = event.clientX - rect.left - 75
  const y = event.clientY - rect.top - 25

  // 创建新节点
  const newNode = {
    id: `node_${Date.now()}`,
    type: 'default',
    position: { x, y },
    data: {
      id: `node_${Date.now()}`,
      name: `${type}_node`,
      executorType: type,
      label: `${type}_node`,
    },
  }

  addNode(newNode)
  selectedNodeId.value = newNode.id
}

// 允许拖放
function onDragOver(event) {
  event.preventDefault()
}

// 节点点击
function onNodeClick(node) {
  selectedNodeId.value = node.id
}

// 画布空白点击
function onPaneClick() {
  selectedNodeId.value = null
}

// 保存工作流
async function handleSave() {
  if (!form.name) {
    Message.warning(t('workflow.name') + ' ' + t('common.required'))
    return
  }

  const nodes = getNodes.value
  const edges = getEdges.value

  // 提取任务和边数据
  const tasks = nodes.map(n => ({
    id: n.data.id,
    name: n.data.name,
    description: n.data.description,
    executorType: n.data.executorType,
    containerConfig: n.data.containerConfig,
    shellConfig: n.data.shellConfig,
    pythonConfig: n.data.pythonConfig,
    apiConfig: n.data.apiConfig,
    resources: n.data.resources || { gpuCount: 1 },
    timeout: n.data.timeout || 0,
    retries: n.data.retries || 0,
    retryDelay: n.data.retryDelay || 0,
    inputs: n.data.inputs || [],
    outputs: n.data.outputs || [],
  }))

  const workflowEdges = edges.map(e => ({
    from: e.source,
    to: e.target,
  }))

  const data = {
    name: form.name,
    description: form.description,
    version: form.version,
    tasks,
    edges: workflowEdges,
  }

  saving.value = true
  try {
    if (isNew) {
      await workflowApi.create(data)
      Message.success(t('common.success'))
      router.push('/workflows')
    } else {
      await workflowApi.update(workflowId, data)
      Message.success(t('common.success'))
    }
  } catch (e) {
    Message.error(e.message || t('common.saveFailed'))
  } finally {
    saving.value = false
  }
}

onMounted(() => {
  loadWorkflow()
})
</script>

<template>
  <a-card :title="isNew ? t('workflow.createTitle') : form.name || t('workflow.editTitle')" class="workflow-editor-card">
    <template #extra>
      <a-space>
        <a-button @click="router.push('/workflows')">
          <template #icon><icon-arrow-left />
          </template>
          {{ t('common.back') }}
        </a-button>
        <a-button type="primary" :loading="saving" @click="handleSave">
          <template #icon><icon-save />
          </template>
          {{ t('common.save') }}
        </a-button>
      </a-space>
    </template>

    <a-row :gutter="12" class="editor-row">
      <a-col :span="5">
        <task-node-panel />
      </a-col>
      <a-col :span="13">
        <div class="react-flow-wrapper" ref="reactFlowWrapper" @drop="onDrop" @dragover="onDragOver">
          <vue-flow
            :nodes="getNodes"
            :edges="getEdges"
            @nodes-change="onNodesChange"
            @edges-change="onEdgesChange"
            @connect="onConnect"
            @node-click="onNodeClick"
            @pane-click="onPaneClick"
            :style="reactFlowStyle"
          />
        </div>
      </a-col>
      <a-col :span="6" v-if="selectedNode">
        <task-config-panel
          v-model="selectedNode.data"
          @update:model-value="setNodes"
        />
      </a-col>
    </a-row>
  </a-card>
</template>

<style scoped>
.workflow-editor-card {
  margin: 0;
}

.editor-row {
  margin-top: 12px;
  height: calc(100vh - 200px);
}

.react-flow-wrapper {
  width: 100%;
  height: 100%;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  background: var(--color-fill-2);
}
</style>
