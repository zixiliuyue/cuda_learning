<script setup>
import { computed, watch } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

const props = defineProps({
  modelValue: {
    type: Object,
    default: () => ({}),
  },
})

const emit = defineEmits(['update:modelValue'])

// 监听选中节点变化
watch(
  () => props.modelValue,
  () => {
    // 重置表单状态 - 这里直接使用modelValue响应式
  },
  { deep: true },
)

// 是否显示对应表单
const showForm = computed(() => {
  const type = props.modelValue.executorType
  return {
    container: type === 'container',
    shell: type === 'shell',
    python: type === 'python',
    api: type === 'api',
  }
})

// 环境变量列表
const envItems = computed(() => {
  const env = props.modelValue.shellConfig?.env || props.modelValue.pythonConfig?.env || props.modelValue.apiConfig?.headers || {}
  return Object.entries(env).map(([key, value]) => ({ key, value }))
})

// 添加环境变量
function addEnv() {
  const type = props.modelValue.executorType
  const config = getConfig()
  if (!config.env) {
    config.env = {}
  }
  config.env[''] = ''
  emitChange()
}

// 删除环境变量
function deleteEnv(key) {
  const type = props.modelValue.executorType
  const config = getConfig()
  delete config.env[key]
  emitChange()
}

// 获取对应配置对象
function getConfig() {
  const type = props.modelValue.executorType
  switch (type) {
    case 'container':
      if (!props.modelValue.containerConfig) {
        props.modelValue.containerConfig = {
          image: '',
          command: [],
          args: [],
          env: {},
        }
      }
      return props.modelValue.containerConfig
    case 'shell':
      if (!props.modelValue.shellConfig) {
        props.modelValue.shellConfig = {
          script: '',
          env: {},
        }
      }
      return props.modelValue.shellConfig
    case 'python':
      if (!props.modelValue.pythonConfig) {
        props.modelValue.pythonConfig = {
          script: '',
          requirements: [],
          env: {},
        }
      }
      return props.modelValue.pythonConfig
    case 'api':
      if (!props.modelValue.apiConfig) {
        props.modelValue.apiConfig = {
          url: '',
          method: 'GET',
          headers: {},
          body: '',
          timeout: 30,
          successCodes: [200],
        }
      }
      return props.modelValue.apiConfig
    default:
      return {}
  }
}

function emitChange() {
  emit('update:modelValue', { ...props.modelValue })
}

// 处理需求列表文本换行转换为数组
function getRequirements() {
  const cfg = getConfig()
  if (!cfg.requirements) {
    return ''
  }
  return cfg.requirements.join('\n')
}

function setRequirements(value) {
  const cfg = getConfig()
  cfg.requirements = value.split('\n').map(line => line.trim()).filter(line => line)
  emitChange()
}
</script>

<template>
  <div class="config-panel">
    <a-form :model="props.modelValue" layout="vertical">
      <a-form-item :label="t('workflowNode.id')">
        <a-input v-model="props.modelValue.id" disabled />
      </a-form-item>
      <a-form-item :label="t('workflowNode.name')">
        <a-input v-model="props.modelValue.name" :placeholder="t('workflowNode.name')" />
      </a-form-item>
      <a-form-item :label="t('workflowNode.description')">
        <a-textarea v-model="props.modelValue.description" :placeholder="t('workflowNode.description')" />
      </a-form-item>
      <a-form-item :label="t('workflowNode.executorType')">
        <a-select v-model="props.modelValue.executorType" :placeholder="t('workflowNode.executorType')">
          <a-option value="container">{{ t('workflowNode.container') }}</a-option>
          <a-option value="shell">{{ t('workflowNode.shell') }}</a-option>
          <a-option value="python">{{ t('workflowNode.python') }}</a-option>
          <a-option value="api">{{ t('workflowNode.api') }}</a-option>
        </a-select>
      </a-form-item>
    </a-form>

    <!-- 容器配置 -->
    <template v-if="showForm.container">
      <a-form-item :label="t('workflowNode.image')">
        <a-input v-model="props.modelValue.containerConfig.image" placeholder="nginx:latest" />
      </a-form-item>
      <a-form-item :label="t('workflowNode.command')">
        <a-input v-model="props.modelValue.containerConfig.command" placeholder='[""]' />
      </a-form-item>
      <a-form-item :label="t('workflowNode.args')">
        <a-input v-model="props.modelValue.containerConfig.args" placeholder='[""]' />
      </a-form-item>
    </template>

    <!-- Shell 配置 -->
    <template v-if="showForm.shell">
      <a-form-item :label="t('workflowNode.script')">
        <a-textarea
          v-model="props.modelValue.shellConfig.script"
          :rows="12"
          :placeholder="t('workflowNode.script')"
        />
      </a-form-item>
      <a-form-item :label="t('workflowNode.environment')">
        <div class="env-list">
          <div v-for="(item, index) in envItems" :key="index" class="env-item">
            <a-space>
              <a-input v-model="item.key" :placeholder="t('workflowNode.envKey')" style="width: 120px" />
              <a-input v-model="item.value" :placeholder="t('workflowNode.envValue')" />
              <a-button type="text" status="danger" @click="deleteEnv(item.key)">
                <icon-delete />
              </a-button>
            </a-space>
          </div>
          <a-button type="primary" @click="addEnv">
            <template #icon><icon-plus />
            </template>
            {{ t('workflowNode.addEnv') }}
          </a-button>
        </div>
      </a-form-item>
    </template>

    <!-- Python 配置 -->
    <template v-if="showForm.python">
      <a-form-item :label="t('workflowNode.script')">
        <a-textarea
          v-model="props.modelValue.pythonConfig.script"
          :rows="12"
          :placeholder="t('workflowNode.script')"
        />
      </a-form-item>
      <a-form-item :label="t('workflowNode.requirements')">
        <a-textarea
          :model-value="getRequirements()"
          @update:model-value="setRequirements"
          :rows="4"
          :placeholder="t('workflowNode.requirements')"
        />
      </a-form-item>
      <a-form-item :label="t('workflowNode.environment')">
        <div class="env-list">
          <div v-for="(item, index) in envItems" :key="index" class="env-item">
            <a-space>
              <a-input v-model="item.key" :placeholder="t('workflowNode.envKey')" style="width: 120px" />
              <a-input v-model="item.value" :placeholder="t('workflowNode.envValue')" />
              <a-button type="text" status="danger" @click="deleteEnv(item.key)">
                <icon-delete />
              </a-button>
            </a-space>
          </div>
          <a-button type="primary" @click="addEnv">
            <template #icon><icon-plus />
            </template>
            {{ t('workflowNode.addEnv') }}
          </a-button>
        </div>
      </a-form-item>
    </template>

    <!-- API 配置 -->
    <template v-if="showForm.api">
      <a-form-item :label="t('workflowNode.url')">
        <a-input v-model="props.modelValue.apiConfig.url" placeholder="https://example.com/api" />
      </a-form-item>
      <a-form-item :label="t('workflowNode.method')">
        <a-select v-model="props.modelValue.apiConfig.method" placeholder="GET">
          <a-option value="GET">GET</a-option>
          <a-option value="POST">POST</a-option>
          <a-option value="PUT">PUT</a-option>
          <a-option value="DELETE">DELETE</a-option>
        </a-select>
      </a-form-item>
      <a-form-item :label="t('workflowNode.headers')">
        <div class="env-list">
          <div v-for="(item, index) in envItems" :key="index" class="env-item">
            <a-space>
              <a-input v-model="item.key" :placeholder="t('workflowNode.envKey')" style="width: 120px" />
              <a-input v-model="item.value" :placeholder="t('workflowNode.envValue')" />
              <a-button type="text" status="danger" @click="deleteEnv(item.key)">
                <icon-delete />
              </a-button>
            </a-space>
          </div>
          <a-button type="primary" @click="addEnv">
            <template #icon><icon-plus />
            </template>
            {{ t('workflowNode.addEnv') }}
          </a-button>
        </div>
      </a-form-item>
      <a-form-item :label="t('workflowNode.body')">
        <a-textarea
          v-model="props.modelValue.apiConfig.body"
          :rows="6"
          :placeholder="t('workflowNode.body')"
        />
      </a-form-item>
      <a-form-item :label="t('workflowNode.timeout')">
        <a-input-number v-model="props.modelValue.apiConfig.timeout" :min="1" :step="10" placeholder="30">
          <template #extra>
            {{ t('workflowNode.timeout') }} (seconds)
          </template>
        </a-input-number>
      </a-form-item>
      <a-form-item :label="t('workflowNode.successCodes')">
        <a-input v-model="props.modelValue.apiConfig.successCodes" placeholder="200 201">
          <template #extra>
            {{ t('workflowNode.successCodes') }} (space separated)
          </template>
        </a-input>
      </a-form-item>
    </template>

    <!-- 公共资源配置 -->
    <a-divider />
    <a-form-item :label="t('workflowNode.gpuCount')">
      <a-input-number v-model="props.modelValue.resources.gpuCount" :min="0" :step="1" />
    </a-form-item>
    <a-form-item :label="t('workflowNode.gpuMemory')">
      <a-input v-model="props.modelValue.resources.gpuMemory" placeholder="16GB" />
    </a-form-item>
    <a-form-item :label="t('workflowNode.gpuModel')">
      <a-input v-model="props.modelValue.resources.gpuModel" placeholder="RTX 3090" />
    </a-form-item>
    <a-form-item :label="t('workflowNode.timeout')">
      <a-input-number v-model="props.modelValue.timeout" :min="0" :step="300" />
      <template #extra>
        0 means unlimited
      </template>
    </a-form-item>
    <a-form-item :label="t('workflowNode.retries')">
      <a-input-number v-model="props.modelValue.retries" :min="0" :step="1" />
    </a-form-item>
  </div>
</template>

<style scoped>
.config-panel {
  padding: 16px;
  height: 100%;
  overflow-y: auto;
  background: var(--color-bg-2);
  border-left: 1px solid var(--color-border);
}

.env-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.env-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 0;
  border-bottom: 1px dashed var(--color-border);
}
</style>
