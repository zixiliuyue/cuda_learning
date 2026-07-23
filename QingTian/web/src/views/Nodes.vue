<script setup>
// 节点管理页：展示所有 GPU 节点及其 GPU 明细、在线状态。
// 新增：支持一键通过 SSH 自动安装 Agent 到远程节点。
import { ref, computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { Message } from '@arco-design/web-vue'
import { nodeApi } from '../api'
import installApi from '../api/install.js'

const { t } = useI18n()
const nodes = ref([])
const loading = ref(false)

// 一键安装模态框
const installModalVisible = ref(false)
const installStep = ref(1) // 1: 输入信息, 2: 查看脚本, 3: 执行中, 4: 完成
const installForm = ref({
  host: '',
  port: 22,
  user: '',
  authType: 'password', // password | privateKey
  password: '',
  privateKey: '',
  passphrase: '',
})
const scriptContent = ref('')
const installLog = ref('')

// 自动获取控制平面地址
function getDefaultControlPlaneURL() {
  return `${window.location.protocol}//${window.location.host}/api/v1`
}

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

// 打开一键安装向导
function openInstallModal() {
  installStep.value = 1
  installForm.value = {
    host: '',
    port: 22,
    user: '',
    authType: 'password',
    password: '',
    privateKey: '',
    passphrase: '',
  }
  scriptContent.value = ''
  installLog.value = ''
  installModalVisible.value = true
}

// 测试SSH连接
async function testConnection() {
  const { host, port, user, authType, password, privateKey, passphrase } = installForm.value
  if (!host.trim()) {
    Message.warning(t('nodes.install.hostRequired'))
    return
  }
  if (!user.trim()) {
    Message.warning(t('nodes.install.userRequired'))
    return
  }
  if (authType === 'password' && !password.trim()) {
    Message.warning(t('nodes.install.passwordRequired'))
    return
  }

  try {
    await installApi.testSSH({
      host,
      port,
      user,
      password,
      privateKey,
      passphrase,
    })
    Message.success(t('nodes.install.connectionSuccess'))
    installStep.value = 2
  } catch (e) {
    Message.error(t('nodes.install.connectionFailed'))
  }
}

// 生成安装脚本并开始安装
async function generateScript() {
  const controlPlaneURL = getDefaultControlPlaneURL()
  try {
    const res = await installApi.installRun({
      ...installForm.value,
      control_plane_url: controlPlaneURL,
    })
    scriptContent.value = res.script
    installLog.value = t('nodes.install.scriptGenerated') + '\n\n' + res.script
    installStep.value = 3
    Message.success(t('nodes.install.scriptGenerated'))
  } catch (e) {
    Message.error(t('nodes.install.generateFailed'))
  }
}

// 完成安装，刷新节点列表
function finishInstall() {
  installModalVisible.value = false
  loadNodes()
}

onMounted(loadNodes)
</script>

<template>
  <a-card :title="t('nodes.title')">
    <template #extra>
      <a-space>
        <a-button type="primary" @click="openInstallModal">{{ t('nodes.install.addNode') }}</a-button>
        <a-button @click="loadNodes" :loading="loading">{{ t('common.refresh') }}</a-button>
      </a-space>
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

  <!-- 一键安装 Agent 向导 -->
  <a-modal
    v-model:visible="installModalVisible"
    :title="t('nodes.install.title')"
    @ok="installStep === 1 ? testConnection() : installStep === 2 ? generateScript() : finishInstall()"
    :ok-text="installStep === 4 ? t('nodes.install.finish') : t('nodes.install.next')"
    :cancel-text="t('common.cancel')"
  >
    <!-- 步骤 1: 输入 SSH 连接信息 -->
    <div v-if="installStep === 1">
      <a-form :model="installForm" layout="vertical">
        <a-form-item :label="t('nodes.install.host')" required>
          <a-input v-model="installForm.host" :placeholder="t('nodes.install.hostPlaceholder')" />
        </a-form-item>
        <a-form-item :label="t('nodes.install.port')">
          <a-input-number v-model="installForm.port" :min="1" :max="65535" />
        </a-form-item>
        <a-form-item :label="t('nodes.install.user')" required>
          <a-input v-model="installForm.user" :placeholder="t('nodes.install.userPlaceholder')" />
        </a-form-item>
        <a-form-item :label="t('nodes.install.authType')">
          <a-radio-group v-model="installForm.authType">
            <a-radio value="password">{{ t('nodes.install.authPassword') }}</a-radio>
            <a-radio value="privateKey">{{ t('nodes.install.authPrivateKey') }}</a-radio>
          </a-radio-group>
        </a-form-item>
        <a-form-item v-if="installForm.authType === 'password'" :label="t('nodes.install.password')" required>
          <a-input-password v-model="installForm.password" />
        </a-form-item>
        <a-form-item v-if="installForm.authType === 'privateKey'" :label="t('nodes.install.privateKey')" required>
          <a-textarea v-model="installForm.privateKey" :rows="8" :placeholder="t('nodes.install.privateKeyPlaceholder')" />
        </a-form-item>
        <a-form-item v-if="installForm.authType === 'privateKey'" :label="t('nodes.install.passphrase')">
          <a-input-password v-model="installForm.passphrase" :placeholder="t('nodes.install.passphrasePlaceholder')" />
        </a-form-item>
      </a-form>
      <div class="mt-4">
        <a-alert type="info" :content="t('nodes.install.help')" />
      </div>
    </div>

    <!-- 步骤 2: 确认并生成脚本 -->
    <div v-if="installStep === 2">
      <p>{{ t('nodes.install.confirm') }}</p>
      <ul>
        <li>{{ t('nodes.install.target') }}: {{ installForm.user }}@{{ installForm.host }}:{{ installForm.port }}</li>
        <li>{{ t('nodes.install.controlPlane') }}: {{ getDefaultControlPlaneURL() }}</li>
      </ul>
      <div class="mt-4">
        <a-alert type="success" :content="t('nodes.install.confirmHelp')" />
      </div>
    </div>

    <!-- 步骤 3: 查看安装脚本 -->
    <div v-if="installStep === 3">
      <p>{{ t('nodes.install.scriptTitle') }}</p>
      <a-textarea
        v-model="scriptContent"
        :rows="25"
        readonly
        style="font-family: monospace; font-size: 12px;"
      />
      <div class="mt-4">
        <a-alert type="warning" :content="t('nodes.install.scriptHelp')" />
      </div>
    </div>

    <!-- 步骤 4: 完成 -->
    <div v-if="installStep === 4">
      <a-alert type="success" :content="t('nodes.install.completeHelp')" />
    </div>
  </a-modal>
</template>

<style scoped>
.mt-4 {
  margin-top: 16px;
}
</style>
