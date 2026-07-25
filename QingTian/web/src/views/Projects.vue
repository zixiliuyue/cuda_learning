<script setup>
// 项目管理页面：列出我参与的项目，新建项目，添加成员。
import { ref, computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { Message, Modal } from '@arco-design/web-vue'
import { projectApi } from '../api'

const { t } = useI18n()

const loading = ref(false)
const projects = ref([])
const createModalVisible = ref(false)
const createForm = ref({ name: '' })
// 添加成员弹窗状态（改用组件化弹窗，避免 Modal.confirm 传 HTML 字符串导致标签被当纯文本显示）。
const memberModalVisible = ref(false)
const memberForm = ref({ projectId: '', username: '' })
// 编辑（重命名）弹窗状态。
const editModalVisible = ref(false)
const editForm = ref({ projectId: '', name: '' })

// 列定义使用 computed 而非 <a-table-column> 子组件，避免 Arco 组件注册问题。
const columns = computed(() => [
  { title: t('project.projectName'), dataIndex: 'name', width: 200 },
  { title: t('project.owner'), dataIndex: 'owner_id', width: 220 },
  { title: t('project.members'), dataIndex: 'members', width: 100, slotName: 'members' },
  { title: t('common.actions'), dataIndex: 'id', slotName: 'actions', width: 240 },
])

// 加载项目列表
async function loadProjects() {
  loading.value = true
  try {
    projects.value = (await projectApi.list()) || []
  } catch (e) {
    // 错误已拦截
  } finally {
    loading.value = false
  }
}

// 打开新建项目弹窗
function openCreateModal() {
  createForm.value = { name: '' }
  createModalVisible.value = true
}

// 提交创建
async function handleCreate() {
  if (!createForm.value.name.trim()) {
    Message.warning(t('project.projectName') + ' ' + t('common.required'))
    return
  }
  try {
    await projectApi.create(createForm.value.name)
    Message.success(t('project.create') + ' ' + t('common.success'))
    createModalVisible.value = false
    loadProjects()
  } catch (e) {
    // 错误已拦截
  }
}

// 打开添加成员弹窗
function openAddMember(project) {
  memberForm.value = { projectId: project.id, username: '' }
  memberModalVisible.value = true
}

// 提交添加成员
async function handleAddMember() {
  const username = memberForm.value.username.trim()
  if (!username) {
    Message.error(t('project.addMemberPlaceholder') + ' ' + t('common.required'))
    return
  }
  try {
    await projectApi.addMember(memberForm.value.projectId, username)
    Message.success(t('project.add') + ' ' + t('common.success'))
    memberModalVisible.value = false
    loadProjects()
  } catch (e) {
    // 错误已拦截
  }
}

// 打开编辑（重命名）弹窗
function openEdit(project) {
  editForm.value = { projectId: project.id, name: project.name }
  editModalVisible.value = true
}

// 提交重命名
async function handleEdit() {
  const name = editForm.value.name.trim()
  if (!name) {
    Message.error(t('project.projectName') + ' ' + t('common.required'))
    return
  }
  try {
    await projectApi.rename(editForm.value.projectId, name)
    Message.success(t('common.success'))
    editModalVisible.value = false
    loadProjects()
  } catch (e) {
    // 错误已拦截
  }
}

// 删除项目（二次确认）
function handleDelete(project) {
  Modal.confirm({
    title: t('common.delete'),
    content: `${t('common.delete')} "${project.name}"?`,
    onOk: async () => {
      await projectApi.delete(project.id)
      Message.success(t('common.success'))
      loadProjects()
    },
  })
}

onMounted(loadProjects)
</script>

<template>
  <a-spin :loading="loading" style="display: block">
    <a-card :title="t('project.title')">
      <template #extra>
        <a-button type="primary" @click="openCreateModal">{{ t('project.createBtn') }}</a-button>
      </template>
      <a-table :data="projects" :columns="columns" :loading="loading" row-key="id" :pagination="{ pageSize: 10 }">
        <template #members="{ record }">
          {{ (record.members || []).length }}
        </template>
        <template #actions="{ record }">
          <a-space>
            <a-button size="small" @click="openAddMember(record)">{{ t('project.addMember') }}</a-button>
            <a-button size="small" @click="openEdit(record)">{{ t('common.edit') }}</a-button>
            <a-button size="small" status="danger" @click="() => handleDelete(record)">{{ t('common.delete') }}</a-button>
          </a-space>
        </template>
      </a-table>
    </a-card>

    <a-modal
      v-model:visible="createModalVisible"
      :title="t('project.createTitle')"
      @ok="handleCreate"
    >
      <a-form :model="createForm" layout="vertical">
        <a-form-item field="name" :label="t('project.projectName')">
          <a-input
            v-model="createForm.name"
            :placeholder="t('project.namePlaceholder')"
          />
        </a-form-item>
      </a-form>
    </a-modal>

    <a-modal
      v-model:visible="memberModalVisible"
      :title="t('project.addMember')"
      @ok="handleAddMember"
    >
      <a-form :model="memberForm" layout="vertical">
        <a-form-item field="username" :label="t('project.addMemberPlaceholder')">
          <a-input
            v-model="memberForm.username"
            :placeholder="t('project.addMemberPlaceholder')"
          />
        </a-form-item>
      </a-form>
    </a-modal>

    <a-modal
      v-model:visible="editModalVisible"
      :title="t('common.edit')"
      @ok="handleEdit"
    >
      <a-form :model="editForm" layout="vertical">
        <a-form-item field="name" :label="t('project.projectName')">
          <a-input
            v-model="editForm.name"
            :placeholder="t('project.namePlaceholder')"
          />
        </a-form-item>
      </a-form>
    </a-modal>
  </a-spin>
</template>

<style scoped>
</style>
