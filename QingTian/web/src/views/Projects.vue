<script setup>
// 项目管理页面：列出我参与的项目，新建项目，添加成员。
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { Message, Modal } from '@arco-design/web-vue'
import { projectApi } from '../api'

const { t } = useI18n()

const loading = ref(false)
const projects = ref([])
const createModalVisible = ref(false)
const createForm = ref({ name: '' })

// 加载项目列表
async function loadProjects() {
  loading.value = true
  try {
    projects.value = await projectApi.list()
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
  Modal.confirm({
    title: t('project.addMember'),
    content: `
      <div>
        <label style="display:block; margin-bottom:8px;">${t('project.addMemberPlaceholder')}</label>
        <input id="add-member-username" type="text" class="arco-input">
      </div>
    `,
    onOk: async () => {
      const username = document.getElementById('add-member-username')?.value?.trim()
      if (!username) {
        Message.error(t('project.addMemberPlaceholder') + ' ' + t('common.required'))
        return Promise.reject()
      }
      await projectApi.addMember(project.id, username)
      Message.success(t('project.add') + ' ' + t('common.success'))
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
      <a-table :data="projects" :loading="loading" row-key="id" :pagination="{ pageSize: 10 }">
        <a-table-column :title="t('project.projectName')" data-index="name" :width="200" />
        <a-table-column :title="t('project.owner')" data-index="ownerUsername" :width="150" />
        <a-table-column :title="t('project.members')" data-index="memberCount" :width="100">
          <template #cell="{ record }">
            {{ (record.members || []).length }}
          </template>
        </a-table-column>
        <a-table-column :title="t('common.actions')" data-index="id">
          <template #cell="{ record }">
            <a-button size="small" @click="openAddMember(record)">{{ t('project.addMember') }}</a-button>
          </template>
        </a-table-column>
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
  </a-spin>
</template>

<style scoped>
</style>
