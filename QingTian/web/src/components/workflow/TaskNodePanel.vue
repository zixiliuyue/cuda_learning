<script setup>
import { ref, computed } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

// 搜索关键词
const searchQuery = ref('')

// 可拖拽的节点类型列表，按分类组织
const nodeCategories = [
  {
    key: 'execution',
    label: t('workflowNode.category.execution'),
    expanded: true,
    nodes: [
      {
        type: 'container',
        label: t('workflowNode.container'),
        icon: 'icon-container',
        description: t('workflowNode.containerDesc'),
      },
      {
        type: 'shell',
        label: t('workflowNode.shell'),
        icon: 'icon-terminal',
        description: t('workflowNode.shellDesc'),
      },
      {
        type: 'python',
        label: t('workflowNode.python'),
        icon: 'icon-code',
        description: t('workflowNode.pythonDesc'),
      },
      {
        type: 'api',
        label: t('workflowNode.api'),
        icon: 'icon-api',
        description: t('workflowNode.apiDesc'),
      },
    ],
  },
]

// 过滤后的节点列表
const filteredNodeTypes = computed(() => {
  const q = searchQuery.value.toLowerCase().trim()
  if (!q) {
    return nodeCategories
  }
  // 对每个分类过滤节点
  return nodeCategories.map(cat => ({
    ...cat,
    nodes: cat.nodes.filter(n =>
      n.label.toLowerCase().includes(q) ||
        (n.description && n.description.toLowerCase().includes(q))
    ),
  })).filter(cat => cat.nodes.length > 0)
})

// 拖拽开始事件
function onDragStart(event, nodeType) {
  event.dataTransfer.setData('application/reactflow', nodeType)
  event.dataTransfer.effectAllowed = 'move'
}

// 切换分类展开/折叠
function toggleCategory(category) {
  category.expanded = !category.expanded
}
</script>

<template>
  <div class="node-panel">
    <div class="panel-search">
      <a-input
        v-model="searchQuery"
        :placeholder="t('workflowNode.searchPlaceholder')"
        clearable
        class="search-input"
      />
    </div>
    <div class="panel-content">
      <div
        v-for="cat in filteredNodeTypes"
        :key="cat.key"
        class="category-block"
      >
        <div class="category-title" @click="toggleCategory(cat)">
          <span>{{ cat.label }}</span>
          <a-icon :name="cat.expanded ? 'icon-arrow-down' : 'icon-arrow-right'" class="toggle-icon" />
        </div>
        <div v-if="cat.expanded" class="node-list">
          <div
            v-for="item in cat.nodes"
            :key="item.type"
            class="node-item"
            draggable
            @dragstart="(e) => onDragStart(e, item.type)"
          >
            <a-icon :name="item.icon" />
            <div class="node-info">
              <span>{{ item.label }}</span>
              <p v-if="item.description" class="node-desc">{{ item.description }}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.node-panel {
  padding: 12px;
  height: 100%;
  overflow-y: auto;
  background: var(--color-bg-2);
  border-right: 1px solid var(--color-border);
}

.panel-search {
  margin-bottom: 12px;
}

.search-input {
  width: 100%;
}

.category-block {
  margin-bottom: 8px;
}

.category-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 8px;
  background: var(--color-fill-3);
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
  color: var(--color-text-1);
}

.toggle-icon {
  font-size: 12px;
  color: var(--color-text-3);
}

.panel-content {
  padding-top: 8px;
}

.node-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding-left: 8px;
}

.node-item {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 8px 12px;
  background: var(--color-fill-2);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  cursor: grab;
  color: var(--color-text-2);
  transition: all 0.2s;
}

.node-item:hover {
  background: var(--color-primary-light-1);
  border-color: var(--color-primary);
  color: var(--color-primary);
}

.node-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.node-desc {
  font-size: 12px;
  color: var(--color-text-3);
  margin: 0;
  line-height: 1.3;
}
</style>
