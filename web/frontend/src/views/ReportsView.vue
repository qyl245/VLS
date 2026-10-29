<script setup lang="ts">
import { onMounted, ref } from 'vue'

import { ApiError, api } from '../api/client'
import type { ReportItem } from '../api/types'

const reports = ref<ReportItem[]>([])
const loading = ref(true)
const error = ref('')

onMounted(async () => {
  try {
    const payload = await api.listReports()
    reports.value = payload.reports
  } catch (err) {
    error.value = err instanceof ApiError ? `${err.code}: ${err.message}` : '加载报告失败'
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <div>
    <header class="page-head">
      <div>
        <h1>冻结报告</h1>
        <p class="dim">T1-4 已生成的单样本 HTML 报告，原样托管，不重新解析。</p>
      </div>
    </header>

    <p v-if="error" class="error">{{ error }}</p>
    <p v-if="loading" class="dim">加载中…</p>

    <div v-else class="list">
      <a
        v-for="item in reports"
        :key="item.report_id"
        :href="item.url"
        target="_blank"
        rel="noopener"
        class="row card"
      >
        <span class="mono id">{{ item.report_id }}</span>
        <span class="dim">{{ (item.size_bytes / 1024).toFixed(0) }} KB</span>
        <span :class="['badge', item.has_media ? 'pass' : 'warn']">
          {{ item.has_media ? '含视频' : '无视频' }}
        </span>
        <span class="open">打开 ↗</span>
      </a>
      <p v-if="reports.length === 0" class="dim">暂无报告。</p>
    </div>
  </div>
</template>

<style scoped>
.page-head {
  margin-bottom: 16px;
}

h1 {
  margin: 0;
  font-size: 22px;
}

.list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.row {
  display: grid;
  grid-template-columns: 1fr auto auto auto;
  align-items: center;
  gap: 16px;
  padding: 10px 14px;
  color: var(--text);
  font-size: 12px;
}

.row:hover {
  border-color: var(--brand);
}

.id {
  font-family: var(--mono);
  font-size: 13px;
}

.open {
  color: var(--concentric);
}

.error {
  color: #f19a90;
}
</style>
