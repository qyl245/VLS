<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'

import { ApiError, api } from '../api/client'
import type { SampleSummary } from '../api/types'

import UploadPanel from '../components/UploadPanel.vue'

const samples = ref<SampleSummary[]>([])
const loading = ref(true)
const error = ref('')
const exerciseFilter = ref('all')
const showUpload = ref(false)

const filtered = computed(() =>
  samples.value.filter((item) => exerciseFilter.value === 'all' || item.exercise === exerciseFilter.value),
)

async function load() {
  loading.value = true
  error.value = ''
  try {
    samples.value = await api.listSamples()
  } catch (err) {
    error.value = err instanceof ApiError ? `${err.code}: ${err.message}` : '加载样本失败'
  } finally {
    loading.value = false
  }
}

onMounted(load)
</script>

<template>
  <div>
    <header class="page-head">
      <div>
        <h1>样本库</h1>
        <p class="dim">冻结产物直接回放：{{ samples.length }} 个样本，点击查看逐 rep 复核与证据帧。</p>
      </div>
      <div class="actions">
        <select v-model="exerciseFilter">
          <option value="all">全部动作</option>
          <option value="pushup">俯卧撑</option>
          <option value="squat">深蹲</option>
        </select>
        <button @click="showUpload = !showUpload">
          {{ showUpload ? '收起上传' : '上传新视频' }}
        </button>
      </div>
    </header>

    <UploadPanel v-if="showUpload" />

    <p v-if="error" class="error">{{ error }}</p>
    <p v-if="loading" class="dim">加载中…</p>

    <div v-else class="grid">
      <RouterLink
        v-for="item in filtered"
        :key="item.sample_id"
        :to="`/samples/${item.sample_id}`"
        class="tile card"
      >
        <div class="tile-head">
          <span class="id">{{ item.sample_id }}</span>
          <span :class="['badge', item.quality_status === 'pass' ? 'pass' : 'warn']">
            质量 {{ item.quality_status }}
          </span>
        </div>
        <div class="meta">
          <span>{{ item.exercise }}</span>
          <span class="dim">·</span>
          <span>{{ item.view }}</span>
          <span class="dim">·</span>
          <span>{{ item.duration_seconds.toFixed(1) }}s</span>
        </div>
        <div class="stats">
          <div>
            <strong>{{ item.rep_count }}</strong>
            <span class="dim">有效 rep</span>
          </div>
          <div>
            <strong>{{ item.active_issue_count }}</strong>
            <span class="dim">待关注问题</span>
          </div>
          <div>
            <strong>{{ item.rejected_rep_count }}</strong>
            <span class="dim">已剔除</span>
          </div>
        </div>
        <div class="sev">
          <span v-for="(count, sev) in item.severity_counts" :key="sev" :class="`sev-${sev}`">
            {{ sev }} {{ count }}
          </span>
          <span v-if="!item.has_media" class="badge fail">缺视频</span>
        </div>
      </RouterLink>
    </div>

    <p v-if="!loading && filtered.length === 0" class="dim">没有匹配的样本。</p>
  </div>
</template>

<style scoped>
.page-head {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}

h1 {
  margin: 0;
  font-size: 22px;
}

.actions {
  display: flex;
  gap: 10px;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 14px;
}

.tile {
  display: block;
  padding: 14px 16px;
  color: var(--text);
  transition: border-color 0.12s ease, transform 0.12s ease;
}

.tile:hover {
  border-color: var(--brand);
  transform: translateY(-1px);
}

.tile-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.id {
  font-family: var(--mono);
  font-size: 13px;
  letter-spacing: 0.02em;
}

.meta {
  margin-top: 4px;
  display: flex;
  gap: 6px;
  font-size: 12px;
}

.stats {
  display: flex;
  gap: 22px;
  margin-top: 12px;
}

.stats div {
  display: flex;
  flex-direction: column;
  line-height: 1.3;
}

.stats strong {
  font-size: 18px;
}

.stats span {
  font-size: 11px;
}

.sev {
  display: flex;
  gap: 10px;
  margin-top: 10px;
  font-size: 12px;
}

.error {
  color: #f19a90;
}
</style>
