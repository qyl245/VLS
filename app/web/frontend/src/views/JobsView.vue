<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'

import { ApiError, api } from '../api/client'
import type { JobRecord, JobStatus } from '../api/types'

const jobs = ref<JobRecord[]>([])
const loading = ref(true)
const error = ref('')
let pollTimer: number | undefined

const ACTIVE_STATUSES = new Set<JobStatus>(['queued', 'validating', 'running'])

const activeCount = computed(() => jobs.value.filter((job) => ACTIVE_STATUSES.has(job.status)).length)

const STATUS_LABELS: Record<JobStatus, string> = {
  queued: '排队中',
  validating: '校验中',
  running: '分析中',
  succeeded: '已完成',
  failed: '失败',
  cancelled: '已取消',
}

const STAGE_LABELS: Record<string, string> = {
  queued: '等待执行',
  validate_upload: '校验视频',
  probe_video: '读取视频信息',
  build_job_manifest: '准备分析配置',
  analysis_v1: '姿态与动作分段',
  repchain: 'RepChain 动作链校验',
  issues: '识别动作问题',
  build_report: '生成证据帧与报告',
  validate_artifacts: '校验产物',
  publish_complete: '发布结果',
  done: '已完成',
}

function statusClass(status: JobStatus) {
  if (status === 'succeeded') return 'pass'
  if (status === 'failed' || status === 'cancelled') return 'fail'
  return 'warn'
}

function statusLabel(status: JobStatus) {
  return STATUS_LABELS[status]
}

function stageLabel(stage: string) {
  return STAGE_LABELS[stage] ?? stage
}

function exerciseLabel(exercise: string | null) {
  return { pushup: '俯卧撑', squat: '深蹲' }[exercise ?? ''] ?? exercise ?? '未知动作'
}

function formatBytes(value: number | null) {
  if (!value) return '—'
  if (value >= 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(1)} MB`
  return `${(value / 1024).toFixed(0)} KB`
}

function formatTime(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString('zh-CN', { hour12: false })
}

function stopPolling() {
  if (pollTimer !== undefined) {
    window.clearInterval(pollTimer)
    pollTimer = undefined
  }
}

function syncPolling() {
  if (activeCount.value > 0 && pollTimer === undefined) {
    pollTimer = window.setInterval(() => void load(true), 2500)
  } else if (activeCount.value === 0) {
    stopPolling()
  }
}

async function load(silent = false) {
  if (!silent) loading.value = true
  error.value = ''
  try {
    jobs.value = await api.listJobs()
    syncPolling()
  } catch (err) {
    error.value = err instanceof ApiError ? `${err.code}: ${err.message}` : '加载分析任务失败'
  } finally {
    if (!silent) loading.value = false
  }
}

onMounted(() => void load())
onBeforeUnmount(stopPolling)
</script>

<template>
  <div>
    <header class="page-head">
      <div>
        <h1>我的分析</h1>
        <p class="dim">
          上传视频和样本重分析任务：{{ jobs.length }} 条
          <span v-if="activeCount">· {{ activeCount }} 条进行中</span>
        </p>
      </div>
      <button :disabled="loading" @click="load()">{{ loading ? '刷新中…' : '刷新' }}</button>
    </header>

    <p v-if="error" class="error">{{ error }}</p>
    <p v-if="loading && jobs.length === 0" class="dim">加载中…</p>

    <div v-else class="list">
      <RouterLink
        v-for="job in jobs"
        :key="job.job_id"
        :to="{ name: 'job', params: { jobId: job.job_id } }"
        class="job card"
      >
        <div class="job-head">
          <span class="filename">{{ job.filename || job.sample_id || job.job_id }}</span>
          <span :class="['badge', statusClass(job.status)]">{{ statusLabel(job.status) }}</span>
        </div>

        <div class="meta">
          <span class="mono">{{ job.job_id }}</span>
          <span>{{ exerciseLabel(job.exercise) }}</span>
          <span>{{ job.view || '未知视角' }}</span>
          <span>{{ formatBytes(job.upload_bytes) }}</span>
          <span>{{ formatTime(job.created_at) }}</span>
        </div>

        <div v-if="ACTIVE_STATUSES.has(job.status)" class="progress">
          <div class="progress-track">
            <div class="progress-fill" :style="{ width: `${Math.round(job.progress * 100)}%` }" />
          </div>
          <span class="dim">{{ stageLabel(job.stage) }} · {{ Math.round(job.progress * 100) }}%</span>
        </div>

        <p v-if="job.error_message" class="error job-error">{{ job.error_message }}</p>

        <div class="foot">
          <span class="dim">{{ job.source_kind === 'upload' ? '上传视频' : '冻结样本重分析' }}</span>
          <span class="open">查看分析 →</span>
        </div>
      </RouterLink>

      <div v-if="jobs.length === 0" class="empty card">
        <p>还没有分析任务。</p>
        <RouterLink to="/">去样本库上传视频</RouterLink>
      </div>
    </div>
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

.list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.job {
  display: block;
  padding: 12px 16px;
  color: var(--text);
  transition: border-color 0.12s ease, transform 0.12s ease;
}

.job:hover {
  border-color: var(--brand);
  transform: translateY(-1px);
}

.job-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.filename {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 14px;
  font-weight: 600;
}

.meta {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-top: 5px;
  color: var(--text-dim);
  font-size: 11px;
}

.mono {
  font-family: var(--mono);
}

.progress {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 10px;
  font-size: 11px;
}

.progress-track {
  flex: 1;
  height: 5px;
  overflow: hidden;
  border-radius: 3px;
  background: var(--surface-2);
}

.progress-fill {
  height: 100%;
  background: var(--brand);
  transition: width 0.25s ease;
}

.foot {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  margin-top: 10px;
  font-size: 11px;
}

.open {
  color: var(--concentric);
}

.job-error {
  margin: 8px 0 0;
  font-size: 12px;
}

.error {
  color: #f19a90;
}

.empty {
  padding: 28px;
  text-align: center;
}

.empty p {
  margin: 0 0 8px;
}
</style>
