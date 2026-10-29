<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { ApiError, api } from '../api/client'
import type { IssueItem, JobRecord, SampleDetail } from '../api/types'
import ConfidencePanel from '../components/ConfidencePanel.vue'
import ContractPanel from '../components/ContractPanel.vue'
import IssueCardList from '../components/IssueCardList.vue'
import PosePlayer from '../components/PosePlayer.vue'
import QualityGateTree from '../components/QualityGateTree.vue'
import RepTimeline from '../components/RepTimeline.vue'

const props = defineProps<{ kind: 'sample' | 'job' }>()

const route = useRoute()
const router = useRouter()

const detail = ref<SampleDetail | null>(null)
const job = ref<JobRecord | null>(null)
const loading = ref(true)
const error = ref('')

const currentTime = ref(0)
const selectedIssueId = ref<string | null>(null)
const player = ref<InstanceType<typeof PosePlayer> | null>(null)

const resourceId = computed(() =>
  String(props.kind === 'sample' ? route.params.sampleId ?? '' : route.params.jobId ?? ''),
)

let pollTimer: number | undefined
let queryTimer: number | undefined

const selectedIssue = computed(
  () => detail.value?.issues.find((issue) => issue.issue_id === selectedIssueId.value) ?? null,
)

const activeRepId = computed(() => {
  const reps = detail.value?.timeline.reps ?? []
  return reps.find((rep) => currentTime.value >= rep.start_s && currentTime.value <= rep.end_s)?.rep_id ?? null
})

const failedGate = computed(() => detail.value !== null && detail.value.quality.status !== 'pass')
const noReps = computed(() => detail.value !== null && detail.value.timeline.reps.length === 0)

function stopPolling() {
  if (pollTimer !== undefined) {
    window.clearInterval(pollTimer)
    pollTimer = undefined
  }
}

async function loadSample() {
  loading.value = true
  error.value = ''
  try {
    detail.value = await api.getSample(resourceId.value)
    restoreFromQuery()
  } catch (err) {
    error.value = err instanceof ApiError ? `${err.code}: ${err.message}` : '加载失败'
  } finally {
    loading.value = false
  }
}

async function refreshJob() {
  try {
    const record = await api.getJob(resourceId.value)
    job.value = record
    if (record.status === 'succeeded') {
      stopPolling()
      detail.value = await api.getJobResult(resourceId.value)
      restoreFromQuery()
      loading.value = false
    } else if (record.status === 'failed' || record.status === 'cancelled') {
      stopPolling()
      error.value = record.error_message ?? `任务${record.status}`
      loading.value = false
    }
  } catch (err) {
    error.value = err instanceof ApiError ? `${err.code}: ${err.message}` : '任务状态读取失败'
    loading.value = false
  }
}

function startJobPolling() {
  loading.value = true
  error.value = ''
  void refreshJob()
  stopPolling()
  pollTimer = window.setInterval(refreshJob, 1500)
}

function restoreFromQuery() {
  const t = Number(route.query.t)
  if (Number.isFinite(t) && t > 0) {
    seek(t)
  }
  const rep = Number(route.query.rep)
  if (Number.isFinite(rep) && rep > 0) {
    const hit = detail.value?.timeline.reps.find((item) => item.rep_id === rep)
    if (hit && !Number.isFinite(t)) seek(hit.start_s)
  }
}

function syncQuery() {
  if (queryTimer !== undefined) return
  queryTimer = window.setTimeout(() => {
    queryTimer = undefined
    void router.replace({
      query: {
        ...(activeRepId.value ? { rep: String(activeRepId.value) } : {}),
        t: currentTime.value.toFixed(2),
      },
    })
  }, 400)
}

function onTime(value: number) {
  currentTime.value = value
  syncQuery()
}

function seek(value: number) {
  player.value?.seek(value)
  currentTime.value = value
}

function selectIssue(issue: IssueItem) {
  selectedIssueId.value = issue.issue_id
  const anchor = issue.anchor_s ?? issue.evidence_interval_s?.[0]
  if (anchor !== null && anchor !== undefined) seek(anchor)
}

function closeOverlay() {
  selectedIssueId.value = null
}

watch(
  resourceId,
  () => {
    detail.value = null
    job.value = null
    stopPolling()
    if (props.kind === 'sample') void loadSample()
    else startJobPolling()
  },
  { immediate: true },
)

onBeforeUnmount(() => {
  stopPolling()
  if (queryTimer !== undefined) window.clearTimeout(queryTimer)
})
</script>

<template>
  <div class="analysis">
    <header class="bar card">
      <div class="bar-main" v-if="detail">
        <span class="mono id">{{ detail.sample.sample_id }}</span>
        <span class="badge">{{ detail.sample.exercise }}</span>
        <span class="badge">{{ detail.sample.view }}</span>
        <span class="dim">
          {{ detail.timeline.reps.length }} rep · {{ detail.video.timebase.duration_seconds.toFixed(1) }}s ·
          {{ detail.video.timebase.fps.toFixed(0) }}fps
        </span>
        <span :class="['badge', detail.quality.status === 'pass' ? 'pass' : 'fail']">
          质量 {{ detail.quality.status }}
        </span>
        <span class="dim">
          {{ detail.issues.filter((i) => !i.suppressed).length }} 问题 /
          {{ detail.issues.filter((i) => i.suppressed).length }} 已抑制
        </span>
      </div>
      <div class="bar-main" v-else>
        <span class="mono id">{{ resourceId }}</span>
        <span class="dim">{{ loading ? '加载中…' : '无数据' }}</span>
      </div>
      <div class="bar-actions">
        <span v-if="props.kind === 'job' && job" class="badge">{{ job.status }} · {{ job.stage }}</span>
        <a v-if="detail?.report_url" :href="detail.report_url" target="_blank" rel="noopener">原始报告 ↗</a>
        <RouterLink to="/">← 样本库</RouterLink>
      </div>
    </header>

    <div v-if="job && job.status !== 'succeeded'" class="job-progress card">
      <div class="progress-bar">
        <div class="progress-fill" :style="{ width: `${Math.round(job.progress * 100)}%` }" />
      </div>
      <div class="progress-meta dim">
        <span>{{ job.status }} · {{ job.stage }}</span>
        <span>{{ Math.round(job.progress * 100) }}%</span>
      </div>
      <div v-if="job.stages.length" class="stage-list">
        <span v-for="stage in job.stages" :key="stage.name" class="badge pass">{{ stage.name }}</span>
        <span class="badge">{{ job.stage }}</span>
      </div>
    </div>

    <p v-if="error" class="error">{{ error }}</p>
    <p v-else-if="loading" class="dim">加载中…</p>

    <template v-else-if="detail">
      <div v-if="failedGate" class="banner warn">
        质量门未通过（{{ detail.quality.failures.join('、') || detail.quality.status }}）。结论按 fail-closed
        处理，仅作参考，不作为训练建议。
      </div>
      <div v-if="noReps" class="banner">
        未检测到完整 rep（no_confident_set）。可查看质量门与置信度定位原因。
      </div>

      <div class="main">
        <div class="left">
          <PosePlayer
            ref="player"
            :src="detail.video.url"
            :overlay-url="selectedIssue?.frame_url ?? null"
            :overlay-title="selectedIssue?.title ?? ''"
            :overlay-time="selectedIssue?.anchor_s ?? null"
            @time="onTime"
            @overlay-close="closeOverlay"
          />
          <RepTimeline
            :reps="detail.timeline.reps"
            :evidence="detail.timeline.evidence_intervals"
            :duration="detail.video.timebase.duration_seconds"
            :current-time="currentTime"
            :active-issue-id="selectedIssueId"
            @seek="seek"
          />
        </div>

        <IssueCardList
          :issues="detail.issues"
          :selected-id="selectedIssueId"
          @select="selectIssue"
        />
      </div>

      <div class="collapsibles">
        <details class="fold card">
          <summary>质量门判定<span class="dim"> · {{ detail.quality.status }}</span></summary>
          <QualityGateTree :quality="detail.quality" />
        </details>
        <details class="fold card">
          <summary>动作契约<span class="dim"> · {{ detail.contract.exercise }} / {{ detail.contract.view }}</span></summary>
          <ContractPanel :contract="detail.contract" />
        </details>
        <details class="fold card">
          <summary>置信度<span class="dim"> · 均值 {{ (detail.confidence.mean_confidence * 100).toFixed(0) }}%</span></summary>
          <ConfidencePanel :confidence="detail.confidence" />
        </details>
      </div>
    </template>
  </div>
</template>

<style scoped>
.analysis {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 14px;
  flex-wrap: wrap;
}

.bar-main {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  font-size: 12px;
}

.bar-actions {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 12px;
}

.id {
  font-family: var(--mono);
  font-size: 13px;
  font-weight: 600;
}

.job-progress {
  padding: 10px 14px;
}

.progress-bar {
  height: 6px;
  background: var(--surface-2);
  border-radius: 3px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: var(--brand);
  transition: width 0.25s ease;
}

.progress-meta {
  display: flex;
  justify-content: space-between;
  margin-top: 6px;
  font-size: 11px;
}

.stage-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
}

.banner {
  padding: 10px 14px;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--surface);
  font-size: 12px;
}

.banner.warn {
  border-color: #7a6128;
  background: #241d10;
  color: #f0c56b;
}

.main {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 380px;
  gap: 12px;
  align-items: start;
}

.left {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: 0;
}

.collapsibles {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 2px;
}

.fold {
  overflow: hidden;
}

.fold summary {
  padding: 10px 16px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 500;
  list-style: none;
}

.fold summary::-webkit-details-marker {
  display: none;
}

.fold summary::before {
  content: '▸ ';
  color: var(--text-dim);
}

.fold[open] summary::before {
  content: '▾ ';
}

.error {
  color: #f19a90;
}

@media (max-width: 1080px) {
  .main {
    grid-template-columns: 1fr;
  }
}
</style>
