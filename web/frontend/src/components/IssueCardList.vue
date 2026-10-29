<script setup lang="ts">
import { computed, ref } from 'vue'

import type { IssueItem } from '../api/types'

const props = defineProps<{
  issues: IssueItem[]
  selectedId: string | null
}>()

const emit = defineEmits<{
  select: [issue: IssueItem]
  'play-evidence': [issue: IssueItem]
}>()

const showSuppressed = ref(false)

const visible = computed(() =>
  props.issues.filter((issue) => showSuppressed.value || (!issue.suppressed && issue.show_by_default)),
)

const suppressedCount = computed(() => props.issues.filter((issue) => issue.suppressed).length)

function severityLabel(severity: string) {
  return { high: '高', medium: '中', low: '低' }[severity] ?? severity
}

function hasInterval(issue: IssueItem) {
  const interval = issue.evidence_interval_s
  return Boolean(interval && interval.length >= 2 && interval[1] > interval[0])
}

function intervalTitle(issue: IssueItem) {
  const [start, end] = issue.evidence_interval_s ?? []
  return `播放 ${start.toFixed(2)}–${end.toFixed(2)}s`
}
</script>

<template>
  <aside class="issues card">
    <header class="issues-head">
      <h3>问题清单</h3>
      <span class="dim">{{ visible.length }} 项</span>
    </header>

    <div class="scroll">
      <article
        v-for="issue in visible"
        :key="issue.issue_id"
        class="issue"
        :class="[{ active: issue.issue_id === selectedId, muted: issue.suppressed }]"
        @click="emit('select', issue)"
      >
        <div class="issue-head">
          <span :class="['sev', `sev-${issue.severity}`]">{{ severityLabel(issue.severity) }}</span>
          <span class="title">{{ issue.title }}</span>
          <span v-if="issue.suppressed" class="badge">已抑制</span>
        </div>

        <p class="detail">{{ issue.detail }}</p>

        <div class="facts">
          <span v-if="issue.rep_ids.length">rep {{ issue.rep_ids.join(', ') }}</span>
          <span v-if="issue.frequency">{{ issue.frequency }}</span>
          <span v-if="issue.anchor_s !== null" class="mono">@{{ issue.anchor_s.toFixed(2) }}s</span>
        </div>

        <div v-if="issue.primary_metric" class="metric">
          <span class="mono">{{ issue.primary_metric }}</span>
          <span class="mono value">
            {{ issue.primary_value }}{{ issue.unit ?? '' }}
          </span>
        </div>

        <p class="cue">提示：{{ issue.cue }}</p>

        <div class="issue-foot">
          <span class="foot-left">
            <span v-if="issue.frame_url" class="badge pass">证据帧可用</span>
            <span v-else class="dim">无证据帧</span>
            <button
              v-if="hasInterval(issue)"
              class="play-evidence"
              :title="intervalTitle(issue)"
              @click.stop="emit('play-evidence', issue)"
            >
              ▶ 播放证据区间
            </button>
          </span>
          <span v-if="issue.confidence !== null" class="dim">
            置信度 {{ issue.confidence.toFixed(2) }}
          </span>
        </div>
      </article>

      <p v-if="visible.length === 0" class="dim empty">该样本没有需要关注的问题。</p>
    </div>

    <footer v-if="suppressedCount > 0" class="issues-foot">
      <button @click="showSuppressed = !showSuppressed">
        {{ showSuppressed ? '仅显示主问题' : `显示已抑制（${suppressedCount}）` }}
      </button>
    </footer>
  </aside>
</template>

<style scoped>
.issues {
  display: flex;
  flex-direction: column;
  min-height: 0;
  max-height: calc(60vh + 120px);
}

.issues-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  padding: 12px 14px 8px;
  border-bottom: 1px solid var(--border);
}

.issues-head h3 {
  margin: 0;
  font-size: 14px;
}

.scroll {
  flex: 1;
  overflow-y: auto;
  padding: 10px 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.issue {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
  background: var(--surface-2);
  cursor: pointer;
  transition: border-color 0.12s ease;
}

.issue:hover {
  border-color: var(--brand-soft);
}

.issue.active {
  border-color: var(--brand);
  background: #1a2b25;
}

.issue.muted {
  opacity: 0.6;
}

.issue-head {
  display: flex;
  align-items: center;
  gap: 8px;
}

.sev {
  font-size: 11px;
  font-weight: 600;
  padding: 0 6px;
  border-radius: 4px;
  border: 1px solid currentColor;
}

.title {
  font-weight: 600;
  font-size: 13px;
  flex: 1;
}

.detail {
  margin: 6px 0 0;
  font-size: 12px;
  color: var(--text-dim);
}

.facts {
  display: flex;
  gap: 10px;
  margin-top: 6px;
  font-size: 11px;
  color: var(--text-dim);
  flex-wrap: wrap;
}

.metric {
  display: flex;
  justify-content: space-between;
  margin-top: 6px;
  padding: 4px 8px;
  background: var(--surface);
  border-radius: 6px;
  font-size: 12px;
}

.value {
  color: var(--concentric);
}

.cue {
  margin: 6px 0 0;
  font-size: 12px;
  color: #b9cbcf;
}

.issue-foot {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 8px;
  font-size: 11px;
}

.foot-left {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.play-evidence {
  padding: 2px 8px;
  font-size: 11px;
  border-radius: 6px;
}

.mono {
  font-family: var(--mono);
}

.issues-foot {
  padding: 8px 12px 12px;
  border-top: 1px solid var(--border);
}

.issues-foot button {
  width: 100%;
  font-size: 12px;
}

.empty {
  text-align: center;
  padding: 18px 0;
}
</style>
