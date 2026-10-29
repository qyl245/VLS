<script setup lang="ts">
import { computed } from 'vue'

import type { QualityGate } from '../api/types'

const props = defineProps<{ quality: QualityGate }>()

interface Row {
  label: string
  value: string
  threshold: string
  verdict: 'pass' | 'warn' | 'fail'
}

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function compare(value: number | null, threshold: number | null, higherIsBetter = true): 'pass' | 'fail' {
  if (value === null || threshold === null) return 'pass'
  const ok = higherIsBetter ? value >= threshold : value <= threshold
  return ok ? 'pass' : 'fail'
}

const rows = computed<Row[]>(() => {
  const checks = props.quality.checks
  const thresholds = props.quality.thresholds
  const out: Row[] = []

  const add = (label: string, valueKey: string, thresholdKey: string, asPercent = true, higherIsBetter = true) => {
    const raw = num(checks[valueKey])
    const limit = num(thresholds[thresholdKey])
    const fmt = (x: number | null) => (x === null ? '—' : asPercent ? `${(x * 100).toFixed(1)}%` : x.toFixed(2))
    out.push({
      label,
      value: fmt(raw),
      threshold: `${higherIsBetter ? '≥' : '≤'} ${fmt(limit)}`,
      verdict: compare(raw, limit, higherIsBetter),
    })
  }

  add('姿态检出率', 'pose_detection_rate', 'pose_detection_pass')
  add('帧覆盖率', 'frame_coverage', 'body_in_frame_pass')
  add('关键点可见度 P10', 'q10_key_visibility', 'visibility_q10_pass')
  add('全部关键点同帧可见率', 'all_exercise_joints_visible_rate', 'visibility_q10_pass')
  add('人体完整入镜率', 'body_in_frame_rate', 'body_in_frame_pass')

  const reps = num(checks['rep_count'])
  const minReps = num(thresholds['min_complete_reps'])
  out.push({
    label: '完整 rep 数',
    value: reps === null ? '—' : String(reps),
    threshold: `≥ ${minReps ?? '—'}`,
    verdict: compare(reps, minReps),
  })

  const meanVis = num(checks['mean_key_visibility'])
  if (meanVis !== null) {
    out.push({
      label: '平均关键点可见度',
      value: `${(meanVis * 100).toFixed(1)}%`,
      threshold: '参考值',
      verdict: 'pass',
    })
  }
  return out
})

const verdictOf = computed(() => {
  if (props.quality.status !== 'pass') return 'fail'
  if (props.quality.warnings.length > 0) return 'warn'
  return 'pass'
})
</script>

<template>
  <section class="panel">
    <header class="panel-head">
      <h3>质量门判定</h3>
      <span :class="['badge', verdictOf]">质量 {{ quality.status }}</span>
    </header>

    <div class="tree">
      <div v-for="row in rows" :key="row.label" class="row">
        <span :class="['badge', row.verdict]">{{ row.verdict === 'pass' ? '通过' : '未通过' }}</span>
        <span class="label">{{ row.label }}</span>
        <span class="mono val">{{ row.value }}</span>
        <span class="mono thr dim">{{ row.threshold }}</span>
      </div>
    </div>

    <div v-if="quality.failures.length" class="notes">
      <p class="note fail">失败：{{ quality.failures.join('、') }}</p>
    </div>
    <div v-if="quality.warnings.length" class="notes">
      <p class="note warn">警告：{{ quality.warnings.join('、') }}</p>
    </div>
    <p v-if="!quality.failures.length && !quality.warnings.length" class="dim note-line">
      所有硬性门限通过，未触发任何降级提示。
    </p>
  </section>
</template>

<style scoped>
.panel {
  padding: 14px 16px;
}

.panel-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
}

.panel-head h3 {
  margin: 0;
  font-size: 14px;
}

.tree {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.row {
  display: grid;
  grid-template-columns: 64px 1fr 78px 88px;
  align-items: center;
  gap: 10px;
  font-size: 12px;
  padding: 4px 0;
  border-bottom: 1px dashed var(--border);
}

.badge {
  text-align: center;
}

.label {
  color: var(--text);
}

.val,
.thr {
  text-align: right;
  font-size: 11px;
}

.mono {
  font-family: var(--mono);
}

.notes {
  margin-top: 10px;
}

.note {
  margin: 0;
  font-size: 12px;
}

.note.fail {
  color: #f19a90;
}

.note.warn {
  color: #f0c56b;
}

.note-line {
  margin: 10px 0 0;
  font-size: 12px;
}
</style>
