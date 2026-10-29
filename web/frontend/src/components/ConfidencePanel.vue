<script setup lang="ts">
import type { ConfidencePanel } from '../api/types'

defineProps<{ confidence: ConfidencePanel }>()

function barWidth(value: number) {
  return `${Math.min(100, Math.max(0, value * 100))}%`
}
</script>

<template>
  <section class="panel">
    <header class="panel-head">
      <h3>置信度</h3>
      <span class="dim">均值 {{ (confidence.mean_confidence * 100).toFixed(0) }}% · 最低 {{ (confidence.min_confidence * 100).toFixed(0) }}%</span>
    </header>

    <div class="bars">
      <div v-for="rep in confidence.reps" :key="rep.rep_id" class="bar-row">
        <span class="mono rep">rep {{ rep.rep_id }}</span>
        <div class="bar">
          <div
            class="fill"
            :class="[rep.confidence < 0.35 ? 'low' : rep.confidence < 0.6 ? 'mid' : 'ok']"
            :style="{ width: barWidth(rep.confidence) }"
          />
        </div>
        <span class="mono pct">{{ (rep.confidence * 100).toFixed(0) }}%</span>
      </div>
      <p v-if="confidence.reps.length === 0" class="dim">没有可用的 rep 置信度。</p>
    </div>

    <div v-if="confidence.low_confidence_rep_ids.length" class="low-note">
      低置信度 rep：<span class="mono">{{ confidence.low_confidence_rep_ids.join(', ') }}</span>，其结论仅供参考。
    </div>

    <details v-if="confidence.notes.length" class="notes">
      <summary>口径说明</summary>
      <ul>
        <li v-for="note in confidence.notes" :key="note">{{ note }}</li>
      </ul>
    </details>
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
  gap: 10px;
  flex-wrap: wrap;
}

.panel-head h3 {
  margin: 0;
  font-size: 14px;
}

.bars {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.bar-row {
  display: grid;
  grid-template-columns: 54px 1fr 42px;
  align-items: center;
  gap: 8px;
  font-size: 11px;
}

.bar {
  height: 8px;
  background: var(--surface-2);
  border-radius: 4px;
  overflow: hidden;
}

.fill {
  height: 100%;
  border-radius: 4px;
}

.fill.ok {
  background: var(--concentric);
}

.fill.mid {
  background: var(--warn);
}

.fill.low {
  background: var(--danger);
}

.pct {
  text-align: right;
}

.mono {
  font-family: var(--mono);
}

.low-note {
  margin-top: 10px;
  font-size: 12px;
  color: #f0c56b;
}

.notes {
  margin-top: 10px;
  font-size: 11px;
  color: var(--text-dim);
}

.notes summary {
  cursor: pointer;
}

ul {
  margin: 6px 0 0;
  padding-left: 16px;
}
</style>
