<script setup lang="ts">
import type { Contract } from '../api/types'

defineProps<{ contract: Contract }>()
</script>

<template>
  <section class="panel">
    <header class="panel-head">
      <h3>动作契约</h3>
      <span class="badge">{{ contract.exercise }} / {{ contract.view }}</span>
    </header>

    <p v-if="contract.view_definition" class="definition">{{ contract.view_definition }}</p>

    <dl class="facts">
      <div>
        <dt>rep 单位</dt>
        <dd class="mono">{{ contract.rep_unit ?? '—' }}</dd>
      </div>
      <div>
        <dt>最少完整 rep</dt>
        <dd class="mono">{{ contract.min_complete_reps ?? '—' }}</dd>
      </div>
      <div v-if="contract.action_display">
        <dt>动作标签</dt>
        <dd>{{ contract.action_display }}</dd>
      </div>
      <div v-if="contract.phase?.phase_landmark">
        <dt>相位基准</dt>
        <dd class="mono">{{ contract.phase.phase_landmark }}</dd>
      </div>
    </dl>

    <div class="metrics">
      <div>
        <h4>可用指标</h4>
        <ul>
          <li v-for="metric in contract.usable_metrics" :key="metric" class="mono">{{ metric }}</li>
          <li v-if="contract.usable_metrics.length === 0" class="dim">无</li>
        </ul>
      </div>
      <div>
        <h4>当前视角禁用</h4>
        <ul>
          <li v-for="metric in contract.disabled_metrics" :key="metric" class="mono disabled">{{ metric }}</li>
          <li v-if="contract.disabled_metrics.length === 0" class="dim">无</li>
        </ul>
      </div>
    </div>
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
  margin-bottom: 8px;
}

.panel-head h3 {
  margin: 0;
  font-size: 14px;
}

.definition {
  margin: 0 0 10px;
  font-size: 12px;
  color: var(--text-dim);
}

.facts {
  display: flex;
  flex-wrap: wrap;
  gap: 18px;
  margin: 0 0 12px;
}

.facts div {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

dt {
  font-size: 11px;
  color: var(--text-dim);
}

dd {
  margin: 0;
  font-size: 13px;
}

.metrics {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}

h4 {
  margin: 0 0 6px;
  font-size: 12px;
  color: var(--text-dim);
  font-weight: 500;
}

ul {
  margin: 0;
  padding-left: 16px;
  font-size: 12px;
}

li {
  font-size: 11px;
  line-height: 1.7;
}

.mono {
  font-family: var(--mono);
}

.disabled {
  color: var(--text-dim);
  text-decoration: line-through;
}
</style>
