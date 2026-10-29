<script setup lang="ts">
import { computed } from 'vue'

import type { EvidenceInterval, RepItem } from '../api/types'

const props = defineProps<{
  reps: RepItem[]
  evidence: EvidenceInterval[]
  duration: number
  currentTime: number
  activeIssueId: string | null
}>()

const emit = defineEmits<{ seek: [value: number] }>()

function pct(t: number) {
  if (!props.duration) return 0
  return Math.min(100, Math.max(0, (t / props.duration) * 100))
}

const repsWithGeometry = computed(() =>
  props.reps.map((rep) => {
    const span = Math.max(rep.end_s - rep.start_s, 0.001)
    return {
      rep,
      left: pct(rep.start_s),
      width: pct(rep.end_s) - pct(rep.start_s),
      eccPct: Math.min(100, Math.max(0, ((rep.bottom_s - rep.start_s) / span) * 100)),
    }
  }),
)

const activeRepId = computed(() => {
  const hit = props.reps.find((rep) => props.currentTime >= rep.start_s && props.currentTime <= rep.end_s)
  return hit?.rep_id ?? null
})

const evidenceBlocks = computed(() =>
  props.evidence.map((item) => ({
    item,
    left: pct(item.start_s),
    width: Math.max(pct(item.end_s) - pct(item.start_s), 0.4),
  })),
)

const ticks = computed(() => {
  if (!props.duration) return []
  const step = props.duration > 60 ? 10 : props.duration > 20 ? 5 : 2
  const out: number[] = []
  for (let t = 0; t <= props.duration; t += step) out.push(t)
  return out
})

function onTrackClick(event: MouseEvent) {
  const track = event.currentTarget as HTMLElement
  const rect = track.getBoundingClientRect()
  if (rect.width === 0) return
  const ratio = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width))
  emit('seek', ratio * props.duration)
}

function onRepClick(event: MouseEvent, rep: RepItem) {
  event.stopPropagation()
  emit('seek', rep.start_s)
}
</script>

<template>
  <div class="timeline card">
    <div class="labels">
      <span class="dim">rep 序列 · 下降相深绿 / 上升相浅绿 · 底端圆点</span>
      <span class="dim mono">{{ currentTime.toFixed(2) }}s / {{ duration.toFixed(2) }}s</span>
    </div>

    <div class="track" @click="onTrackClick">
      <div class="rep-row">
        <div
          v-for="geo in repsWithGeometry"
          :key="geo.rep.rep_id"
          class="rep-block"
          :class="{ active: geo.rep.rep_id === activeRepId }"
          :style="{
            left: `${geo.left}%`,
            width: `${geo.width}%`,
            background: `linear-gradient(to right, var(--eccentric) ${geo.eccPct}%, var(--concentric) ${geo.eccPct}%)`,
          }"
          :title="`rep ${geo.rep.rep_id} · ${geo.rep.start_s.toFixed(2)}–${geo.rep.end_s.toFixed(2)}s · 置信度 ${geo.rep.confidence.toFixed(2)}`"
          @click="onRepClick($event, geo.rep)"
        >
          <span class="rep-dot" :style="{ left: `${geo.eccPct}%` }" />
          <span v-if="geo.width > 3.5" class="rep-id">{{ geo.rep.rep_id }}</span>
        </div>
      </div>

      <div class="evidence-row">
        <div
          v-for="block in evidenceBlocks"
          :key="block.item.issue_id"
          class="evidence"
          :class="[`sev-${block.item.severity}`, { active: block.item.issue_id === activeIssueId }]"
          :style="{ left: `${block.left}%`, width: `${block.width}%` }"
          :title="`${block.item.issue_id} · ${block.item.start_s.toFixed(2)}–${block.item.end_s.toFixed(2)}s`"
        />
      </div>

      <div class="cursor" :style="{ left: `${pct(currentTime)}%` }" />
    </div>

    <div class="axis">
      <span v-for="t in ticks" :key="t" class="tick" :style="{ left: `${pct(t)}%` }">{{ t }}s</span>
    </div>
  </div>
</template>

<style scoped>
.timeline {
  padding: 10px 14px 6px;
}

.labels {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  margin-bottom: 8px;
}

.mono {
  font-family: var(--mono);
}

.track {
  position: relative;
  padding: 2px 0;
  cursor: crosshair;
}

.rep-row {
  position: relative;
  height: 30px;
}

.rep-block {
  position: absolute;
  top: 4px;
  height: 22px;
  min-width: 2px;
  border-radius: 4px;
  opacity: 0.82;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: visible;
  transition: opacity 0.12s ease;
}

.rep-block.active {
  opacity: 1;
  outline: 1px solid rgba(232, 238, 240, 0.65);
}

.rep-id {
  font-size: 11px;
  color: #04150e;
  font-weight: 600;
  pointer-events: none;
}

.rep-dot {
  position: absolute;
  bottom: -3px;
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: #e8eef0;
  transform: translateX(-50%);
  pointer-events: none;
}

.evidence-row {
  position: relative;
  height: 10px;
  margin-top: 4px;
  border-top: 1px dashed var(--border);
}

.evidence {
  position: absolute;
  top: 3px;
  height: 5px;
  border-radius: 2px;
  opacity: 0.75;
}

.evidence.sev-high {
  background: var(--danger);
}

.evidence.sev-medium {
  background: var(--warn);
}

.evidence.sev-low {
  background: var(--info);
}

.evidence.active {
  opacity: 1;
  box-shadow: 0 0 0 1px #e8eef0;
}

.cursor {
  position: absolute;
  top: 0;
  bottom: 12px;
  width: 1px;
  background: #e8eef0;
  pointer-events: none;
}

.cursor::before {
  content: '';
  position: absolute;
  top: -1px;
  left: -3px;
  border-left: 3.5px solid transparent;
  border-right: 3.5px solid transparent;
  border-top: 5px solid #e8eef0;
}

.axis {
  position: relative;
  height: 16px;
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text-dim);
}

.tick {
  position: absolute;
  transform: translateX(-50%);
}
</style>
