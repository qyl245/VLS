<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from 'vue'

const props = defineProps<{
  src: string
  overlayUrl: string | null
  overlayTitle: string
  overlayTime: number | null
}>()

const emit = defineEmits<{
  time: [value: number]
  'overlay-close': []
}>()

const video = ref<HTMLVideoElement | null>(null)
const playing = ref(false)
let rafId = 0
let lastEmitted = -1
let rangeEnd: number | null = null
let suppressSeek = false

function tick() {
  const el = video.value
  if (el && playing.value) {
    if (rangeEnd !== null && el.currentTime >= rangeEnd) {
      el.pause()
      el.currentTime = rangeEnd
      lastEmitted = rangeEnd
      emit('time', rangeEnd)
      rangeEnd = null
      return
    }
    const now = el.currentTime
    if (Math.abs(now - lastEmitted) > 0.01) {
      lastEmitted = now
      emit('time', now)
    }
    rafId = requestAnimationFrame(tick)
  }
}

function startLoop() {
  playing.value = true
  cancelAnimationFrame(rafId)
  rafId = requestAnimationFrame(tick)
}

function stopLoop() {
  playing.value = false
  cancelAnimationFrame(rafId)
}

function seek(value: number) {
  const el = video.value
  if (!el) return
  rangeEnd = null
  el.currentTime = Math.max(0, value)
  lastEmitted = el.currentTime
  emit('time', el.currentTime)
}

function playRange(start: number, end: number) {
  const el = video.value
  if (!el) return
  const from = Math.max(0, start)
  const to = Number.isFinite(el.duration) ? Math.min(el.duration, end) : end
  if (!(to > from)) {
    seek(from)
    return
  }
  rangeEnd = to
  suppressSeek = true
  el.currentTime = from
  lastEmitted = from
  emit('time', from)
  void el.play().finally(() => {
    suppressSeek = false
  })
}

function onSeeking() {
  // 用户手动拖拽进度条时退出区间播放模式
  if (!suppressSeek) rangeEnd = null
}

function togglePlay() {
  const el = video.value
  if (!el) return
  if (el.paused) {
    void el.play()
  } else {
    el.pause()
  }
}

function onLoaded() {
  const el = video.value
  if (el) {
    lastEmitted = el.currentTime
    emit('time', el.currentTime)
  }
}

watch(
  () => props.src,
  () => {
    stopLoop()
    lastEmitted = -1
  },
)

onBeforeUnmount(() => cancelAnimationFrame(rafId))

defineExpose({ seek, togglePlay, playRange })
</script>

<template>
  <div class="player card">
    <div class="stage">
      <div class="video-wrap">
        <video
          ref="video"
          :src="src"
          preload="metadata"
          playsinline
          controls
          @loadedmetadata="onLoaded"
          @seeking="onSeeking"
          @seeked="(e) => emit('time', (e.target as HTMLVideoElement).currentTime)"
          @play="startLoop"
          @pause="stopLoop"
          @ended="stopLoop"
        />
      </div>
      <aside v-if="overlayUrl" class="evidence">
        <div class="evidence-head">
          <span class="evidence-title">{{ overlayTitle }}</span>
          <span v-if="overlayTime !== null" class="evidence-time">
            t={{ overlayTime.toFixed(2) }}s · 证据帧
          </span>
          <button class="evidence-close" title="关闭证据帧" @click="emit('overlay-close')">×</button>
        </div>
        <img :src="overlayUrl" :alt="overlayTitle" />
      </aside>
    </div>
  </div>
</template>

<style scoped>
.player {
  overflow: hidden;
}

.stage {
  position: relative;
  background: #050809;
  display: flex;
  align-items: stretch;
  min-height: 260px;
}

.video-wrap {
  flex: 1 1 auto;
  min-width: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}

video {
  display: block;
  width: 100%;
  max-height: 60vh;
  background: #050809;
}

.evidence {
  flex: 0 0 clamp(220px, 30%, 360px);
  display: flex;
  flex-direction: column;
  background: #0a0e10;
  border-left: 1px solid var(--border);
  overflow: hidden;
}

.evidence-head {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  font-size: 12px;
  border-bottom: 1px solid var(--border);
}

.evidence-title {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.evidence-time {
  color: var(--text-dim);
  font-family: var(--mono);
}

.evidence-close {
  padding: 0 6px;
  line-height: 1.2;
  border-radius: 6px;
}

.evidence img {
  display: block;
  width: 100%;
  object-fit: contain;
}

@media (max-width: 860px) {
  .stage {
    flex-direction: column;
  }

  .evidence {
    flex: none;
    border-left: none;
    border-top: 1px solid var(--border);
  }
}
</style>
