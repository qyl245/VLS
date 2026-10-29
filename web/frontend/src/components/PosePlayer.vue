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

function tick() {
  const el = video.value
  if (el && playing.value) {
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
  el.currentTime = Math.max(0, value)
  lastEmitted = el.currentTime
  emit('time', el.currentTime)
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

defineExpose({ seek, togglePlay })
</script>

<template>
  <div class="player card">
    <div class="stage">
      <video
        ref="video"
        :src="src"
        preload="metadata"
        playsinline
        controls
        @loadedmetadata="onLoaded"
        @seeked="(e) => emit('time', (e.target as HTMLVideoElement).currentTime)"
        @play="startLoop"
        @pause="stopLoop"
        @ended="stopLoop"
      />
      <div v-if="overlayUrl" class="overlay">
        <div class="overlay-head">
          <span class="overlay-title">{{ overlayTitle }}</span>
          <span v-if="overlayTime !== null" class="overlay-time">
            t={{ overlayTime.toFixed(2) }}s · 证据帧
          </span>
          <button class="overlay-close" title="关闭证据帧" @click="emit('overlay-close')">×</button>
        </div>
        <img :src="overlayUrl" :alt="overlayTitle" />
      </div>
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
  align-items: center;
  justify-content: center;
  min-height: 260px;
}

video {
  display: block;
  width: 100%;
  max-height: 60vh;
  background: #050809;
}

.overlay {
  position: absolute;
  inset: auto 12px 12px auto;
  width: min(46%, 380px);
  background: rgba(10, 14, 16, 0.94);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
}

.overlay-head {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  font-size: 12px;
  border-bottom: 1px solid var(--border);
}

.overlay-title {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.overlay-time {
  color: var(--text-dim);
  font-family: var(--mono);
}

.overlay-close {
  padding: 0 6px;
  line-height: 1.2;
  border-radius: 6px;
}

.overlay img {
  display: block;
  width: 100%;
}
</style>
