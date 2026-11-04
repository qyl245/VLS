<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'

import { ApiError, api } from '../api/client'

const router = useRouter()
const file = ref<File | null>(null)
const exercise = ref('pushup')
const view = ref('front')
const rotation = ref('none')
const busy = ref(false)
const error = ref('')

function onPick(event: Event) {
  const input = event.target as HTMLInputElement
  file.value = input.files?.[0] ?? null
  error.value = ''
}

async function submit() {
  if (!file.value || busy.value) return
  busy.value = true
  error.value = ''
  try {
    const job = await api.uploadJob(file.value, exercise.value, view.value, rotation.value)
    router.push(`/jobs/${job.job_id}`)
  } catch (err) {
    error.value = err instanceof ApiError ? `${err.code}: ${err.message}` : '上传失败'
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <section class="upload card">
    <header>
      <h3>上传新视频分析</h3>
      <p class="dim">支持 mp4 / mov / avi / mkv / webm，3–120 秒。视频需保持单一机位与完整人体入镜。</p>
    </header>
    <div class="row">
      <input type="file" accept="video/*" @change="onPick" />
      <select v-model="exercise">
        <option value="pushup">俯卧撑 pushup</option>
        <option value="squat">深蹲 squat</option>
      </select>
      <select v-model="view">
        <option value="front">正视 front</option>
        <option value="rear">背视 rear</option>
        <option value="side_left">左侧视 side_left</option>
        <option value="side_right">右侧视 side_right</option>
        <option value="oblique_left">斜前左 oblique_left</option>
        <option value="oblique_right">斜前右 oblique_right</option>
        <option value="unknown">未知 unknown</option>
      </select>
      <select v-model="rotation">
        <option value="none">不旋转</option>
        <option value="90cw">顺时针 90°</option>
        <option value="180">180°</option>
        <option value="90ccw">逆时针 90°</option>
      </select>
      <button class="primary" :disabled="!file || busy" @click="submit">
        {{ busy ? '提交中…' : '开始分析' }}
      </button>
    </div>
    <p v-if="error" class="error">{{ error }}</p>
  </section>
</template>

<style scoped>
.upload {
  padding: 14px 16px;
  margin-bottom: 18px;
}

h3 {
  margin: 0 0 2px;
  font-size: 15px;
}

p {
  margin: 4px 0 0;
}

.row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 12px;
  align-items: center;
}

.error {
  color: #f19a90;
}
</style>
