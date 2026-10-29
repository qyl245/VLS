<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
</script>

<template>
  <div class="shell">
    <header class="topbar">
      <div class="brand">
        <span class="dot" />
        <span>VLS 动作复核</span>
      </div>
      <nav>
        <RouterLink to="/" active-class="active">样本库</RouterLink>
        <RouterLink to="/reports" active-class="active">报告</RouterLink>
      </nav>
    </header>
    <main class="content">
      <RouterView />
    </main>
  </div>
</template>

<style scoped>
.shell {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.topbar {
  position: sticky;
  top: 0;
  z-index: 10;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 20px;
  background: rgba(15, 20, 23, 0.92);
  border-bottom: 1px solid var(--border);
  backdrop-filter: blur(6px);
}

.brand {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
  letter-spacing: 0.02em;
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--brand);
}

nav {
  display: flex;
  gap: 16px;
}

nav a {
  color: var(--text-dim);
  padding: 4px 2px;
  border-bottom: 2px solid transparent;
}

nav a.active {
  color: var(--text);
  border-bottom-color: var(--brand);
}

.content {
  flex: 1;
  width: 100%;
  max-width: 1440px;
  margin: 0 auto;
  padding: 18px 20px 40px;
}
</style>
