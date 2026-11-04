import { createRouter, createWebHistory } from 'vue-router'

import AnalysisView from './views/AnalysisView.vue'
import JobsView from './views/JobsView.vue'
import LibraryView from './views/LibraryView.vue'
import ReportsView from './views/ReportsView.vue'

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'library', component: LibraryView },
    { path: '/samples/:sampleId', name: 'sample', component: AnalysisView, props: { kind: 'sample' } },
    { path: '/jobs', name: 'jobs', component: JobsView },
    { path: '/jobs/:jobId', name: 'job', component: AnalysisView, props: { kind: 'job' } },
    { path: '/reports', name: 'reports', component: ReportsView },
  ],
})
