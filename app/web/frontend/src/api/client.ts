import type {
  ApiErrorBody,
  JobRecord,
  ReportItem,
  SampleDetail,
  SampleSummary,
} from './types'

const BASE = '/api/v1'

export class ApiError extends Error {
  code: string
  status: number

  constructor(status: number, code: string, message: string) {
    super(message)
    this.status = status
    this.code = code
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, init)
  if (!response.ok) {
    let code = 'http_error'
    let message = `请求失败（${response.status}）`
    try {
      const body = (await response.json()) as ApiErrorBody
      if (body?.error?.code) {
        code = body.error.code
        message = body.error.message
      }
    } catch {
      /* 保留默认错误信息 */
    }
    throw new ApiError(response.status, code, message)
  }
  return (await response.json()) as T
}

export const api = {
  listSamples: () => request<SampleSummary[]>('/samples'),
  getSample: (sampleId: string) => request<SampleDetail>(`/samples/${encodeURIComponent(sampleId)}`),

  listReports: () => request<{ reports: ReportItem[] }>('/reports'),

  listJobs: () => request<JobRecord[]>('/jobs'),
  getJob: (jobId: string) => request<JobRecord>(`/jobs/${encodeURIComponent(jobId)}`),
  getJobResult: (jobId: string) => request<SampleDetail>(`/jobs/${encodeURIComponent(jobId)}/result`),
  cancelJob: (jobId: string) =>
    request<JobRecord>(`/jobs/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' }),

  uploadJob: (file: File, exercise: string, view: string, rotation: string) => {
    const form = new FormData()
    form.append('file', file)
    form.append('exercise', exercise)
    form.append('view', view)
    form.append('rotation', rotation)
    return request<JobRecord>('/jobs/upload', { method: 'POST', body: form })
  },

  analyzeSample: (sampleId: string) =>
    request<JobRecord>('/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source_kind: 'sample', sample_id: sampleId }),
    }),
}
