/** 与后端 Pydantic DTO 对齐的契约类型（手写，配合 tests/test_web_api.py 防漂移）。 */

export interface Timebase {
  fps: number
  duration_seconds: number
  clip_local_origin_s: number
  source_trim_start_frame: number
  video_time_origin: 'clip_start'
  variable_fps: boolean
}

export interface VideoRef {
  url: string
  timebase: Timebase
}

export interface SampleSummary {
  sample_id: string
  exercise: string
  action_display: string | null
  view: string
  split: string
  status: string
  quality_status: string
  rep_count: number
  rejected_rep_count: number
  active_issue_count: number
  suppressed_issue_count: number
  severity_counts: Record<string, number>
  has_media: boolean
  has_report: boolean
  duration_seconds: number
}

export interface RepItem {
  rep_id: number
  start_s: number
  end_s: number
  bottom_s: number
  duration_s: number
  confidence: number
  review_priority: string
  status: string
  quality_flags: string[]
  reject_reasons: string[]
}

export interface PhaseItem {
  rep_id: number
  phase: string
  time_s: number
}

export interface EvidenceInterval {
  issue_id: string
  severity: string
  start_s: number
  end_s: number
  anchor_s: number
}

export interface Timeline {
  reps: RepItem[]
  phases: PhaseItem[]
  evidence_intervals: EvidenceInterval[]
}

export interface IssueItem {
  issue_id: string
  issue_code: string
  title: string
  detail: string
  cue: string
  severity: string
  confidence: number | null
  rep_ids: number[]
  representative_rep_id: number | null
  frequency: string
  frequency_ratio: number
  anchor_s: number | null
  evidence_interval_s: number[] | null
  frame_url: string | null
  primary_metric: string | null
  primary_value: number | null
  unit: string | null
  thresholds: Record<string, unknown>
  highlight_joints: string[]
  suppressed: boolean
  show_by_default: boolean
}

export interface QualityGate {
  status: string
  failures: string[]
  warnings: string[]
  checks: Record<string, unknown>
  thresholds: Record<string, unknown>
  view_support: Record<string, unknown>
}

export interface Contract {
  exercise: string
  view: string
  view_definition: string | null
  action_display: string | null
  rep_unit: string | null
  min_complete_reps: number | null
  usable_metrics: string[]
  disabled_metrics: string[]
  metric_groups: Record<string, string[]>
  phase: Record<string, unknown>
}

export interface ConfidenceRep {
  rep_id: number
  confidence: number
  review_priority: string
}

export interface ConfidencePanel {
  mean_confidence: number
  min_confidence: number
  low_confidence_rep_ids: number[]
  reps: ConfidenceRep[]
  checks: Record<string, unknown>
  notes: string[]
}

export interface SampleDetail {
  sample: SampleSummary
  video: VideoRef
  timeline: Timeline
  issues: IssueItem[]
  quality: QualityGate
  contract: Contract
  confidence: ConfidencePanel
  report_url: string | null
}

export type JobStatus = 'queued' | 'validating' | 'running' | 'succeeded' | 'failed' | 'cancelled'

export interface JobStage {
  name: string
  status: string
  started_at: string | null
  finished_at: string | null
  duration_ms: number | null
  error: string | null
}

export interface JobRecord {
  job_id: string
  source_kind: 'sample' | 'upload'
  sample_id: string | null
  status: JobStatus
  stage: string
  progress: number
  created_at: string
  started_at: string | null
  finished_at: string | null
  error_code: string | null
  error_message: string | null
  pipeline_version: string
  stages: JobStage[]
  artifact_manifest_url: string | null
}

export interface ReportItem {
  report_id: string
  url: string
  has_media: boolean
  size_bytes: number
}

export interface ApiErrorBody {
  error: { code: string; message: string; [key: string]: unknown }
}
