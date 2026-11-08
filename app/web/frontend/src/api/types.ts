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

export type RepSelectionMode = 'primary' | 'boundary_refinement' | 'fallback' | 'none'

export interface RepProvenance {
  selection: RepSelectionMode
  top_boundary_source: string
  bottom_boundary_source: string
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
  provenance: RepProvenance | null
}

export interface RepSelectionCounts {
  accepted_cycle: number
  complete_cycle: number
  endpoint_partial_cycle: number
  rejected_candidate: number
}

export interface RepSelection {
  schema_version: 'vls-rep-selection-v1'
  mode: RepSelectionMode
  selected_source: 'repchain' | 'legacy' | null
  reason: string
  counts: RepSelectionCounts
  aliases: Record<string, number>
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
  decision: string | null
  family: string | null
  support_rep_ids: number[]
  anchor_fallback: boolean
  feedback_state: string | null
  shadow_only: boolean
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

export type FeedbackState =
  | 'active_issues'
  | 'no_reliable_issue'
  | 'low_confidence'
  | 'quality_reject'
  | 'no_reps'

export interface AssessmentSummary {
  conclusion: string
  quality_status: string
  rep_count: number
  evaluable_metric_count: number
  candidate_issue_count: number
  reportable_issue_count: number
  diagnostic_issue_count: number
  suppressed_issue_count: number
  abstained: boolean
  abstain_reason: string | null
  shadow_only: boolean
  product_actions_enabled: boolean
}

export interface FeedbackSummary {
  schema_version: string
  state: FeedbackState
  message: string
  card_count: number
  candidate_card_count: number
  reportable_card_count: number
  diagnostic_issue_count: number
  suppressed_issue_count: number
  shadow_only: boolean
  product_actions_enabled: boolean
  expert_calibrated: boolean
}

export interface SampleDetail {
  sample: SampleSummary
  video: VideoRef
  timeline: Timeline
  issues: IssueItem[]
  quality: QualityGate
  contract: Contract
  confidence: ConfidencePanel
  rep_selection: RepSelection | null
  assessment: AssessmentSummary | null
  feedback: FeedbackSummary | null
  state: FeedbackState
  report_url: string | null
  feedback_report_url: string | null
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
  filename: string | null
  exercise: string | null
  view: string | null
  upload_bytes: number | null
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
