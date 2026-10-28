#!/usr/bin/env python3
"""Build and validate the T1-7A RepChain v3 protocol lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_DIR = ROOT / "results" / "task1_6" / "evaluation_quality_motion"
DEFAULT_SOURCES = ROOT / "configs" / "task1_6_sources.json"
DEFAULT_T1_6_PROTOCOL = ROOT / "configs" / "task1_6_eval_protocol.json"
DEFAULT_SCHEMA = ROOT / "schemas" / "task1_7_clip_analysis_v3.schema.json"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "task1_7" / "manifest"
DEFAULT_CONFIG_OUTPUT = ROOT / "configs" / "task1_7_protocol.json"

PROTOCOL_SCHEMA_VERSION = "task1-7-protocol-v1"
SOURCES_SCHEMA_VERSION = "task1-7-sources-lock-v1"
BASELINE_SCHEMA_VERSION = "task1-7-baseline-lock-v1"
SPLIT_SCHEMA_VERSION = "task1-7-split-lock-v1"
VALIDATION_SCHEMA_VERSION = "task1-7a-validation-v1"

BASELINE_EXPECTATIONS = {
    "verdict.decision": "NO_GO",
    "fit3d_holdout.clip_count": 32,
    "fit3d_holdout.cluster_count": 8,
    "fit3d_holdout.all_cameras.event_recall": 0.901042,
    "fit3d_holdout.all_cameras.event_precision": 0.935135,
    "fit3d_holdout.all_cameras.high_confidence_event_fp": 12,
    "fit3d_holdout.all_cameras.first_event_recall": 0.8125,
    "fit3d_holdout.all_cameras.last_event_recall": 0.90625,
    "fit3d_holdout.primary_camera.event_abs_error_median_frames": 6.0,
    "fit3d_holdout.primary_camera.event_abs_error_p90_frames": 16.8,
    "fit3d_holdout.primary_camera.high_confidence_event_fp": 0,
    "fit3d_holdout.count_consensus.count_mae": 0.0,
    "fit3d_holdout.count_consensus.exact_count_correct": 8,
    "fit3d_holdout.count_consensus.cluster_count": 8,
    "fit3d_holdout.top_to_top_interval_median_frames": 5.0,
    "fit3d_holdout.confidence.calibrated_threshold.high_confidence_fp": 2,
    "fit3d_holdout.coverage.gated.reportable_clip_count": 31,
    "fit3d_holdout.coverage.gated.clip_coverage": 0.96875,
    "fit3d_holdout.coverage.gated.reportable_cluster_count": 8,
    "qevd_sealed_holdout.segment_count": 6,
    "qevd_sealed_holdout.annotation_ready_count": 0,
    "qevd_sealed_holdout.reportable_segment_count": 3,
    "qevd_sealed_holdout.quality_gate_conflict_count": 0,
    "negative_controls.count": 5,
    "negative_controls.passed": 5,
}

PROHIBITED_CORE_FEATURES = [
    "dataset_name",
    "source_id",
    "video_id",
    "subject_id",
    "camera_id",
    "clip_id",
    "fixed_rep_count",
    "fixed_frame_offset",
    "fixed_plateau_width",
    "camera_order_or_consensus",
    "per_dataset_threshold",
    "sealed_holdout_gt_tuning",
    "coverage_reduction_as_improvement",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-protocol", action="store_true")
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--sources-config", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument(
        "--t1-6-protocol", type=Path, default=DEFAULT_T1_6_PROTOCOL
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--config-output", type=Path, default=DEFAULT_CONFIG_OUTPUT
    )
    parser.add_argument("--generated-at", default=None)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_path(payload: Any, dotted_path: str) -> Any:
    current = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"missing baseline path: {dotted_path}")
        current = current[part]
    return current


def write_checksums(directory: Path, names: Iterable[str]) -> None:
    lines = [
        f"{sha256_path(directory / name)}  {name}"
        for name in sorted(names)
        if (directory / name).is_file()
    ]
    (directory / "checksums.sha256").write_text("\n".join(lines) + "\n")


def validate_baseline(summary: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    errors = []
    for dotted_path, expected in BASELINE_EXPECTATIONS.items():
        observed = get_path(summary, dotted_path)
        passed = observed == expected
        checks.append(
            {
                "check_id": f"baseline:{dotted_path}",
                "expected": expected,
                "observed": observed,
                "passed": passed,
            }
        )
        if not passed:
            errors.append(
                f"baseline mismatch at {dotted_path}: "
                f"expected {expected!r}, observed {observed!r}"
            )
    if errors:
        raise ValueError("; ".join(errors))
    return checks


def extract_baseline(
    summary: dict[str, Any],
    summary_path: Path,
) -> dict[str, Any]:
    fit3d = summary["fit3d_holdout"]
    qevd = summary["qevd_sealed_holdout"]
    source = summary.get("source") or {}
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "task": "T1-7",
        "stage": "T1-7A",
        "baseline_id": "t1_6g_quality_motion",
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": sha256_path(summary_path),
        "summary_generated_at": summary.get("generated_at"),
        "decision": summary["verdict"]["decision"],
        "verdict": summary["verdict"],
        "fit3d": {
            "clip_count": fit3d["clip_count"],
            "cluster_count": fit3d["cluster_count"],
            "all_cameras": fit3d["all_cameras"],
            "primary_camera": fit3d["primary_camera"],
            "count_consensus": {
                key: fit3d["count_consensus"][key]
                for key in (
                    "cluster_count",
                    "gt_rep_sum",
                    "pred_rep_sum",
                    "count_mae",
                    "exact_count_correct",
                )
            },
            "top_to_top_interval_median_frames": fit3d[
                "top_to_top_interval_median_frames"
            ],
            "top_to_top_interval_p90_frames": fit3d[
                "top_to_top_interval_p90_frames"
            ],
            "primary_top_to_top_interval_median_frames": fit3d[
                "primary_top_to_top_interval_median_frames"
            ],
            "confidence": {
                "evaluation_unit_count": fit3d["confidence"][
                    "evaluation_unit_count"
                ],
                "calibrated_threshold": fit3d["confidence"][
                    "calibrated_threshold"
                ],
            },
            "coverage": fit3d["coverage"],
            "quality_status_counts": fit3d["quality_status_counts"],
            "quality_excluded_clip_ids": fit3d["quality_excluded_clip_ids"],
            "motion_excluded_clip_ids": fit3d["motion_excluded_clip_ids"],
        },
        "qevd_sealed": qevd,
        "robustness": summary.get("robustness") or [],
        "negative_controls": summary.get("negative_controls") or {},
        "go_no_go": summary.get("go_no_go") or [],
        "reproducibility": summary.get("reproducibility") or {},
        "algorithm_configs": {
            "segmenter": {
                "path": source.get("segmenter_config"),
                "sha256": source.get("segmenter_config_sha256"),
            },
            "chain": {
                "path": source.get("chain_config"),
                "sha256": source.get("chain_config_sha256"),
            },
            "confidence": {
                "path": source.get("confidence_config"),
                "sha256": source.get("confidence_config_sha256"),
            },
        },
    }


def build_sources_lock(
    sources: dict[str, Any],
    sources_path: Path,
    validation: list[dict[str, Any]],
) -> dict[str, Any]:
    clips = sources.get("clips") or []
    dataset_counts: dict[str, int] = {}
    for clip in clips:
        dataset = str(clip.get("dataset") or "unknown")
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
    return {
        "schema_version": SOURCES_SCHEMA_VERSION,
        "task": "T1-7",
        "stage": "T1-7A",
        "inherits": str(sources_path.resolve()),
        "inherits_sha256": sha256_path(sources_path),
        "adapter_contract": sources.get("adapter_contract"),
        "source_manifests": sources.get("source_manifests"),
        "split_policy": sources.get("split_policy"),
        "clip_count": len(clips),
        "dataset_counts": dataset_counts,
        "validation": {
            "passed": bool(validation)
            and all(check.get("passed") for check in validation),
            "checks": validation,
        },
    }


def _split_summary(
    clips: list[dict[str, Any]],
    split_role: str,
) -> dict[str, Any]:
    selected = [clip for clip in clips if clip.get("split_role") == split_role]
    action_counts: dict[str, int] = {}
    subject_ids = set()
    cluster_ids = set()
    for clip in selected:
        action = str(clip.get("action") or "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1
        if clip.get("subject_id") is not None:
            subject_ids.add(str(clip["subject_id"]))
        if clip.get("cluster_id") is not None:
            cluster_ids.add(str(clip["cluster_id"]))
    return {
        "clip_count": len(selected),
        "subject_count": len(subject_ids),
        "subject_ids": sorted(subject_ids),
        "cluster_count": len(cluster_ids),
        "action_counts": action_counts,
        "selected_for_phase_count": sum(
            bool(clip.get("selected_for_phase")) for clip in selected
        ),
        "clip_ids": sorted(str(clip["clip_id"]) for clip in selected),
    }


def build_split_lock(
    sources: dict[str, Any],
    t1_6_protocol: dict[str, Any],
    sources_path: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    fit3d_clips = [
        clip for clip in sources.get("clips") or [] if clip.get("dataset") == "Fit3D"
    ]
    split_policy = (sources.get("split_policy") or {}).get("fit3d") or {}
    split_sets = {
        "dev": set(split_policy.get("dev_subjects") or []),
        "regression_holdout": set(split_policy.get("holdout_subjects") or []),
        "smoke": set(split_policy.get("smoke_subjects") or []),
    }
    role_map = {
        "dev": "dev",
        "holdout": "regression_holdout",
        "smoke": "smoke",
    }
    errors = []
    names = list(split_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                errors.append(
                    f"subject overlap between {left} and {right}: {sorted(overlap)}"
                )
    subject_roles = {}
    for role, subjects in split_sets.items():
        for subject in subjects:
            subject_roles[subject] = role
    for clip in fit3d_clips:
        subject = str(clip.get("subject_id"))
        expected_role = subject_roles.get(subject)
        observed_role = role_map.get(str(clip.get("split_role")))
        if expected_role != observed_role:
            errors.append(
                f"split role mismatch for {clip.get('clip_id')}: "
                f"expected {expected_role}, observed {observed_role}"
            )

    fit3d_splits = {
        "dev": _split_summary(fit3d_clips, "dev"),
        "regression_holdout": _split_summary(fit3d_clips, "holdout"),
        "smoke": _split_summary(fit3d_clips, "smoke"),
    }
    if fit3d_splits["dev"]["clip_count"] != 32:
        errors.append(
            f"expected 32 Fit3D dev clips, found {fit3d_splits['dev']['clip_count']}"
        )
    if fit3d_splits["regression_holdout"]["clip_count"] != 32:
        errors.append(
            "expected 32 Fit3D regression holdout clips, found "
            f"{fit3d_splits['regression_holdout']['clip_count']}"
        )
    if fit3d_splits["smoke"]["clip_count"] != 6:
        errors.append(
            f"expected 6 Fit3D smoke clips, found {fit3d_splits['smoke']['clip_count']}"
        )

    sealed = t1_6_protocol.get("sealed_holdout") or {}
    sealed_segments = []
    for segment in sealed.get("segments") or []:
        sealed_segments.append(
            {
                "clip_id": segment.get("clip_id"),
                "video_id": segment.get("video_id"),
                "action": segment.get("action"),
                "trim": segment.get("trim"),
                "annotation_status": segment.get("annotation_status"),
            }
        )

    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "task": "T1-7",
        "stage": "T1-7A",
        "locked": not errors,
        "sources": str(sources_path.resolve()),
        "sources_sha256": sha256_path(sources_path),
        "t1_6_protocol": str(protocol_path.resolve()),
        "t1_6_protocol_sha256": sha256_path(protocol_path),
        "fit3d": {
            "unit": "subject",
            "dev": fit3d_splits["dev"],
            "regression_holdout": fit3d_splits["regression_holdout"],
            "smoke": fit3d_splits["smoke"],
            "holdout_is_observed": True,
            "independent_holdout_available": False,
            "claim": "regression_only_until_new_unseen_clips_are_locked",
        },
        "qevd_sealed": {
            "segment_count": len(sealed_segments),
            "annotation_status": sealed.get("annotation_status"),
            "formal_metric_evaluable": bool(
                sealed.get("cross_domain_acceptance_complete")
            ),
            "segments": sealed_segments,
            "holdout_artifacts": sealed.get("artifacts"),
        },
        "errors": errors,
    }


def build_acceptance(t1_6_protocol: dict[str, Any]) -> dict[str, Any]:
    inherited = t1_6_protocol.get("acceptance") or {}
    qevd = inherited.get("qevd_sealed") or {}
    return {
        "fit3d": {
            "gated_clip_coverage_gte": 0.95,
            "cluster_coverage": "8/8",
            "count_mae_lte": 0.5,
            "count_exact_at_least": "7/8",
            "count_must_not_degrade_from_t1_6g": True,
            "first_event_recall_gte": 0.75,
            "last_event_recall_gte": 0.75,
            "primary_event_median_abs_error_frames_lte": 5,
            "primary_event_p90_abs_error_frames_lte": 12,
            "primary_high_confidence_fp": 0,
            "all_camera_high_confidence_fp": 0,
            "complete_rep_high_confidence_fp": 0,
            "top_to_top_interval_median_abs_error_frames_lte": 3.0,
            "top_to_top_interval_must_not_degrade": True,
        },
        "qevd_sealed": {
            "count_exact_at_least": qevd.get("count_exact_at_least", "5/6 segments"),
            "count_mae_lte": qevd.get("count_mae_lte", 0.25),
            "boundary_abs_error_median_frames_lte": qevd.get(
                "boundary_abs_error_median_frames_lte", 5
            ),
            "boundary_abs_error_p95_frames_lte": qevd.get(
                "boundary_abs_error_p95_frames_lte", 12
            ),
            "f1_at_iou_0_30_gte": qevd.get("f1_at_iou_0_30_gte", 0.95),
            "high_confidence_fp": qevd.get("high_confidence_fp", 0),
            "non_rep_output_as_complete_rep": qevd.get(
                "non_rep_output_as_complete_rep", 0
            ),
            "requires_completed_human_gt": True,
        },
        "robustness": {
            "eligible_count_baseline": 31,
            "development_structure_match_min": "30/31",
            "final_structure_match_target": "31/31",
            "changed_clips_require_audit": True,
            "quality_reject_is_not_success": True,
        },
        "coverage": {
            "raw_and_gated_must_both_be_reported": True,
            "no_metric_improvement_by_new_exclusions": True,
        },
    }


def build_protocol_config(
    *,
    generated_at: str,
    baseline: dict[str, Any],
    baseline_path: Path,
    sources_lock: dict[str, Any],
    sources_lock_path: Path,
    split_lock: dict[str, Any],
    split_lock_path: Path,
    t1_6_protocol: dict[str, Any],
    t1_6_protocol_path: Path,
    schema_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "task": "T1-7",
        "stage": "T1-7A",
        "generated_at": generated_at,
        "status": "protocol_locked",
        "baseline": {
            "baseline_id": baseline["baseline_id"],
            "path": str(baseline_path.resolve()),
            "sha256": sha256_path(baseline_path),
            "summary_path": baseline["summary_path"],
            "summary_sha256": baseline["summary_sha256"],
            "decision": baseline["decision"],
            "algorithm_configs": baseline["algorithm_configs"],
        },
        "inherits": {
            "t1_6_protocol": {
                "path": str(t1_6_protocol_path.resolve()),
                "sha256": sha256_path(t1_6_protocol_path),
                "schema_version": t1_6_protocol.get("schema_version"),
            },
            "sources": {
                "path": str(sources_lock_path.resolve()),
                "sha256": sha256_path(sources_lock_path),
                "inherits": sources_lock["inherits"],
                "inherits_sha256": sources_lock["inherits_sha256"],
            },
            "split_lock": {
                "path": str(split_lock_path.resolve()),
                "sha256": sha256_path(split_lock_path),
                "locked": split_lock["locked"],
            },
        },
        "analysis_contract": {
            "schema_version": "task1-7-clip-analysis-v3",
            "schema_path": str(schema_path.resolve()),
            "schema_sha256": sha256_path(schema_path),
            "required_top_level_fields": [
                "schema_version",
                "clip_id",
                "algorithm",
                "status",
                "timebase",
                "signal_evidence",
                "phase_hypotheses",
                "action_verification",
                "quality_evidence",
                "confidence",
                "sets",
                "rejected_candidates",
                "non_rep_intervals",
            ],
            "quality_statuses": ["pass", "warn", "reject", "unknown"],
            "quality_scopes": ["clip", "set", "rep", "frame"],
            "action_verification_statuses": [
                "supported",
                "mismatch",
                "unsupported",
                "ambiguous",
            ],
            "confidence_layers": ["complete", "in_set", "combined"],
            "abstention_required": True,
        },
        "algorithm_core": {
            "contract_version": "repchain-v3",
            "dataset_independent": True,
            "same_frozen_config_across_sources": True,
            "prohibited_features": PROHIBITED_CORE_FEATURES,
            "selection_inputs": [
                "timebase",
                "multi_scale_signal_evidence",
                "phase_hypotheses",
                "action_verification",
                "state_intervals",
                "chain_shape_cost",
                "chain_timing_cost",
                "endpoint_evidence",
                "quality_evidence",
                "calibrated_confidence",
            ],
            "observation_fusion_scope": "outside_core_only",
        },
        "timebase_contract": {
            "preferred_input": "timestamps",
            "fallback_input": "constant_fps",
            "threshold_units": ["seconds", "phase_fraction"],
            "preserve_source_frame_numbers": True,
            "record_gap_and_interpolation": True,
            "explicit_variable_fps": True,
            "frame_only_thresholds_allowed": False,
        },
        "quality_contract": {
            "pass": "reportable",
            "warn": "reportable_with_confidence_downgrade",
            "reject": "fail_closed_for_scope",
            "unknown": "fail_closed_and_not_evaluable",
            "evidence_requires_span_source_value_threshold": True,
        },
        "confidence_contract": {
            "fit_scope": "dev_or_calibration_only",
            "sealed_holdout_used_for_fit": False,
            "component_scores_required": True,
            "risk_coverage_required": True,
            "high_confidence_threshold_frozen_before_holdout": True,
            "abstention_is_allowed": True,
        },
        "evaluation": {
            "fit3d_primary_unit": "subject_action",
            "fit3d_camera_observations": "repeated_observations_not_independent_samples",
            "qevd_primary_unit": "segment_or_video",
            "smoke_rule": "never_enter_error_statistics",
            "event_matching": t1_6_protocol.get("evaluation", {}).get(
                "event_matching"
            ),
            "high_confidence_threshold": t1_6_protocol.get("evaluation", {}).get(
                "high_confidence_threshold"
            ),
            "split": split_lock,
        },
        "acceptance": build_acceptance(t1_6_protocol),
        "generalization": {
            "fit3d_split_claim": "observed_regression_holdout",
            "qeud_sealed_formal_evaluable": split_lock["qevd_sealed"][
                "formal_metric_evaluable"
            ],
            "qeud_sealed_annotation_status": split_lock["qevd_sealed"][
                "annotation_status"
            ],
            "allow_cross_domain_claim": False,
            "reason": "QEVD sealed human GT is incomplete.",
        },
        "phase_status": {
            "T1-7A": "protocol_locked",
            "T1-7B": "not_started",
            "T1-7C": "not_started",
            "T1-7D": "not_started",
            "T1-7E": "not_started",
            "T1-7F": "not_started",
            "T1-7G": "not_started",
        },
    }


def validate_source_manifests(sources: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    errors = []
    for name, manifest in (sources.get("source_manifests") or {}).items():
        path = Path(manifest["path"])
        observed = sha256_path(path) if path.is_file() else None
        passed = observed == manifest.get("sha256")
        checks.append(
            {
                "check_id": f"source_manifest:{name}",
                "path": str(path),
                "expected_sha256": manifest.get("sha256"),
                "observed_sha256": observed,
                "passed": passed,
            }
        )
        if not passed:
            errors.append(f"source manifest hash mismatch: {name}")
    if errors:
        raise ValueError("; ".join(errors))
    return checks


def validate_schema(schema_path: Path) -> dict[str, Any]:
    schema = load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    return {
        "check_id": "analysis_schema_valid",
        "schema_path": str(schema_path.resolve()),
        "schema_sha256": sha256_path(schema_path),
        "passed": True,
    }


def build_protocol(args: argparse.Namespace) -> dict[str, Any]:
    generated_at = args.generated_at or utc_now()
    baseline_summary_path = args.baseline_dir / "summary.json"
    baseline_summary = load_json(baseline_summary_path)
    t1_6_protocol = load_json(args.t1_6_protocol)
    sources = load_json(args.sources_config)

    baseline_checks = validate_baseline(baseline_summary)
    source_checks = validate_source_manifests(sources)
    schema_check = validate_schema(args.schema)

    baseline = extract_baseline(baseline_summary, baseline_summary_path)
    sources_lock = build_sources_lock(
        sources, args.sources_config, baseline_checks
    )
    split_lock = build_split_lock(
        sources,
        t1_6_protocol,
        args.sources_config,
        args.t1_6_protocol,
    )
    if not split_lock["locked"]:
        raise ValueError("; ".join(split_lock["errors"]))

    baseline_path = args.output_dir / "baseline_t1_6g.json"
    sources_lock_path = args.output_dir / "sources.json"
    split_lock_path = args.output_dir / "split_lock.json"
    validation_path = args.output_dir / "validation.json"

    write_json(baseline_path, baseline)
    write_json(sources_lock_path, sources_lock)
    write_json(split_lock_path, split_lock)

    protocol = build_protocol_config(
        generated_at=generated_at,
        baseline=baseline,
        baseline_path=baseline_path,
        sources_lock=sources_lock,
        sources_lock_path=sources_lock_path,
        split_lock=split_lock,
        split_lock_path=split_lock_path,
        t1_6_protocol=t1_6_protocol,
        t1_6_protocol_path=args.t1_6_protocol,
        schema_path=args.schema,
    )
    write_json(args.config_output, protocol)

    validation = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "task": "T1-7",
        "stage": "T1-7A",
        "generated_at": generated_at,
        "passed": True,
        "checks": [*baseline_checks, *source_checks, schema_check],
        "errors": [],
        "counts": {
            "fit3d_dev_clips": split_lock["fit3d"]["dev"]["clip_count"],
            "fit3d_regression_holdout_clips": split_lock["fit3d"][
                "regression_holdout"
            ]["clip_count"],
            "fit3d_smoke_clips": split_lock["fit3d"]["smoke"]["clip_count"],
            "qevd_sealed_segments": split_lock["qevd_sealed"]["segment_count"],
            "qevd_sealed_annotation_ready": baseline_summary[
                "qevd_sealed_holdout"
            ]["annotation_ready_count"],
        },
        "artifacts": {
            "protocol": {
                "path": str(args.config_output.resolve()),
                "sha256": sha256_path(args.config_output),
            },
            "baseline": {
                "path": str(baseline_path.resolve()),
                "sha256": sha256_path(baseline_path),
            },
            "sources": {
                "path": str(sources_lock_path.resolve()),
                "sha256": sha256_path(sources_lock_path),
            },
            "split_lock": {
                "path": str(split_lock_path.resolve()),
                "sha256": sha256_path(split_lock_path),
            },
            "schema": schema_check,
        },
        "baseline_decision": baseline["decision"],
        "next_phase": "T1-7B",
    }
    write_json(validation_path, validation)
    write_checksums(
        args.output_dir,
        [
            "baseline_t1_6g.json",
            "sources.json",
            "split_lock.json",
            "validation.json",
        ],
    )
    return validation


def main() -> int:
    args = parse_args()
    if not args.build_protocol:
        raise SystemExit(
            "T1-7A only implements --build-protocol; algorithm runs begin in T1-7B"
        )
    validation = build_protocol(args)
    print(
        json.dumps(
            {
                "stage": validation["stage"],
                "passed": validation["passed"],
                "baseline_decision": validation["baseline_decision"],
                "counts": validation["counts"],
                "protocol": validation["artifacts"]["protocol"],
                "next_phase": validation["next_phase"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
