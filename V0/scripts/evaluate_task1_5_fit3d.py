#!/usr/bin/env python3
"""Evaluate MediaPipe pose/angle baselines against Fit3D 3D ground truth."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import segment_reps
import task1_5_metrics as metrics


DEFAULT_MANIFEST = ROOT / "configs" / "task1_5_fit3d_clips.json"
DEFAULT_ANALYSIS_DIR = ROOT / "results" / "task1_5" / "clip_analysis"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "task1_5" / "evaluation"

ANGLE_NAMES_BY_EXERCISE = {
    "squat": (
        "left_knee_flexion",
        "right_knee_flexion",
        "left_hip_flexion",
        "right_hip_flexion",
        "torso_flexion",
    ),
    "pushup": (
        "left_elbow_flexion",
        "right_elbow_flexion",
        "left_hip_flexion",
        "right_hip_flexion",
        "torso_flexion",
    ),
}

FLEXION_ANGLES = {
    "squat": "knee_flexion",
    "pushup": "elbow_flexion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=("all", "dev", "holdout"), default="all")
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261026)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def angle_side(angle_name: str) -> str:
    if angle_name.startswith("left_"):
        return "left"
    if angle_name.startswith("right_"):
        return "right"
    return "bilateral"


def selected_signal(analysis: dict[str, Any]) -> tuple[str | None, str | None, str | None, bool]:
    segmentation = analysis.get("segmentation") or {}
    selection = segmentation.get("signal_selection")
    if selection is not None:
        return (
            selection.get("selected_angle"),
            selection.get("selected_side"),
            selection.get("selection_reason"),
            bool(selection.get("low_confidence", False)),
        )
    angle = segmentation.get("phase_angle")
    side = angle_side(angle) if angle else None
    reason = "frontal_or_rear_default_phase_signal"
    low_confidence = False
    return angle, side, reason, low_confidence


def quality_row(clip: dict[str, Any], analysis: dict[str, Any] | None) -> dict[str, Any]:
    if analysis is None:
        return {
            "clip_id": clip["clip_id"],
            "subject_id": clip["subject_id"],
            "camera_id": clip["camera_id"],
            "exercise": clip["exercise"],
            "view": clip["view"],
            "split": clip["split"],
            "camera_side": (clip.get("view_geometry") or {}).get("camera_side"),
            "analysis_status": "missing",
            "quality_status": "missing",
            "rep_count": 0,
            "pose_detection_rate": None,
            "q10_key_visibility": None,
            "body_in_frame_rate": None,
            "selected_signal": None,
            "selected_side": None,
            "selection_reason": None,
            "selection_low_confidence": None,
        }
    checks = (analysis.get("quality") or {}).get("checks") or {}
    selected, side, reason, low_confidence = selected_signal(analysis)
    return {
        "clip_id": clip["clip_id"],
        "subject_id": clip["subject_id"],
        "camera_id": clip["camera_id"],
        "exercise": clip["exercise"],
        "view": clip["view"],
        "split": clip["split"],
        "camera_side": (clip.get("view_geometry") or {}).get("camera_side"),
        "analysis_status": analysis.get("status"),
        "quality_status": analysis.get("quality_status"),
        "rep_count": analysis.get("rep_count"),
        "pose_detection_rate": checks.get("pose_detection_rate"),
        "q10_key_visibility": checks.get("q10_key_visibility"),
        "body_in_frame_rate": checks.get("body_in_frame_rate"),
        "selected_signal": selected,
        "selected_side": side,
        "selection_reason": reason,
        "selection_low_confidence": low_confidence,
    }


def evaluate_clip(
    clip: dict[str, Any], analysis_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    analysis_path = analysis_dir / clip["clip_id"] / "analysis.json"
    if not analysis_path.is_file():
        return [], quality_row(clip, None)
    analysis = json.loads(analysis_path.read_text())
    quality = quality_row(clip, analysis)
    if not clip.get("has_gt"):
        return [], quality

    wide_path = analysis_dir / clip["clip_id"] / "angles" / "angles_wide.csv"
    if not wide_path.is_file():
        quality["analysis_status"] = "missing_angles"
        return [], quality
    _, wide = segment_reps.load_wide(wide_path)
    joints_path = Path(clip["gt"]["joints3d_25"])
    joints = np.asarray(
        json.loads(joints_path.read_text())["joints3d_25"], dtype=np.float64
    )
    selected, selected_side, selection_reason, low_confidence = selected_signal(analysis)
    selection = (analysis.get("segmentation") or {}).get("signal_selection") or {}
    near_side = selection.get("near_side")
    near_visibility = ((selection.get("candidates") or {}).get(near_side) or {}).get(
        "visibility_median"
    )
    geometry = clip.get("view_geometry") or {}

    rows = []
    for angle_name in ANGLE_NAMES_BY_EXERCISE[clip["exercise"]]:
        if angle_name not in wide:
            continue
        reference = metrics.fit3d_angle_series(joints, clip["exercise"], angle_name)
        error = metrics.angle_error_stats(wide[angle_name], reference)
        rows.append(
            {
                "clip_id": clip["clip_id"],
                "subject_id": clip["subject_id"],
                "camera_id": clip["camera_id"],
                "exercise": clip["exercise"],
                "action_variant": clip.get("action_variant"),
                "split": clip["split"],
                "view": clip["view"],
                "view_angle_deg": geometry.get("angle_from_frontal_normal_deg"),
                "camera_side": geometry.get("camera_side"),
                "angle_name": angle_name,
                "angle_side": angle_side(angle_name),
                "selected_signal": bool(angle_name == selected),
                "selected_side": selected_side,
                "near_side": near_side,
                "near_side_visibility_median": near_visibility,
                "selection_reason": selection_reason,
                "selection_low_confidence": low_confidence,
                **error,
            }
        )
    return rows, quality


def group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["exercise"], row["view"], row["angle_name"])].append(row)
    return dict(grouped)


def summarize_group(
    rows: list[dict[str, Any]], seed: int, samples: int
) -> dict[str, Any]:
    valid = [row for row in rows if row["mae_deg"] is not None]
    cluster_values: defaultdict[str, list[float]] = defaultdict(list)
    for row in valid:
        cluster_values[f"{row['subject_id']}_{row['exercise']}"].append(
            float(row["mae_deg"])
        )
    cluster_means = {
        cluster: float(np.mean(values)) for cluster, values in cluster_values.items()
    }
    bootstrap = metrics.cluster_bootstrap_ci(cluster_means, seed=seed, samples=samples)
    clip_mae = np.asarray([row["mae_deg"] for row in valid], dtype=np.float64)
    return {
        "clip_count": len(rows),
        "scored_clip_count": len(valid),
        "cluster_count": bootstrap["clusters"],
        "mean_clip_mae_deg": round(float(clip_mae.mean()), 6) if valid else None,
        "median_clip_mae_deg": round(float(np.median(clip_mae)), 6) if valid else None,
        "max_clip_mae_deg": round(float(clip_mae.max()), 6) if valid else None,
        "cluster_mean_mae_deg": bootstrap["mean"],
        "cluster_bootstrap_ci95_deg": [bootstrap["ci95_low"], bootstrap["ci95_high"]],
        "mean_bias_deg": round(
            float(np.mean([row["bias_deg"] for row in valid])), 6
        )
        if valid
        else None,
        "mean_p90_abs_error_deg": round(
            float(np.mean([row["p90_abs_error_deg"] for row in valid])), 6
        )
        if valid
        else None,
    }


def split_summary(
    rows: list[dict[str, Any]], split: str, seed: int, samples: int
) -> list[dict[str, Any]]:
    subset = rows if split == "all" else [row for row in rows if row["split"] == split]
    result = []
    for (exercise, view, angle_name), group in sorted(group_rows(subset).items()):
        result.append(
            {
                "exercise": exercise,
                "view": view,
                "angle_name": angle_name,
                **summarize_group(group, seed=seed, samples=samples),
            }
        )
    return result


def gate_for_mae(mae: float | None) -> str:
    if mae is None:
        return "insufficient_data"
    if mae <= 10.0:
        return "pass"
    if mae <= 15.0:
        return "low_confidence"
    return "disabled"


def selected_signal_summary(
    rows: list[dict[str, Any]], seed: int, samples: int
) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["selected_signal"]]
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[(row["exercise"], row["view"])].append(row)

    result = []
    for (exercise, view), group in sorted(grouped.items()):
        holdout = [row for row in group if row["split"] == "holdout"]
        gate_rows = holdout if holdout else group
        summary = summarize_group(gate_rows, seed=seed, samples=samples)
        all_summary = summarize_group(group, seed=seed, samples=samples)
        gate_mae = summary["cluster_mean_mae_deg"]
        result.append(
            {
                "exercise": exercise,
                "view": view,
                "flexion_group": FLEXION_ANGLES[exercise],
                "selected_angles": sorted({row["angle_name"] for row in group}),
                "gate_split": "holdout" if holdout else "all",
                "all": all_summary,
                "gate_data": summary,
                "pose_gate": gate_for_mae(gate_mae),
                "note": (
                    "shallow_oblique_only_no_true_side_squat"
                    if exercise == "squat" and view.startswith("oblique")
                    else "frontal_or_rear_not_side_validation"
                    if exercise == "squat"
                    else "side_view_pushup"
                ),
            }
        )
    return result


def quality_summary(quality_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in quality_rows:
        grouped[(row["exercise"], row["view"])].append(row)
    result = []
    for (exercise, view), group in sorted(grouped.items()):
        detection = [
            float(row["pose_detection_rate"])
            for row in group
            if row["pose_detection_rate"] is not None
        ]
        result.append(
            {
                "exercise": exercise,
                "view": view,
                "clip_count": len(group),
                "status_counts": dict(Counter(row["analysis_status"] for row in group)),
                "quality_status_counts": dict(
                    Counter(row["quality_status"] for row in group)
                ),
                "mean_pose_detection_rate": round(float(np.mean(detection)), 6)
                if detection
                else None,
                "min_pose_detection_rate": round(float(np.min(detection)), 6)
                if detection
                else None,
                "pose_detection_pass_clips": sum(value >= 0.90 for value in detection),
                "body_in_frame_pass_clips": sum(
                    row["body_in_frame_rate"] is not None
                    and float(row["body_in_frame_rate"]) >= 0.95
                    for row in group
                ),
                "q10_visibility_pass_clips": sum(
                    row["q10_key_visibility"] is not None
                    and float(row["q10_key_visibility"]) >= 0.50
                    for row in group
                ),
                "low_confidence_signal_clips": sum(
                    bool(row["selection_low_confidence"]) for row in group
                ),
            }
        )
    return result


def signal_selection_counts(quality_rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = Counter()
    side_counts = Counter()
    total_side_clips = 0
    low_confidence = 0
    for row in quality_rows:
        if row["view"] not in metrics.SIDE_VIEWS:
            continue
        total_side_clips += 1
        reason = row.get("selection_reason")
        if reason:
            reasons[str(reason)] += 1
        if row.get("selected_side") in {"left", "right"}:
            side_counts[str(row["selected_side"])] += 1
        low_confidence += int(bool(row.get("selection_low_confidence")))
    return {
        "side_view_clips": total_side_clips,
        "selected_side_counts": dict(sorted(side_counts.items())),
        "selection_reason_counts": dict(sorted(reasons.items())),
        "low_confidence_signal_clips": low_confidence,
    }


def flatten_selected_summary(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        all_summary = item["all"]
        gate_data = item["gate_data"]
        ci = gate_data["cluster_bootstrap_ci95_deg"]
        rows.append(
            {
                "exercise": item["exercise"],
                "view": item["view"],
                "flexion_group": item["flexion_group"],
                "selected_angles": ";".join(item["selected_angles"]),
                "gate_split": item["gate_split"],
                "clip_count": gate_data["clip_count"],
                "cluster_count": gate_data["cluster_count"],
                "gate_mean_clip_mae_deg": gate_data["mean_clip_mae_deg"],
                "gate_cluster_mean_mae_deg": gate_data["cluster_mean_mae_deg"],
                "gate_ci95_low_deg": ci[0],
                "gate_ci95_high_deg": ci[1],
                "gate_mean_bias_deg": gate_data["mean_bias_deg"],
                "gate_mean_p90_abs_error_deg": gate_data["mean_p90_abs_error_deg"],
                "all_clip_count": all_summary["clip_count"],
                "all_mean_clip_mae_deg": all_summary["mean_clip_mae_deg"],
                "all_mean_bias_deg": all_summary["mean_bias_deg"],
                "all_mean_p90_abs_error_deg": all_summary["mean_p90_abs_error_deg"],
                "pose_gate": item["pose_gate"],
                "note": item["note"],
            }
        )
    return rows


def camera_selected_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["selected_signal"]]
    grouped: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[(row["exercise"], row["camera_id"], row["angle_name"])].append(row)
    result = []
    for (exercise, camera_id, angle_name), group in sorted(grouped.items()):
        valid = [row for row in group if row["mae_deg"] is not None]
        result.append(
            {
                "exercise": exercise,
                "camera_id": camera_id,
                "view": group[0]["view"],
                "angle_name": angle_name,
                "splits": ";".join(sorted({str(row["split"]) for row in group})),
                "clip_count": len(group),
                "scored_clip_count": len(valid),
                "mean_clip_mae_deg": round(
                    float(np.mean([row["mae_deg"] for row in valid])), 6
                )
                if valid
                else None,
                "mean_bias_deg": round(
                    float(np.mean([row["bias_deg"] for row in valid])), 6
                )
                if valid
                else None,
                "mean_p90_abs_error_deg": round(
                    float(np.mean([row["p90_abs_error_deg"] for row in valid])), 6
                )
                if valid
                else None,
                "mean_near_side_visibility": round(
                    float(
                        np.mean(
                            [
                                row["near_side_visibility_median"]
                                for row in group
                                if row["near_side_visibility_median"] is not None
                            ]
                        )
                    ),
                    6,
                )
                if any(row["near_side_visibility_median"] is not None for row in group)
                else None,
            }
        )
    return result


def near_far_side_diagnostic(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair near/far flexion errors for diagnosis only; never product fallback."""
    by_clip: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["view"] in metrics.SIDE_VIEWS:
            by_clip[row["clip_id"]].append(row)
    result = []
    for clip_id, clip_rows in sorted(by_clip.items()):
        first = clip_rows[0]
        near_side = first.get("near_side")
        if near_side not in {"left", "right"}:
            continue
        near_angle = f"{near_side}_{FLEXION_ANGLES[first['exercise']]}"
        far_side = "right" if near_side == "left" else "left"
        far_angle = f"{far_side}_{FLEXION_ANGLES[first['exercise']]}"
        by_angle = {row["angle_name"]: row for row in clip_rows}
        near_row = by_angle.get(near_angle)
        far_row = by_angle.get(far_angle)
        if near_row is None or far_row is None:
            continue
        near_mae = near_row["mae_deg"]
        far_mae = far_row["mae_deg"]
        result.append(
            {
                "clip_id": clip_id,
                "subject_id": first["subject_id"],
                "exercise": first["exercise"],
                "camera_id": first["camera_id"],
                "view": first["view"],
                "split": first["split"],
                "near_side": near_side,
                "far_side": far_side,
                "near_side_visibility_median": first.get("near_side_visibility_median"),
                "near_angle_name": near_angle,
                "near_mae_deg": near_mae,
                "far_angle_name": far_angle,
                "far_mae_deg": far_mae,
                "near_minus_far_mae_deg": (
                    round(float(near_mae) - float(far_mae), 6)
                    if near_mae is not None and far_mae is not None
                    else None
                ),
                "diagnostic_only": True,
            }
        )
    return result


def subject_angle_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (row["subject_id"], row["exercise"], row["view"], row["angle_name"])
        ].append(row)
    result = []
    for (subject, exercise, view, angle_name), group in sorted(grouped.items()):
        valid = [row for row in group if row["mae_deg"] is not None]
        result.append(
            {
                "subject_id": subject,
                "exercise": exercise,
                "view": view,
                "angle_name": angle_name,
                "clip_count": len(group),
                "scored_clip_count": len(valid),
                "mean_clip_mae_deg": round(
                    float(np.mean([row["mae_deg"] for row in valid])), 6
                )
                if valid
                else None,
                "mean_bias_deg": round(
                    float(np.mean([row["bias_deg"] for row in valid])), 6
                )
                if valid
                else None,
                "mean_p90_abs_error_deg": round(
                    float(np.mean([row["p90_abs_error_deg"] for row in valid])), 6
                )
                if valid
                else None,
            }
        )
    return result


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    clips = [
        clip
        for clip in manifest["clips"]
        if clip.get("has_gt")
        and not clip.get("smoke_only")
        and (args.split == "all" or clip.get("split") == args.split)
    ]
    if not clips:
        raise SystemExit("no train GT clips selected")

    angle_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for clip in clips:
        clip_rows, quality = evaluate_clip(clip, args.analysis_dir.resolve())
        angle_rows.extend(clip_rows)
        quality_rows.append(quality)

    groups_all = split_summary(angle_rows, "all", args.seed, args.bootstrap_samples)
    groups_dev = split_summary(angle_rows, "dev", args.seed, args.bootstrap_samples)
    groups_holdout = split_summary(
        angle_rows, "holdout", args.seed, args.bootstrap_samples
    )
    selected = selected_signal_summary(angle_rows, args.seed, args.bootstrap_samples)
    summary = {
        "schema_version": "task1-5b-pose-evaluation-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "analysis_dir": str(args.analysis_dir.resolve()),
        "split_filter": args.split,
        "bootstrap": {
            "unit": "subject-action",
            "samples": args.bootstrap_samples,
            "seed": args.seed,
            "ci": 0.95,
        },
        "counts": {
            "expected_clips": len(clips),
            "analysis_found": sum(row["analysis_status"] != "missing" for row in quality_rows),
            "angle_rows": len(angle_rows),
            "scored_angle_rows": sum(row["mae_deg"] is not None for row in angle_rows),
            "pose_status_counts": dict(Counter(row["analysis_status"] for row in quality_rows)),
            "quality_status_counts": dict(
                Counter(row["quality_status"] for row in quality_rows)
            ),
        },
        "pose_quality_by_exercise_view": quality_summary(quality_rows),
        "signal_selection": signal_selection_counts(quality_rows),
        "angle_groups": {
            "all": groups_all,
            "dev": groups_dev,
            "holdout": groups_holdout,
        },
        "selected_signal": selected,
        "limitations": [
            "Fit3D squat is a barbell back squat; it is not the T1-4 bodyweight squat domain.",
            "Fit3D squat has no true side view; oblique results cannot enable squat_side metrics.",
            "rep_ann frames are near extension and are not used as bottom ground truth here.",
            "Angle MAE is computed on directly frame-aligned video and 3D annotation frames.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "pose_angles.csv", angle_rows)
    write_csv(args.output_dir / "pose_quality.csv", quality_rows)
    write_csv(
        args.output_dir / "subject_angle_summary.csv",
        subject_angle_summary(angle_rows),
    )
    write_csv(
        args.output_dir / "selected_signal_summary.csv",
        flatten_selected_summary(selected),
    )
    write_csv(
        args.output_dir / "camera_selected_summary.csv",
        camera_selected_summary(angle_rows),
    )
    write_csv(
        args.output_dir / "near_far_side_diagnostic.csv",
        near_far_side_diagnostic(angle_rows),
    )
    write_json(args.output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "counts": summary["counts"],
                "selected_signal": [
                    {
                        "exercise": item["exercise"],
                        "view": item["view"],
                        "mae": item["gate_data"]["cluster_mean_mae_deg"],
                        "gate": item["pose_gate"],
                    }
                    for item in selected
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
