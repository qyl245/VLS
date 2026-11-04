#!/usr/bin/env python3
"""Summarize expanded front-view squat RepChain and v3 results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "multiview_front_squat_expanded.json"
DEFAULT_ROOT = ROOT / "results" / "multiview_front_squat_expanded"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def aggregate(clip_rows: list[dict[str, Any]], match_rows: list[dict[str, str]]) -> dict[str, Any]:
    tp = sum(row["event_tp_iou30"] for row in clip_rows)
    fp = sum(row["event_fp_iou30"] for row in clip_rows)
    fn = sum(row["event_fn_iou30"] for row in clip_rows)
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = (
        round(2 * precision * recall / (precision + recall), 6)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    clip_ids = {row["clip_id"] for row in clip_rows}
    matched = [
        row
        for row in match_rows
        if row["clip_id"] in clip_ids
        and row["outcome"] == "tp"
        and row["start_abs_error_frames"]
        and row["end_abs_error_frames"]
    ]
    start_errors = [float(row["start_abs_error_frames"]) for row in matched]
    end_errors = [float(row["end_abs_error_frames"]) for row in matched]
    boundary_errors = start_errors + end_errors
    return {
        "clip_count": len(clip_rows),
        "gt_rep_count": sum(row["gt_rep_count"] for row in clip_rows),
        "predicted_event_cycle_count": sum(row["event_cycle_count"] for row in clip_rows),
        "event_tp_iou30": tp,
        "event_fp_iou30": fp,
        "event_fn_iou30": fn,
        "event_precision_iou30": precision,
        "event_recall_iou30": recall,
        "event_f1_iou30": f1,
        "count_mae": round(
            statistics.mean(row["event_cycle_count_abs_error"] for row in clip_rows), 6
        ),
        "count_exact_clip_count": sum(
            row["event_cycle_count_abs_error"] == 0 for row in clip_rows
        ),
        "count_within_one_clip_count": sum(
            row["event_cycle_count_abs_error"] <= 1 for row in clip_rows
        ),
        "boundary_start_median_abs_error_frames": (
            statistics.median(start_errors) if start_errors else None
        ),
        "boundary_end_median_abs_error_frames": (
            statistics.median(end_errors) if end_errors else None
        ),
        "boundary_p95_abs_error_frames": (
            round(float(np.percentile(boundary_errors, 95)), 6) if boundary_errors else None
        ),
        "v3_supported_count": sum(row["v3_status"] == "ok" for row in clip_rows),
        "v3_abstained_count": sum(row["v3_abstained"] for row in clip_rows),
    }


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    clips = {clip["clip_id"]: clip for clip in manifest["clips"]}
    metric_rows = read_csv(args.results_root / "repchain_v2" / "evaluation" / "clip_metrics.csv")
    match_rows = read_csv(args.results_root / "repchain_v2" / "evaluation" / "rep_matches.csv")
    lite_payload = json.loads((args.results_root / "lite_v3" / "summary.json").read_text())
    lite = {row["clip_id"]: row for row in lite_payload["clips"]}

    enriched = []
    for row in metric_rows:
        clip_id = row["clip_id"]
        clip = clips[clip_id]
        quality = json.loads(
            (args.results_root / "clip_analysis" / clip_id / "quality.json").read_text()
        )
        checks = quality["checks"]
        v3 = lite[clip_id]
        enriched.append(
            {
                "clip_id": clip_id,
                "subject_id": clip_id.split("_")[1],
                "source_quality_label": clip["source_quality_label"],
                "gt_rep_count": int(row["gt_rep_count"]),
                "event_cycle_count": int(row["event_cycle_count"]),
                "event_cycle_count_abs_error": int(row["event_cycle_count_abs_error"]),
                "event_tp_iou30": int(row["event_tp_iou30"]),
                "event_fp_iou30": int(row["event_fp_iou30"]),
                "event_fn_iou30": int(row["event_fn_iou30"]),
                "event_start_abs_error_median_frames": (
                    float(row["event_start_abs_error_median_frames"])
                    if row["event_start_abs_error_median_frames"]
                    else None
                ),
                "event_end_abs_error_median_frames": (
                    float(row["event_end_abs_error_median_frames"])
                    if row["event_end_abs_error_median_frames"]
                    else None
                ),
                "formal_complete_rep_count": int(row["formal_complete_rep_count"]),
                "quality_status": quality["status"],
                "pose_detection_rate": checks["pose_detection_rate"],
                "q10_key_visibility": checks["q10_key_visibility"],
                "body_in_frame_rate": checks["body_in_frame_rate"],
                "candidate_rep_count": checks["candidate_rep_count"],
                "rejected_rep_count": checks["rejected_rep_count"],
                "v3_status": v3["status"],
                "v3_action_status": v3["action_status"],
                "v3_abstained": bool(v3["abstained"]),
                "v3_abstain_reason": v3["abstain_reason"],
            }
        )

    by_quality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_subject_quality: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        by_quality[row["source_quality_label"]].append(row)
        by_subject[row["subject_id"]].append(row)
        by_subject_quality[(row["subject_id"], row["source_quality_label"])].append(row)

    summary = {
        "schema_version": "multiview-front-squat-expanded-summary-v1",
        "manifest": str(args.manifest.resolve()),
        "overall": aggregate(enriched, match_rows),
        "by_source_quality": {
            key: aggregate(value, match_rows) for key, value in sorted(by_quality.items())
        },
        "by_subject": {
            key: aggregate(value, match_rows) for key, value in sorted(by_subject.items())
        },
        "clips": enriched,
    }

    output_dir = args.results_root / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "expanded_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    with (output_dir / "clip_metrics_enriched.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(enriched[0]))
        writer.writeheader()
        writer.writerows(enriched)

    group_rows = []
    for (subject, quality), values in sorted(by_subject_quality.items()):
        group_rows.append(
            {"group_type": "subject_quality", "group": subject, "quality": quality, **aggregate(values, match_rows)}
        )
    for subject, values in sorted(by_subject.items()):
        group_rows.append(
            {"group_type": "subject_all", "group": subject, "quality": "all", **aggregate(values, match_rows)}
        )
    with (output_dir / "subject_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)
    print(json.dumps({"overall": summary["overall"], "by_source_quality": summary["by_source_quality"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
