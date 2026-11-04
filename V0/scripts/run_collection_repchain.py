#!/usr/bin/env python3
"""Run frozen RepChain v2 on a clip manifest and evaluate against frozen rep GT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import task1_6_chain_selection as chain_selection
import task1_6_phase_states as phase_states
import task1_6_quality as quality_gates


DEFAULT_MANIFEST = ROOT / "configs" / "collection_squat_clips.json"
DEFAULT_GT = ROOT / "results" / "collection_squat" / "annotation" / "gt_reps.csv"
DEFAULT_LABELS = ROOT / "results" / "collection_squat" / "annotation" / "gt_clip_labels.csv"
DEFAULT_ANALYSIS_DIR = ROOT / "results" / "collection_squat" / "clip_analysis"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "collection_squat" / "repchain_v2"
DEFAULT_SPECS = ROOT / "configs" / "task1_4_exercise_specs.json"
DEFAULT_SEGMENTER = ROOT / "configs" / "task1_6_segmenter.json"
DEFAULT_CHAIN = ROOT / "configs" / "task1_6_chain.json"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = phase_states.load_json(path)
    clips = payload.get("clips") if isinstance(payload, dict) else payload
    if not isinstance(clips, list):
        raise ValueError(f"manifest has no clip list: {path}")
    output = []
    seen = set()
    for clip in clips:
        clip_id = str(clip.get("clip_id") or "")
        if not clip_id or clip_id in seen:
            raise ValueError(f"invalid or duplicate clip_id in {path}: {clip_id!r}")
        seen.add(clip_id)
        action = str(clip.get("action") or clip.get("exercise") or "").lower()
        if not action:
            raise ValueError(f"clip {clip_id} has no action/exercise")
        output.append({**clip, "action": action})
    return output


def default_set(analysis: dict[str, Any]) -> dict[str, Any] | None:
    sets = analysis.get("sets") or []
    return next(
        (item for item in sets if item.get("is_default")),
        sets[0] if sets else None,
    )


def selected_event_cycles(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    selected = default_set(analysis)
    if not selected:
        return []
    set_id = selected.get("set_id")
    events = [
        event
        for event in (analysis.get("diagnostics") or {}).get(
            "selected_top_events", []
        )
        if set_id in (event.get("set_ids") or [])
    ]
    event_frames = sorted({int(event["frame"]) for event in events})
    status_by_top = {
        int((rep.get("top") or {}).get("anchor_frame")): {
            "rep_id": int(rep["rep_id"]),
            "status": str(rep.get("status", "complete")),
        }
        for rep in selected.get("reps") or []
        if (rep.get("top") or {}).get("anchor_frame") is not None
    }
    cycles = []
    for index, (left, right) in enumerate(
        zip(event_frames[:-1], event_frames[1:]), start=1
    ):
        if right <= left:
            continue
        metadata = status_by_top.get(left, {})
        cycles.append(
            {
                "rep_id": int(metadata.get("rep_id", index)),
                "start_frame": left,
                "end_frame": right - 1,
                "cycle_status": str(metadata.get("status", "unclassified")),
            }
        )
    return cycles


def complete_rep_intervals(cycles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        cycle for cycle in cycles if cycle.get("cycle_status") == "complete"
    ]


def percentile(values: list[float], q: float) -> float | None:
    return phase_states.percentile(values, q)


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return round(float(numerator) / float(denominator), 6) if denominator else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gt-reps", type=Path, default=DEFAULT_GT)
    parser.add_argument("--gt-labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    parser.add_argument("--segmenter-config", type=Path, default=DEFAULT_SEGMENTER)
    parser.add_argument("--chain-config", type=Path, default=DEFAULT_CHAIN)
    parser.add_argument(
        "--allow-legacy-fallback",
        action="store_true",
        help="Enable the frozen legacy fallback; disabled by default for a true RepChain test.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clips = load_manifest(args.manifest)
    clip_index = {clip["clip_id"]: clip for clip in clips}
    specs = phase_states.load_json(args.specs)
    segmenter_config = phase_states.load_json(args.segmenter_config)
    chain_config = chain_selection.load_chain_config(args.chain_config)
    phase_policy = segmenter_config["core"]["phase_signal"]["policy"]
    detector = segmenter_config["core"].get("detector")

    gt_rows = read_csv(args.gt_reps)
    label_rows = read_csv(args.gt_labels)
    labels = {row["clip_id"]: row for row in label_rows}
    if set(labels) != set(clip_index):
        raise ValueError("GT label clip_ids do not match the manifest")
    gt_by_clip: dict[str, list[dict[str, Any]]] = {clip_id: [] for clip_id in clip_index}
    for row in gt_rows:
        clip_id = row["clip_id"]
        if clip_id not in clip_index:
            raise ValueError(f"GT references unknown clip: {clip_id}")
        clip_gt = {
            "rep_id": int(row["rep_id"]),
            "start_frame": int(row["start_frame"]),
            "end_frame": int(row["end_frame"]),
            "bottom_frame": int(row["bottom_frame"]),
        }
        if clip_gt["start_frame"] > clip_gt["end_frame"]:
            raise ValueError(f"invalid GT interval for {clip_id}")
        gt_by_clip[clip_id].append(clip_gt)

    analysis_dir = args.output_dir / "analysis"
    gate_dir = args.output_dir / "gates"
    evaluation_dir = args.output_dir / "evaluation"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    gate_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    clip_metrics: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    predicted_unmatched: list[dict[str, Any]] = []
    pooled_start_errors: list[float] = []
    pooled_end_errors: list[float] = []
    tp30 = fp30 = fn30 = 0
    tp50 = fp50 = fn50 = 0
    event_tp30 = event_fp30 = event_fn30 = 0
    event_tp50 = event_fp50 = event_fn50 = 0
    event_start_errors: list[float] = []
    event_end_errors: list[float] = []
    status_counts: Counter[str] = Counter()
    negative_correct = 0

    for clip in clips:
        clip_id = clip["clip_id"]
        clip_dir = args.analysis_dir / clip_id
        legacy = phase_states.load_json(clip_dir / "analysis.json")
        signals = phase_states.load_angle_signals(clip_dir / "angles" / "angles_wide.csv")
        action = phase_states.action_config(clip["action"], specs)
        prior_signal = (legacy.get("segmentation") or {}).get("phase_angle")
        gate = quality_gates.from_legacy_quality(legacy)
        phase_states.write_json(gate_dir / f"{clip_id}.json", gate)

        analysis, meta = chain_selection.analyze_clip_signals(
            clip,
            signals,
            action,
            prior_signal=prior_signal,
            phase_policy=phase_policy,
            chain_config=chain_config,
            legacy_analysis=legacy,
            allow_legacy_fallback=bool(args.allow_legacy_fallback),
            quality_gate=gate,
            detector=detector,
        )
        phase_states.write_json(analysis_dir / f"{clip_id}.json", analysis)

        gt = sorted(gt_by_clip[clip_id], key=lambda item: int(item["start_frame"]))
        event_intervals = selected_event_cycles(analysis)
        pred_intervals = complete_rep_intervals(event_intervals)
        matches30, start_errors30, end_errors30 = chain_selection.match_intervals_by_iou(
            gt, pred_intervals, minimum_iou=0.30
        )
        matches50, _, _ = chain_selection.match_intervals_by_iou(
            gt, pred_intervals, minimum_iou=0.50
        )
        matched_gt30 = {gt_index for gt_index, _, _ in matches30}
        matched_pred30 = {pred_index for _, pred_index, _ in matches30}
        matched_gt50 = {gt_index for gt_index, _, _ in matches50}
        matched_pred50 = {pred_index for _, pred_index, _ in matches50}

        tp30 += len(matches30)
        fn30 += len(gt) - len(matches30)
        fp30 += len(pred_intervals) - len(matches30)
        tp50 += len(matches50)
        fn50 += len(gt) - len(matches50)
        fp50 += len(pred_intervals) - len(matches50)
        pooled_start_errors.extend(start_errors30)
        pooled_end_errors.extend(end_errors30)

        event_matches30, event_start_errors30, event_end_errors30 = (
            chain_selection.match_intervals_by_iou(
                gt, event_intervals, minimum_iou=0.30
            )
        )
        event_matches50, _, _ = chain_selection.match_intervals_by_iou(
            gt, event_intervals, minimum_iou=0.50
        )
        event_matched_gt30 = {gt_index for gt_index, _, _ in event_matches30}
        event_matched_pred30 = {pred_index for _, pred_index, _ in event_matches30}
        event_tp30 += len(event_matches30)
        event_fn30 += len(gt) - len(event_matches30)
        event_fp30 += len(event_intervals) - len(event_matches30)
        event_tp50 += len(event_matches50)
        event_fn50 += len(gt) - len(event_matches50)
        event_fp50 += len(event_intervals) - len(event_matches50)
        event_start_errors.extend(event_start_errors30)
        event_end_errors.extend(event_end_errors30)

        expected = int(labels[clip_id]["expected_complete_reps"])
        if expected != len(gt):
            raise ValueError(f"label count mismatch for {clip_id}: {expected} != {len(gt)}")
        if expected == 0 and not event_intervals:
            negative_correct += 1
        status_counts[str(analysis.get("status") or "unknown")] += 1

        match_by_gt = {
            gt_index: row for row in event_matches30 for gt_index in [row[0]]
        }
        for gt_index, gt_rep in enumerate(gt):
            matched = match_by_gt.get(gt_index)
            row = {
                "clip_id": clip_id,
                "gt_rep_id": gt_rep["rep_id"],
                "gt_start_frame": gt_rep["start_frame"],
                "gt_end_frame": gt_rep["end_frame"],
                "pred_rep_id": "",
                "pred_start_frame": "",
                "pred_end_frame": "",
                "iou": 0.0,
                "start_abs_error_frames": "",
                "end_abs_error_frames": "",
                "outcome": "fn",
                "cycle_status": "",
            }
            if matched:
                _, pred_index, iou = matched
                pred = event_intervals[pred_index]
                row.update(
                    {
                        "pred_rep_id": pred["rep_id"],
                        "pred_start_frame": pred["start_frame"],
                        "pred_end_frame": pred["end_frame"],
                        "iou": round(iou, 6),
                        "start_abs_error_frames": abs(pred["start_frame"] - gt_rep["start_frame"]),
                        "end_abs_error_frames": abs(pred["end_frame"] - gt_rep["end_frame"]),
                        "outcome": "tp",
                        "cycle_status": pred.get("cycle_status"),
                    }
                )
            match_rows.append(row)

        for pred_index, pred in enumerate(event_intervals):
            if pred_index not in event_matched_pred30:
                predicted_unmatched.append(
                    {
                        "clip_id": clip_id,
                        "pred_rep_id": pred["rep_id"],
                        "pred_start_frame": pred["start_frame"],
                        "pred_end_frame": pred["end_frame"],
                    }
                )

        clip_metrics.append(
            {
                "clip_id": clip_id,
                "label": labels[clip_id]["label"],
                "gt_rep_count": len(gt),
                "formal_complete_rep_count": len(pred_intervals),
                "formal_complete_match_iou30": len(matches30),
                "event_cycle_count": len(event_intervals),
                "event_cycle_count_abs_error": abs(len(event_intervals) - len(gt)),
                "event_tp_iou30": len(event_matches30),
                "event_fp_iou30": len(event_intervals) - len(event_matches30),
                "event_fn_iou30": len(gt) - len(event_matches30),
                "event_tp_iou50": len(event_matches50),
                "event_fp_iou50": len(event_intervals) - len(event_matches50),
                "event_fn_iou50": len(gt) - len(event_matches50),
                "event_start_abs_error_median_frames": percentile(
                    event_start_errors30, 50
                ),
                "event_end_abs_error_median_frames": percentile(
                    event_end_errors30, 50
                ),
                "status": analysis.get("status"),
                "motion_status": analysis.get("motion_status"),
                "phase_signal": (analysis.get("diagnostics") or {}).get("selected_phase_signal"),
                "set_count": len(analysis.get("sets") or []),
                "quality_status": gate.get("status"),
                "fallback_used": bool(meta.get("fallback_used")),
            }
        )

    precision30 = ratio(tp30, tp30 + fp30)
    recall30 = ratio(tp30, tp30 + fn30)
    f1_30 = (
        round(2 * precision30 * recall30 / (precision30 + recall30), 6)
        if precision30 is not None and recall30 is not None and precision30 + recall30
        else None
    )
    precision50 = ratio(tp50, tp50 + fp50)
    recall50 = ratio(tp50, tp50 + fn50)
    f1_50 = (
        round(2 * precision50 * recall50 / (precision50 + recall50), 6)
        if precision50 is not None and recall50 is not None and precision50 + recall50
        else None
    )
    event_precision30 = ratio(event_tp30, event_tp30 + event_fp30)
    event_recall30 = ratio(event_tp30, event_tp30 + event_fn30)
    event_f1_30 = (
        round(
            2
            * event_precision30
            * event_recall30
            / (event_precision30 + event_recall30),
            6,
        )
        if event_precision30 is not None
        and event_recall30 is not None
        and event_precision30 + event_recall30
        else None
    )
    event_precision50 = ratio(event_tp50, event_tp50 + event_fp50)
    event_recall50 = ratio(event_tp50, event_tp50 + event_fn50)
    event_f1_50 = (
        round(
            2
            * event_precision50
            * event_recall50
            / (event_precision50 + event_recall50),
            6,
        )
        if event_precision50 is not None
        and event_recall50 is not None
        and event_precision50 + event_recall50
        else None
    )
    count_errors = [row["event_cycle_count_abs_error"] for row in clip_metrics]
    summary = {
        "schema_version": "collection-squat-repchain-eval-v2",
        "clip_count": len(clips),
        "gt_complete_rep_count": len(gt_rows),
        "formal_complete_rep": {
            "predicted_count": sum(
                row["formal_complete_rep_count"] for row in clip_metrics
            ),
            "matched_iou30": tp30,
            "fp_iou30": fp30,
            "fn_iou30": fn30,
            "precision_iou30": precision30,
            "recall_iou30": recall30,
            "f1_iou30": f1_30,
            "boundary_start_median_abs_error_frames": percentile(
                pooled_start_errors, 50
            ),
            "boundary_end_median_abs_error_frames": percentile(
                pooled_end_errors, 50
            ),
            "boundary_p95_abs_error_frames": percentile(
                pooled_start_errors + pooled_end_errors, 95
            ),
        },
        "event_cycles": {
            "predicted_count": sum(
                row["event_cycle_count"] for row in clip_metrics
            ),
            "count_mae": round(sum(count_errors) / len(count_errors), 6),
            "count_exact_clip_count": sum(error == 0 for error in count_errors),
            "count_within_one_clip_count": sum(
                error <= 1 for error in count_errors
            ),
            "iou30": {
                "tp": event_tp30,
                "fp": event_fp30,
                "fn": event_fn30,
                "precision": event_precision30,
                "recall": event_recall30,
                "f1": event_f1_30,
            },
            "iou50": {
                "tp": event_tp50,
                "fp": event_fp50,
                "fn": event_fn50,
                "precision": event_precision50,
                "recall": event_recall50,
                "f1": event_f1_50,
            },
            "boundary_start_median_abs_error_frames": percentile(
                event_start_errors, 50
            ),
            "boundary_end_median_abs_error_frames": percentile(
                event_end_errors, 50
            ),
            "boundary_p95_abs_error_frames": percentile(
                event_start_errors + event_end_errors, 95
            ),
        },
        "negative_clip_count": sum(
            int(row["expected_complete_reps"]) == 0 for row in label_rows
        ),
        "negative_clip_correct_count": negative_correct,
        "status_counts": dict(sorted(status_counts.items())),
        "fallback_enabled": bool(args.allow_legacy_fallback),
        "inputs": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_path(args.manifest),
            "gt_reps": str(args.gt_reps.resolve()),
            "gt_reps_sha256": sha256_path(args.gt_reps),
            "gt_labels": str(args.gt_labels.resolve()),
            "gt_labels_sha256": sha256_path(args.gt_labels),
            "analysis_dir": str(args.analysis_dir.resolve()),
            "segmenter_config": str(args.segmenter_config.resolve()),
            "segmenter_config_sha256": sha256_path(args.segmenter_config),
            "chain_config": str(args.chain_config.resolve()),
            "chain_config_sha256": sha256_path(args.chain_config),
        },
        "clips": clip_metrics,
    }
    phase_states.write_json(evaluation_dir / "summary.json", summary)
    write_csv(
        evaluation_dir / "clip_metrics.csv",
        clip_metrics,
        list(clip_metrics[0]),
    )
    write_csv(
        evaluation_dir / "rep_matches.csv",
        match_rows,
        list(match_rows[0]),
    )
    write_csv(
        evaluation_dir / "predicted_unmatched.csv",
        predicted_unmatched,
        ["clip_id", "pred_rep_id", "pred_start_frame", "pred_end_frame"],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
