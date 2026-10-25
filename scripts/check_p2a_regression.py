#!/usr/bin/env python3
"""Validate the frozen P2-A default against the p2a golden file."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = ROOT / "configs" / "p2a_golden.json"
DEFAULT_SEGMENTATION = ROOT / "results" / "segmentation_p2a"
DEFAULT_REFERENCE = ROOT / "results" / "segmentation_p1d"
FLOAT_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--segmentation-dir", type=Path, default=DEFAULT_SEGMENTATION)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def values_equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) <= FLOAT_TOLERANCE
    return expected == actual


def add_difference(
    differences: list[dict[str, Any]],
    scope: str,
    key: str,
    expected: Any,
    actual: Any,
) -> None:
    differences.append(
        {
            "scope": scope,
            "key": key,
            "expected": expected,
            "actual": actual,
        }
    )


def boundaries(path: Path) -> dict[tuple[str, str, str], list[tuple[int, int]]]:
    grouped: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    for row in read_csv(path):
        key = (row["video_id"], row["action"], row["method"])
        grouped[key].append((int(row["start_frame"]), int(row["end_frame"])))
    for values in grouped.values():
        values.sort()
    return dict(grouped)


def segment_counts(
    grouped: dict[tuple[str, str, str], list[tuple[int, int]]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"aifit": 0, "peak": 0})
    for (video_id, action, method), values in grouped.items():
        counts[f"{video_id}/{action}"][method] = len(values)
    return dict(counts)


def parse_evidence(row: dict[str, str]) -> dict[str, Any]:
    value = row["evidence"]
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def blind_summary(path: Path) -> dict[str, dict[str, str]]:
    return {
        f"{float(row['iou_threshold']):g}": row
        for row in read_csv(path)
        if row["method"] == "aifit" and row["rep_unit"] == "all"
    }


def main() -> int:
    args = parse_args()
    golden = json.loads(args.golden.read_text())
    segmentation_dir = args.segmentation_dir.resolve()
    reference_dir = args.reference_dir.resolve()
    actual = boundaries(segmentation_dir / "reps.csv")
    expected = boundaries(reference_dir / "reps.csv")
    differences: list[dict[str, Any]] = []

    run_payload = json.loads((segmentation_dir / "run.json").read_text())
    for key, expected_value in golden["parameters"].items():
        actual_value = run_payload.get("parameters", {}).get(key)
        if not values_equal(expected_value, actual_value):
            add_difference(differences, "parameters", key, expected_value, actual_value)

    actual_aifit = sum(len(values) for key, values in actual.items() if key[2] == "aifit")
    actual_peak = sum(len(values) for key, values in actual.items() if key[2] == "peak")
    if actual_aifit != golden["expected_aifit_reps"]:
        add_difference(
            differences,
            "totals",
            "aifit_reps",
            golden["expected_aifit_reps"],
            actual_aifit,
        )
    if actual_peak != golden["expected_peak_reps"]:
        add_difference(
            differences,
            "totals",
            "peak_reps",
            golden["expected_peak_reps"],
            actual_peak,
        )

    actual_counts = segment_counts(actual)
    expected_segments = {
        tuple(key.split("/", 1)) for key in golden["segment_counts"]
    }
    actual_segments = {(video_id, action) for video_id, action, _method in actual}
    extra_segments = sorted("/".join(key) for key in actual_segments - expected_segments)
    if extra_segments:
        add_difference(differences, "segment_counts", "unexpected_segments", [], extra_segments)
    for key, expected_counts in golden["segment_counts"].items():
        actual_segment_counts = actual_counts.get(key, {"aifit": 0, "peak": 0})
        for method, expected_value in expected_counts.items():
            actual_value = actual_segment_counts.get(method, 0)
            if actual_value != expected_value:
                add_difference(
                    differences,
                    "segment_counts",
                    f"{key}/{method}",
                    expected_value,
                    actual_value,
                )

    target = golden["target"]
    target_key = (target["video_id"], target["action"], "aifit")
    if target_key not in expected or target_key not in actual:
        differences.append(
            {"scope": "target", "key": "boundaries", "reason": "missing target segment"}
        )
    else:
        expected_target = expected[target_key]
        actual_target = actual[target_key]
        if len(expected_target) != target["reference_reps"]:
            add_difference(
                differences,
                "target",
                "reference_reps",
                target["reference_reps"],
                len(expected_target),
            )
        if len(actual_target) != target["expected_reps"]:
            add_difference(
                differences,
                "target",
                "actual_reps",
                target["expected_reps"],
                len(actual_target),
            )
        if actual_target != expected_target[:-1]:
            differences.append(
                {
                    "scope": "target",
                    "key": "boundaries",
                    "reason": "P2-A must only remove the final reference rep",
                }
            )

        rejected_rows = [
            row
            for row in read_csv(segmentation_dir / "rejected_reps.csv")
            if row["video_id"] == target["video_id"]
            and row["action"] == target["action"]
            and row["method"] == "aifit"
            and row["reason"] == target["reject_reason"]
        ]
        if len(rejected_rows) != 1:
            add_difference(
                differences,
                "target_rejection",
                "rows",
                1,
                len(rejected_rows),
            )
        else:
            rejected = rejected_rows[0]
            removed = expected_target[-1]
            if (int(rejected["start_frame"]), int(rejected["end_frame"])) != removed:
                add_difference(
                    differences,
                    "target_rejection",
                    "interval",
                    list(removed),
                    [int(rejected["start_frame"]), int(rejected["end_frame"])],
                )
            evidence = parse_evidence(rejected)
            if int(evidence.get("support_count", 0)) < target["minimum_support_count"]:
                add_difference(
                    differences,
                    "target_rejection",
                    "support_count",
                    target["minimum_support_count"],
                    evidence.get("support_count"),
                )
            if not evidence.get("missing_phase_apex"):
                add_difference(
                    differences,
                    "target_rejection",
                    "missing_phase_apex",
                    True,
                    evidence.get("missing_phase_apex"),
                )
            if not evidence.get("low_motion_evidence"):
                add_difference(
                    differences,
                    "target_rejection",
                    "low_motion_evidence",
                    True,
                    evidence.get("low_motion_evidence"),
                )

    unchanged = golden["unchanged_air_jump"]
    unchanged_key = (unchanged["video_id"], unchanged["action"], "aifit")
    actual_unchanged = len(actual.get(unchanged_key, []))
    if actual_unchanged != unchanged["expected_reps"]:
        add_difference(
            differences,
            "unchanged_air_jump",
            "/".join(unchanged_key),
            unchanged["expected_reps"],
            actual_unchanged,
        )

    summary_path = segmentation_dir / "annotation_blind_eval" / "blind_eval_summary.csv"
    if not summary_path.exists():
        differences.append({"scope": "blind_eval", "key": "summary", "reason": "missing"})
    else:
        summary = blind_summary(summary_path)
        for threshold, expected_metrics in golden["blind_eval"].items():
            row = summary.get(threshold)
            if row is None:
                add_difference(
                    differences,
                    "blind_eval",
                    f"IoU={threshold}",
                    "present",
                    "missing",
                )
                continue
            for field, expected_value in expected_metrics.items():
                try:
                    actual_value: Any = float(row[field])
                except (KeyError, ValueError):
                    actual_value = row.get(field)
                if not values_equal(expected_value, actual_value):
                    add_difference(
                        differences,
                        "blind_eval",
                        f"IoU={threshold}/{field}",
                        expected_value,
                        actual_value,
                    )

    walking_path = segmentation_dir / "p1b1_walking_lunges_eval.json"
    if not walking_path.exists():
        differences.append({"scope": "walking_lunge", "key": "evaluation", "reason": "missing"})
    else:
        walking = json.loads(walking_path.read_text())
        acceptance = walking.get("acceptance", {})
        expected_walking = golden["walking_lunge_stride"]
        if acceptance.get("status") != "pass":
            add_difference(
                differences,
                "walking_lunge",
                "status",
                "pass",
                acceptance.get("status"),
            )
        for field in ("gt_reps", "pred_reps", "matched", "f1"):
            actual_value = acceptance.get("aifit_stride_combined", {}).get(field)
            if not values_equal(expected_walking[field], actual_value):
                add_difference(
                    differences,
                    "walking_lunge",
                    field,
                    expected_walking[field],
                    actual_value,
                )

    report = {
        "status": "pass" if not differences else "fail",
        "golden": str(args.golden.resolve()),
        "segmentation_dir": str(segmentation_dir),
        "reference_dir": str(reference_dir),
        "actual_aifit_reps": actual_aifit,
        "actual_peak_reps": actual_peak,
        "differences": differences,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
