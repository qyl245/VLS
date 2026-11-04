#!/usr/bin/env python3
"""Validate P1-D evidence scoring against frozen P0-D2 boundaries and blind metrics."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEGMENTATION = ROOT / "results" / "segmentation_p1d"
DEFAULT_REFERENCE = ROOT / "results" / "segmentation_p0d2"
DEFAULT_BLIND_DIR = ROOT / "results" / "segmentation" / "annotation_blind"
SEGMENT_SCRIPT = ROOT / "scripts" / "segment_reps.py"
BLIND_EVAL_SCRIPT = ROOT / "scripts" / "evaluate_blind_annotations.py"
P1D_EVAL_SCRIPT = ROOT / "scripts" / "evaluate_p1d_confidence.py"
ANNOTATION_FILES = (
    "blind_reps.csv",
    "blind_segments.csv",
    "blind_exclusions.csv",
    "second_pass_sample.csv",
)
FLOAT_TOLERANCE = 1e-9
TEMPLATE_TOLERANCE = 2e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-dir", type=Path, default=DEFAULT_SEGMENTATION)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--blind-dir", type=Path, default=DEFAULT_BLIND_DIR)
    parser.add_argument("--use-existing", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def run_command(command: list[str], label: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(f"{label} failed with exit code {result.returncode}\n")
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
    return result


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


def values_equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) <= FLOAT_TOLERANCE
    return expected == actual


def rep_boundaries(path: Path) -> dict[tuple[str, str, str], list[tuple[int, int]]]:
    grouped: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    for row in read_csv(path):
        key = (row["video_id"], row["action"], row["method"])
        grouped[key].append((int(row["start_frame"]), int(row["end_frame"])))
    for values in grouped.values():
        values.sort()
    return dict(grouped)


def read_summary(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    summary = {}
    for row in read_csv(path):
        if row["rep_unit"] != "all":
            continue
        summary[(row["method"], f"{float(row['iou_threshold']):g}")] = {
            key: float(value)
            for key, value in row.items()
            if key not in {"method", "iou_threshold", "rep_unit"}
        }
    return summary


def read_blind_segment_counts(path: Path, threshold: float = 0.30) -> dict[str, dict[str, int]]:
    return {
        row["segment_id"]: {
            "gt_reps": int(float(row["gt_reps"])),
            "pred_reps": int(float(row["pred_reps"])),
        }
        for row in read_csv(path)
        if row["method"] == "aifit"
        and abs(float(row["iou_threshold"]) - threshold) < 1e-9
    }


def prepare_annotation_dir(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ANNOTATION_FILES:
        shutil.copyfile(source / name, destination / name)


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "status": "pass",
        "acceptance_status": "partial",
        "segmentation_dir": str(args.segmentation_dir.resolve()),
        "reference_dir": str(args.reference_dir.resolve()),
        "differences": [],
        "warnings": [],
        "synthetic_tests": "skipped",
    }
    differences: list[dict[str, Any]] = report["differences"]
    warnings: list[str] = report["warnings"]

    if not args.skip_synthetic:
        result = run_command(
            [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests" / "test_segment_reps.py")],
            "synthetic tests",
        )
        report["synthetic_tests"] = "pass" if result.returncode == 0 else "fail"
        if result.returncode != 0:
            add_difference(differences, "synthetic_tests", "pytest", "pass", "fail")

    with tempfile.TemporaryDirectory(prefix="vls-p1d-regression-") as temporary:
        temp_root = Path(temporary)
        segmentation_dir = args.segmentation_dir.resolve()
        annotation_dir = segmentation_dir / "annotation_blind_eval"
        if not args.use_existing:
            segmentation_dir = temp_root / "segmentation"
            result = run_command(
                [
                    sys.executable,
                    str(SEGMENT_SCRIPT),
                    "--output-dir",
                    str(segmentation_dir),
                    "--no-plots",
                    "--p2a-tail-filter",
                    "off",
                ],
                "P1-D segmentation",
            )
            if result.returncode != 0:
                report["status"] = "fail"
                if args.report:
                    args.report.write_text(json.dumps(report, indent=2) + "\n")
                return 1
            annotation_dir = temp_root / "annotation_blind_eval"
            prepare_annotation_dir(args.blind_dir.resolve(), annotation_dir)
            result = run_command(
                [
                    sys.executable,
                    str(BLIND_EVAL_SCRIPT),
                    "--blind-dir",
                    str(annotation_dir),
                    "--reps",
                    str(segmentation_dir / "reps.csv"),
                    "--segments",
                    str(segmentation_dir / "action_segments.csv"),
                ],
                "P1-D blind evaluation",
            )
            if result.returncode != 0:
                add_difference(differences, "blind_evaluation", "command", 0, result.returncode)

        p1d_eval_path = segmentation_dir / "p1d_confidence_eval.json"
        result = run_command(
            [
                sys.executable,
                str(P1D_EVAL_SCRIPT),
                "--segmentation-dir",
                str(segmentation_dir),
                "--blind-dir",
                str(args.blind_dir.resolve()),
                "--output-dir",
                str(segmentation_dir),
            ],
            "P1-D confidence evaluation",
        )
        if result.returncode != 0 or not p1d_eval_path.exists():
            add_difference(differences, "confidence_evaluation", "command", 0, result.returncode)
            p1d_eval: dict[str, Any] = {}
        else:
            p1d_eval = json.loads(p1d_eval_path.read_text())

        reference_boundaries = rep_boundaries(args.reference_dir.resolve() / "reps.csv")
        actual_boundaries = rep_boundaries(segmentation_dir / "reps.csv")
        all_keys = sorted(set(reference_boundaries) | set(actual_boundaries))
        for key in all_keys:
            expected = reference_boundaries.get(key)
            actual = actual_boundaries.get(key)
            if expected != actual:
                add_difference(
                    differences,
                    "rep_boundaries",
                    "/".join(key),
                    expected,
                    actual,
                )
        report["aifit_reps"] = sum(
            len(values) for key, values in actual_boundaries.items() if key[2] == "aifit"
        )
        report["peak_reps"] = sum(
            len(values) for key, values in actual_boundaries.items() if key[2] == "peak"
        )

        reference_summary_path = (
            args.reference_dir.resolve() / "annotation_blind_eval" / "blind_eval_summary.csv"
        )
        actual_summary_path = annotation_dir / "blind_eval_summary.csv"
        if reference_summary_path.exists() and actual_summary_path.exists():
            reference_summary = read_summary(reference_summary_path)
            actual_summary = read_summary(actual_summary_path)
            metric_fields = {
                "matched",
                "false_positives",
                "false_negatives",
                "micro_f1",
                "micro_iou",
                "p95_boundary_error_frames",
                "split_events",
                "merge_events",
            }
            report["blind_metrics"] = {}
            for threshold in ("0.1", "0.3", "0.5"):
                key = ("aifit", threshold)
                expected = reference_summary.get(key, {})
                actual = actual_summary.get(key, {})
                report["blind_metrics"][threshold] = {
                    field: actual.get(field) for field in sorted(metric_fields)
                }
                for field in metric_fields:
                    if not values_equal(expected.get(field), actual.get(field)):
                        add_difference(
                            differences,
                            "blind_metrics",
                            f"IoU={threshold}/{field}",
                            expected.get(field),
                            actual.get(field),
                        )
        else:
            add_difference(
                differences,
                "blind_metrics",
                "summary_file",
                "present",
                "missing",
            )

        reference_counts = read_blind_segment_counts(
            args.reference_dir.resolve() / "annotation_blind_eval" / "blind_eval.csv"
        )
        actual_counts = read_blind_segment_counts(annotation_dir / "blind_eval.csv")
        report["blind_segment_counts"] = actual_counts
        for segment_id, expected in reference_counts.items():
            if actual_counts.get(segment_id) != expected:
                add_difference(
                    differences,
                    "blind_segment_counts",
                    segment_id,
                    expected,
                    actual_counts.get(segment_id),
                )

        if p1d_eval:
            blind_pred_reps = sum(
                values["pred_reps"] for values in actual_counts.values()
            )
            if p1d_eval.get("pred_reps") != blind_pred_reps:
                add_difference(
                    differences,
                    "confidence_evaluation",
                    "pred_reps",
                    blind_pred_reps,
                    p1d_eval.get("pred_reps"),
                )
            acceptance = p1d_eval.get("acceptance", {})
            report["confidence_acceptance"] = {
                key: value for key, value in acceptance.items() if key != "auto_delete_reason"
            }
            if acceptance.get("auto_delete_enabled") is not False:
                add_difference(
                    differences,
                    "confidence_evaluation",
                    "auto_delete_enabled",
                    False,
                    acceptance.get("auto_delete_enabled"),
                )
            primary = acceptance.get("0.3", {})
            if primary.get("explained_fp", 0) < 1:
                add_difference(
                    differences,
                    "confidence_evaluation",
                    "explained_fp_iou_0.3",
                    ">=1",
                    primary.get("explained_fp", 0),
                )
            if primary.get("best_ap_gain_vs_legacy", 0.0) <= 0.0:
                add_difference(
                    differences,
                    "confidence_evaluation",
                    "best_ap_gain_vs_legacy",
                    ">0",
                    primary.get("best_ap_gain_vs_legacy"),
                )
            if primary.get("auto_gate_recommended") is not False:
                add_difference(
                    differences,
                    "confidence_evaluation",
                    "auto_gate_recommended",
                    False,
                    primary.get("auto_gate_recommended"),
                )
            template_validation = p1d_eval.get("template_ablation_validation", {})
            irls_diff = template_validation.get("max_abs_diff", {}).get("irls", 0.0)
            report["template_validation"] = template_validation
            if irls_diff > TEMPLATE_TOLERANCE:
                add_difference(
                    differences,
                    "template_validation",
                    "irls_shape_difference",
                    f"<={TEMPLATE_TOLERANCE}",
                    irls_diff,
                )
            loso = p1d_eval.get("loso_by_threshold", {})
            report["loso"] = loso
            if any(item.get("status") != "evaluated" for item in loso.values()):
                warnings.append(
                    "LOSO calibration is not estimable because all FP labels are concentrated "
                    "in one segment; no fitted score was used for gating."
                )
            report["fixed_recall_safety"] = {
                threshold: {
                    "target_recall": values.get("target_recall"),
                    "precision": values.get("fixed_recall_precision"),
                    "tp_retained": values.get("fixed_recall_tp_retained"),
                    "positives": values.get("positives"),
                }
                for threshold, values in _fixed_recall_rows(segmentation_dir / "p1d_score_ablation.csv").items()
            }
            for threshold, values in report["fixed_recall_safety"].items():
                achieved_recall = (
                    values["tp_retained"] / values["positives"]
                    if values["positives"]
                    else 0.0
                )
                if achieved_recall + 1e-12 < values["target_recall"]:
                    add_difference(
                        differences,
                        "fixed_recall_safety",
                        f"IoU={threshold}/all_four_geomean",
                        values["target_recall"],
                        achieved_recall,
                    )

    report["difference_count"] = len(differences)
    if differences:
        report["status"] = "fail"
        report["acceptance_status"] = "fail"
    else:
        report["status"] = "pass"
        report["acceptance_status"] = "partial" if warnings else "pass_review_only"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "acceptance_status": report["acceptance_status"],
                "difference_count": report["difference_count"],
                "warnings": report["warnings"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "pass" else 1


def _fixed_recall_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = {}
    for row in read_csv(path):
        if row["score"] != "all_four_geomean":
            continue
        rows[f"{float(row['iou_threshold']):g}"] = {
            "target_recall": float(row["target_recall"]),
            "fixed_recall_precision": float(row["fixed_recall_precision"]),
            "fixed_recall_tp_retained": int(float(row["fixed_recall_tp_retained"])),
            "positives": int(float(row["positives"])),
        }
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
