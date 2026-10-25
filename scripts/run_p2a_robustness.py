#!/usr/bin/env python3
"""Run P2-A smoothing ablation and deterministic noise-robustness sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "segmentation_p2a_robustness"
DEFAULT_BASELINE_DIR = ROOT / "results" / "segmentation_p2a"
DEFAULT_ANGLES_DIR = ROOT / "outputs" / "angles"
DEFAULT_LANDMARKS_DIR = ROOT / "outputs" / "pose_landmarker_full"
DEFAULT_BLIND_DIR = ROOT / "results" / "segmentation" / "annotation_blind"
DEFAULT_P1B_ANNOTATION_DIR = ROOT / "results" / "segmentation_p1b" / "annotation"
DEFAULT_REP_METADATA = ROOT / "configs" / "rep_metadata.json"

SEGMENT_SCRIPT = ROOT / "scripts" / "segment_reps.py"
BLIND_EVALUATION_SCRIPT = ROOT / "scripts" / "evaluate_blind_annotations.py"
P1B_EVALUATION_SCRIPT = ROOT / "scripts" / "evaluate_p1b1_rep_units.py"
ANGLE_VOCABULARY = ROOT / "outputs" / "angles" / "angle_vocabulary.csv"

ANNOTATION_FILES = (
    "blind_reps.csv",
    "blind_segments.csv",
    "blind_exclusions.csv",
    "second_pass_sample.csv",
)
BLIND_OUTPUT_FILES = (
    "blind_eval.csv",
    "blind_eval.json",
    "blind_eval_summary.csv",
    "blind_eval_by_rep_unit.csv",
    "blind_reps.csv",
    "blind_segmentation_errors.csv",
    "blind_false_positives.csv",
    "blind_validation.csv",
    "blind_segments.csv",
    "blind_exclusions.csv",
)
TARGET = ("0218", "air_jump_rope")
UNCHANGED_AIR_JUMP = ("0173", "air_jump_rope")
TAIL_REJECTION_REASON = "air_jump_rope_transition_tail"
LANDMARK_PERTURB_VIDEOS = {"0173", "0218"}
NOSE_INDEX = 0
SMOOTHING_CONFIGS = (
    ("none", 0, 2),
    ("w5p2", 5, 2),
    ("w7p2", 7, 2),
    ("w9p2", 9, 2),
    ("w11p2", 11, 2),
    ("w7p3", 7, 3),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--angles-dir", type=Path, default=DEFAULT_ANGLES_DIR)
    parser.add_argument("--landmarks-dir", type=Path, default=DEFAULT_LANDMARKS_DIR)
    parser.add_argument("--blind-dir", type=Path, default=DEFAULT_BLIND_DIR)
    parser.add_argument(
        "--p1b-annotation-dir", type=Path, default=DEFAULT_P1B_ANNOTATION_DIR
    )
    parser.add_argument("--rep-metadata", type=Path, default=DEFAULT_REP_METADATA)
    parser.add_argument("--angle-sigmas", nargs="*", type=float, default=(0.0, 1.0, 2.0, 5.0, 10.0))
    parser.add_argument("--angle-trials", type=int, default=3)
    parser.add_argument(
        "--landmark-sigmas", nargs="*", type=float, default=(0.002, 0.005, 0.01, 0.02)
    )
    parser.add_argument("--landmark-trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--skip-smoothing", action="store_true")
    parser.add_argument("--skip-angle-noise", action="store_true")
    parser.add_argument("--skip-landmark-noise", action="store_true")
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse an existing segmentation run when reps.csv and run.json already exist.",
    )
    args = parser.parse_args()
    if args.angle_trials < 1 or args.landmark_trials < 1:
        parser.error("trial counts must be positive")
    if any(sigma < 0 for sigma in args.angle_sigmas):
        parser.error("angle sigmas must be non-negative")
    if any(sigma < 0 for sigma in args.landmark_sigmas):
        parser.error("landmark sigmas must be non-negative")
    return args


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def run_command(
    command: list[str],
    label: str,
    log_path: Path,
    allowed_codes: tuple[int, ...] = (0,),
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout + ("\n--- stderr ---\n" + result.stderr if result.stderr else ""))
    if result.returncode not in allowed_codes:
        raise RuntimeError(
            f"{label} failed with exit code {result.returncode}; see {log_path}\n"
            f"{result.stderr[-2000:]}"
        )
    return result


def boundaries(path: Path) -> dict[tuple[str, str, str], list[tuple[int, int]]]:
    grouped: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    for row in read_csv(path):
        key = (row["video_id"], row["action"], row["method"])
        grouped[key].append((int(row["start_frame"]), int(row["end_frame"])))
    for values in grouped.values():
        values.sort()
    return dict(grouped)


def interval_iou(first: tuple[int, int] | None, second: tuple[int, int] | None) -> float:
    if first is None or second is None:
        return 0.0
    intersection = min(first[1], second[1]) - max(first[0], second[0]) + 1
    if intersection <= 0:
        return 0.0
    union = max(first[1], second[1]) - min(first[0], second[0]) + 1
    return float(intersection / union)


def angle_columns() -> list[str]:
    return [
        row["name"]
        for row in read_csv(ANGLE_VOCABULARY)
        if row.get("unit", "").strip().lower() == "deg"
    ]


def write_noisy_angles(
    source_dir: Path,
    destination: Path,
    sigma: float,
    rng: np.random.Generator,
    columns: Iterable[str],
) -> None:
    angular = set(columns)
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_dir.glob("*/angles_wide.csv")):
        target_dir = destination / source.parent.name
        target_dir.mkdir(parents=True, exist_ok=True)
        with source.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
        with (target_dir / "angles_wide.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                output: dict[str, str] = {}
                for name in fieldnames:
                    value = row[name]
                    if name not in angular:
                        output[name] = value
                        continue
                    number = float(value)
                    if math.isfinite(number):
                        number += float(rng.normal(0.0, sigma))
                        number = min(180.0, max(0.0, number))
                    output[name] = f"{number:.6f}"
                writer.writerow(output)


def write_noisy_landmarks(
    source_dir: Path,
    destination: Path,
    sigma: float,
    rng: np.random.Generator,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_dir.iterdir()):
        if not source.is_file():
            continue
        video_id = source.name.split("_", 1)[0]
        is_target_landmarks = source.name.endswith("_landmarks.json") and video_id in LANDMARK_PERTURB_VIDEOS
        target = destination / source.name
        if not is_target_landmarks:
            os.symlink(source.resolve(), target)
            continue
        records = json.loads(source.read_text())
        for record in records:
            landmarks = record.get("landmarks_world")
            if not record.get("detected") or not landmarks or NOSE_INDEX >= len(landmarks):
                continue
            point = landmarks[NOSE_INDEX]
            if not isinstance(point, list) or len(point) < 3:
                continue
            landmarks[NOSE_INDEX] = [
                float(point[axis]) + float(rng.normal(0.0, sigma)) for axis in range(3)
            ]
        target.write_text(json.dumps(records, separators=(",", ":")) + "\n")


def run_segmentation(
    segmentation_dir: Path,
    angles_dir: Path,
    landmarks_dir: Path,
    smooth_window: int,
    smooth_polyorder: int,
    args: argparse.Namespace,
) -> None:
    if args.reuse and (segmentation_dir / "reps.csv").exists() and (segmentation_dir / "run.json").exists():
        return
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SEGMENT_SCRIPT),
        "--angles-dir",
        str(angles_dir),
        "--landmarks-dir",
        str(landmarks_dir),
        "--output-dir",
        str(segmentation_dir),
        "--no-plots",
        "--angle-scoring",
        "legacy",
        "--p2a-tail-filter",
        "guarded",
        "--smooth-window",
        str(smooth_window),
        "--smooth-polyorder",
        str(smooth_polyorder),
    ]
    run_command(command, "segmentation", segmentation_dir / "segmentation.stdout.txt")


def evaluate_segmentation(
    segmentation_dir: Path,
    args: argparse.Namespace,
) -> None:
    with tempfile.TemporaryDirectory(prefix="vls-p2a-blind-") as temporary:
        blind_eval_dir = Path(temporary)
        for name in ANNOTATION_FILES:
            shutil.copy2(args.blind_dir / name, blind_eval_dir / name)
        result = run_command(
            [
                sys.executable,
                str(BLIND_EVALUATION_SCRIPT),
                "--blind-dir",
                str(blind_eval_dir),
                "--reps",
                str(segmentation_dir / "reps.csv"),
                "--segments",
                str(segmentation_dir / "action_segments.csv"),
                "--fps",
                "30",
            ],
            "blind evaluation",
            segmentation_dir / "blind_eval.stdout.txt",
            allowed_codes=(0, 1),
        )
        destination = segmentation_dir / "annotation_blind_eval"
        destination.mkdir(parents=True, exist_ok=True)
        for name in BLIND_OUTPUT_FILES:
            source = blind_eval_dir / name
            if source.exists():
                shutil.copy2(source, destination / name)
        if result.returncode not in (0, 1):
            raise RuntimeError(f"blind evaluation failed for {segmentation_dir}")

    run_command(
        [
            sys.executable,
            str(P1B_EVALUATION_SCRIPT),
            "--segmentation-dir",
            str(segmentation_dir),
            "--p1b-annotation-dir",
            str(args.p1b_annotation_dir),
            "--existing-annotation-dir",
            str(args.blind_dir),
            "--rep-metadata",
            str(args.rep_metadata),
            "--output-dir",
            str(segmentation_dir),
        ],
        "walking-lunge evaluation",
        segmentation_dir / "p1b1_eval.stdout.txt",
        allowed_codes=(0, 1),
    )


def read_params(segmentation_dir: Path) -> dict[str, Any]:
    return json.loads((segmentation_dir / "run.json").read_text()).get("parameters", {})


def blind_metrics(segmentation_dir: Path) -> dict[str, dict[str, Any]]:
    path = segmentation_dir / "annotation_blind_eval" / "blind_eval_summary.csv"
    if not path.exists():
        return {}
    metrics: dict[str, dict[str, Any]] = {}
    for row in read_csv(path):
        if row["method"] != "aifit" or row["rep_unit"] != "all":
            continue
        threshold = f"{float(row['iou_threshold']):g}"
        metrics[threshold] = {
            key: float_or_none(value) for key, value in row.items() if key not in {"method", "rep_unit"}
        }
    return metrics


def walking_metrics(segmentation_dir: Path) -> dict[str, Any]:
    path = segmentation_dir / "p1b1_walking_lunges_eval.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    acceptance = payload.get("acceptance", {})
    stride = acceptance.get("aifit_stride_combined", {})
    return {
        "status": acceptance.get("status"),
        "gt_reps": stride.get("gt_reps"),
        "pred_reps": stride.get("pred_reps"),
        "matched": stride.get("matched"),
        "f1": stride.get("f1"),
    }


def target_rejection(segmentation_dir: Path) -> dict[str, Any] | None:
    path = segmentation_dir / "rejected_reps.csv"
    if not path.exists():
        return None
    for row in read_csv(path):
        if (
            row["video_id"] == TARGET[0]
            and row["action"] == TARGET[1]
            and row["method"] == "aifit"
            and row["reason"] == TAIL_REJECTION_REASON
        ):
            return row
    return None


def extract_row(
    kind: str,
    name: str,
    sigma: float | None,
    sigma_unit: str,
    trial: int,
    seed: int,
    segmentation_dir: Path,
    baseline: dict[tuple[str, str, str], list[tuple[int, int]]],
    baseline_removed: tuple[int, int] | None,
) -> dict[str, Any]:
    params = read_params(segmentation_dir)
    actual = boundaries(segmentation_dir / "reps.csv")
    target_key = (TARGET[0], TARGET[1], "aifit")
    unchanged_key = (UNCHANGED_AIR_JUMP[0], UNCHANGED_AIR_JUMP[1], "aifit")
    rejection = target_rejection(segmentation_dir)
    rejection_interval = None
    if rejection is not None:
        rejection_interval = (int(rejection["start_frame"]), int(rejection["end_frame"]))
    target_count = len(actual.get(target_key, []))
    expected_target_count = len(baseline.get(target_key, []))
    tail_overlap_iou = interval_iou(rejection_interval, baseline_removed)
    blind = blind_metrics(segmentation_dir)
    walking = walking_metrics(segmentation_dir)

    changed_keys = [
        "/".join(key)
        for key in sorted(set(actual) | set(baseline))
        if actual.get(key) != baseline.get(key)
    ]
    row: dict[str, Any] = {
        "kind": kind,
        "name": name,
        "sigma": sigma,
        "sigma_unit": sigma_unit,
        "trial": trial,
        "seed": seed,
        "smooth_window": params.get("smooth_window"),
        "smooth_polyorder": params.get("smooth_polyorder"),
        "angle_scoring": params.get("angle_scoring"),
        "p2a_tail_filter": params.get("p2a_tail_filter"),
        "segmentation_dir": str(segmentation_dir),
        "aifit_reps": sum(len(values) for key, values in actual.items() if key[2] == "aifit"),
        "peak_reps": sum(len(values) for key, values in actual.items() if key[2] == "peak"),
        "aifit_reps_delta_vs_baseline": sum(
            len(values) for key, values in actual.items() if key[2] == "aifit"
        )
        - sum(len(values) for key, values in baseline.items() if key[2] == "aifit"),
        "target_0218_air_jump_rope_reps": target_count,
        "unchanged_0173_air_jump_rope_reps": len(actual.get(unchanged_key, [])),
        "p2a_count_ok": int(target_count == expected_target_count),
        "p2a_rejection_rows": int(rejection is not None),
        "p2a_rejected_interval_correct": int(
            rejection_interval is not None and rejection_interval == baseline_removed
        ),
        "p2a_tail_overlap_iou": tail_overlap_iou,
        "p2a_success": int(
            target_count == expected_target_count
            and rejection_interval is not None
            and tail_overlap_iou >= 0.80
        ),
        "changed_segment_methods": len(changed_keys),
        "changed_segments": ";".join(changed_keys),
        "walking_lunge_status": walking.get("status"),
        "walking_lunge_stride_f1": walking.get("f1"),
        "walking_lunge_stride_pred_reps": walking.get("pred_reps"),
    }
    for threshold in ("0.3", "0.5"):
        metrics = blind.get(threshold, {})
        prefix = f"blind_{threshold.replace('.', '')}"
        row[f"{prefix}_pred_reps"] = metrics.get("pred_reps")
        row[f"{prefix}_matched"] = metrics.get("matched")
        row[f"{prefix}_false_positives"] = metrics.get("false_positives")
        row[f"{prefix}_false_negatives"] = metrics.get("false_negatives")
        row[f"{prefix}_micro_f1"] = metrics.get("micro_f1")
        row[f"{prefix}_micro_iou"] = metrics.get("micro_iou")
        row[f"{prefix}_mean_abs_start_error_frames"] = metrics.get(
            "mean_abs_start_error_frames"
        )
        row[f"{prefix}_mean_abs_end_error_frames"] = metrics.get("mean_abs_end_error_frames")
        row[f"{prefix}_p95_boundary_error_frames"] = metrics.get(
            "p95_boundary_error_frames"
        )
    return row


def run_config(
    kind: str,
    name: str,
    sigma: float | None,
    sigma_unit: str,
    trial: int,
    segmentation_dir: Path,
    args: argparse.Namespace,
    angles_dir: Path,
    landmarks_dir: Path,
    smooth_window: int,
    smooth_polyorder: int,
    baseline: dict[tuple[str, str, str], list[tuple[int, int]]],
    baseline_removed: tuple[int, int] | None,
) -> dict[str, Any]:
    config = {
        "kind": kind,
        "name": name,
        "sigma": sigma,
        "sigma_unit": sigma_unit,
        "trial": trial,
        "seed": args.seed,
        "smooth_window": smooth_window,
        "smooth_polyorder": smooth_polyorder,
        "angle_scoring": "legacy",
        "p2a_tail_filter": "guarded",
    }
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    (segmentation_dir / "robustness_config.json").write_text(json.dumps(config, indent=2) + "\n")
    run_segmentation(
        segmentation_dir,
        angles_dir,
        landmarks_dir,
        smooth_window,
        smooth_polyorder,
        args,
    )
    evaluate_segmentation(segmentation_dir, args)
    return extract_row(
        kind,
        name,
        sigma,
        sigma_unit,
        trial,
        args.seed,
        segmentation_dir,
        baseline,
        baseline_removed,
    )


def run_smoothing(
    args: argparse.Namespace,
    baseline: dict[tuple[str, str, str], list[tuple[int, int]]],
    baseline_removed: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = args.output_root / "smoothing"
    for name, window, polyorder in SMOOTHING_CONFIGS:
        print(f"[smoothing] {name} window={window} polyorder={polyorder}", flush=True)
        rows.append(
            run_config(
                "smoothing",
                name,
                None,
                "",
                1,
                root / name,
                args,
                args.angles_dir,
                args.landmarks_dir,
                window,
                polyorder,
                baseline,
                baseline_removed,
            )
        )
    return rows


def run_angle_noise(
    args: argparse.Namespace,
    baseline: dict[tuple[str, str, str], list[tuple[int, int]]],
    baseline_removed: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = args.output_root / "angle_noise"
    columns = angle_columns()
    for sigma in args.angle_sigmas:
        trials = 1 if sigma == 0 else args.angle_trials
        for trial in range(1, trials + 1):
            print(f"[angle-noise] sigma={sigma:g} deg trial={trial}", flush=True)
            segmentation_dir = root / f"sigma_{sigma:g}deg" / f"trial_{trial}"
            if sigma == 0:
                rows.append(
                    run_config(
                    "angle_noise",
                    f"sigma_{sigma:g}deg",
                    sigma,
                    "deg",
                    trial,
                    segmentation_dir,
                    args,
                    args.angles_dir,
                    args.landmarks_dir,
                    7,
                    2,
                    baseline,
                    baseline_removed,
                    )
                )
                continue
            with tempfile.TemporaryDirectory(prefix="vls-p2a-angle-noise-") as temporary:
                noisy_dir = Path(temporary) / "angles"
                rng = np.random.default_rng(args.seed + trial * 1009 + int(round(sigma * 1000)))
                write_noisy_angles(args.angles_dir, noisy_dir, sigma, rng, columns)
                rows.append(
                    run_config(
                        "angle_noise",
                        f"sigma_{sigma:g}deg",
                        sigma,
                        "deg",
                        trial,
                        segmentation_dir,
                        args,
                        noisy_dir,
                        args.landmarks_dir,
                        7,
                        2,
                        baseline,
                        baseline_removed,
                    )
                )
    return rows


def run_landmark_noise(
    args: argparse.Namespace,
    baseline: dict[tuple[str, str, str], list[tuple[int, int]]],
    baseline_removed: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = args.output_root / "landmark_noise"
    for sigma in args.landmark_sigmas:
        for trial in range(1, args.landmark_trials + 1):
            print(f"[landmark-noise] sigma={sigma:g} m trial={trial}", flush=True)
            segmentation_dir = root / f"sigma_{sigma:g}m" / f"trial_{trial}"
            with tempfile.TemporaryDirectory(prefix="vls-p2a-landmark-noise-") as temporary:
                noisy_dir = Path(temporary) / "landmarks"
                rng = np.random.default_rng(
                    args.seed + 500000 + trial * 1013 + int(round(sigma * 100000))
                )
                write_noisy_landmarks(args.landmarks_dir, noisy_dir, sigma, rng)
                rows.append(
                    run_config(
                        "landmark_noise",
                        f"sigma_{sigma:g}m",
                        sigma,
                        "m",
                        trial,
                        segmentation_dir,
                        args,
                        args.angles_dir,
                        noisy_dir,
                        7,
                        2,
                        baseline,
                        baseline_removed,
                    )
                )
    return rows


def summarize_noise(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sigma"])].append(row)
    metrics = (
        "aifit_reps",
        "peak_reps",
        "blind_03_micro_f1",
        "blind_03_micro_iou",
        "blind_03_mean_abs_start_error_frames",
        "blind_03_mean_abs_end_error_frames",
        "blind_03_p95_boundary_error_frames",
        "blind_05_micro_f1",
        "blind_05_micro_iou",
        "walking_lunge_stride_f1",
    )
    summary: dict[str, dict[str, Any]] = {}
    for sigma, values in sorted(grouped.items(), key=lambda item: float(item[0])):
        item: dict[str, Any] = {
            "sigma": float(sigma),
            "sigma_unit": values[0]["sigma_unit"],
            "trials": len(values),
            "p2a_success_rate": float(np.mean([row["p2a_success"] for row in values])),
            "p2a_count_ok_rate": float(np.mean([row["p2a_count_ok"] for row in values])),
            "rejected_interval_correct_rate": float(
                np.mean([row["p2a_rejected_interval_correct"] for row in values])
            ),
            "mean_changed_segment_methods": float(
                np.mean([row["changed_segment_methods"] for row in values])
            ),
        }
        for metric in metrics:
            numbers = [float_or_none(row.get(metric)) for row in values]
            numbers = [number for number in numbers if number is not None]
            if numbers:
                item[metric] = {
                    "mean": float(np.mean(numbers)),
                    "std": float(np.std(numbers)),
                    "min": float(np.min(numbers)),
                    "max": float(np.max(numbers)),
                }
        summary[sigma] = item
    return summary


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.baseline_dir = args.baseline_dir.resolve()
    args.angles_dir = args.angles_dir.resolve()
    args.landmarks_dir = args.landmarks_dir.resolve()
    args.blind_dir = args.blind_dir.resolve()
    args.p1b_annotation_dir = args.p1b_annotation_dir.resolve()
    args.rep_metadata = args.rep_metadata.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    baseline = boundaries(args.baseline_dir / "reps.csv")
    baseline_rejection = target_rejection(args.baseline_dir)
    baseline_removed = None
    if baseline_rejection is not None:
        baseline_removed = (
            int(baseline_rejection["start_frame"]),
            int(baseline_rejection["end_frame"]),
        )

    smoothing_rows: list[dict[str, Any]] = []
    angle_rows: list[dict[str, Any]] = []
    landmark_rows: list[dict[str, Any]] = []
    if not args.skip_smoothing:
        smoothing_rows = run_smoothing(args, baseline, baseline_removed)
    if not args.skip_angle_noise:
        angle_rows = run_angle_noise(args, baseline, baseline_removed)
    if not args.skip_landmark_noise:
        landmark_rows = run_landmark_noise(args, baseline, baseline_removed)

    smoothing_fields = sorted({key for row in smoothing_rows for key in row})
    noise_rows = angle_rows + landmark_rows
    noise_fields = sorted({key for row in noise_rows for key in row})
    if smoothing_rows:
        write_csv(args.output_root / "smoothing_ablation.csv", smoothing_rows, smoothing_fields)
    if noise_rows:
        write_csv(args.output_root / "noise_robustness.csv", noise_rows, noise_fields)

    payload = {
        "task": "task1-3 P2-A robustness",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "baseline_dir": str(args.baseline_dir),
        "definitions": {
            "angle_noise": "i.i.d. Gaussian noise added to every finite deg angle cell; NaN and time columns unchanged; values clipped to [0, 180].",
            "landmark_noise": "i.i.d. Gaussian noise added to nose world coordinates for 0173 and 0218; other landmarks and all angle inputs unchanged.",
            "p2a_success": "0218/air_jump_rope has 39 AIFit reps and the rejected tail interval has IoU >= 0.80 with the frozen baseline; exact interval match is reported separately.",
            "mediapipe_scale_note": "No paired 3D joint-angle GT exists in QEVD-FIT, so the sigma sweep is a sensitivity analysis, not a calibrated MediaPipe error estimate.",
        },
        "smoothing": smoothing_rows,
        "angle_noise": angle_rows,
        "landmark_noise": landmark_rows,
        "angle_noise_summary": summarize_noise(angle_rows),
        "landmark_noise_summary": summarize_noise(landmark_rows),
    }
    (args.output_root / "robustness_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    (args.output_root / "noise_robustness.json").write_text(
        json.dumps(
            {
                **payload,
                "smoothing": [],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    if smoothing_rows:
        (args.output_root / "smoothing_ablation.json").write_text(
            json.dumps(
                {
                    "task": "task1-3 P2-A smoothing ablation",
                    "generated_at": payload["generated_at"],
                    "baseline_dir": str(args.baseline_dir),
                    "rows": smoothing_rows,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )

    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "smoothing_configs": len(smoothing_rows),
                "angle_noise_runs": len(angle_rows),
                "landmark_noise_runs": len(landmark_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
