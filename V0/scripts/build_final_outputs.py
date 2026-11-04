#!/usr/bin/env python3
"""Build the standardized task1-3 V0 final output directory."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import savgol_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "results" / "segmentation_p2a"
DEFAULT_REFERENCE = ROOT / "results" / "segmentation_p1d"
DEFAULT_ROBUSTNESS = ROOT / "results" / "segmentation_p2a_robustness"
DEFAULT_OUTPUT = ROOT / "results" / "segmentation_final"
DEFAULT_ANGLES = ROOT / "outputs" / "angles"
DEFAULT_GOLDEN = ROOT / "configs" / "p2a_golden.json"
DEFAULT_REP_METADATA = ROOT / "configs" / "rep_metadata.json"
DEFAULT_PHASE_CONFIG = ROOT / "configs" / "phase_guidance.json"
SEGMENT_SCRIPT = ROOT / "scripts" / "segment_reps.py"
CHECK_SCRIPT = ROOT / "scripts" / "check_p2a_regression.py"

SEGMENTATION_FILES = (
    "reps.csv",
    "summary.csv",
    "action_segments.csv",
    "non_rep_intervals.csv",
    "rejected_reps.csv",
    "run.json",
    "p1b1_walking_lunges_eval.csv",
    "p1b1_walking_lunges_eval.json",
    "p1b1_walking_lunges_summary.csv",
    "p1b1_walking_lunges_false_positives.csv",
    "blind_eval.stdout.txt",
    "p1b1_eval.stdout.txt",
)
BLIND_FILES = (
    "blind_eval.csv",
    "blind_eval.json",
    "blind_eval_summary.csv",
    "blind_eval_by_rep_unit.csv",
    "blind_reps.csv",
    "blind_segments.csv",
    "blind_exclusions.csv",
    "blind_segmentation_errors.csv",
    "blind_false_positives.csv",
    "blind_validation.csv",
    "blind_second_pass.csv",
    "blind_second_pass_summary.csv",
    "blind_vs_first_round.csv",
    "focus.csv",
)
ROBUSTNESS_FILES = (
    "robustness_config.json",
    "reps.csv",
    "rejected_reps.csv",
    "summary.csv",
)
SMOOTHING_ORDER = ("none", "w5p2", "w7p2", "w9p2", "w11p2", "w7p3")
COLORS = {
    "aifit": "#c43c4b",
    "peak": "#1f6fb2",
    "default": "#2e7d32",
    "rejected": "#e07b39",
    "muted": "#777777",
    "grid": "#d9d9d9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--robustness-dir", type=Path, default=DEFAULT_ROBUSTNESS)
    parser.add_argument("--angles-dir", type=Path, default=DEFAULT_ANGLES)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--rep-metadata", type=Path, default=DEFAULT_REP_METADATA)
    parser.add_argument("--phase-config", type=Path, default=DEFAULT_PHASE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-qualitative",
        action="store_true",
        help="Reuse existing qualitative plots instead of regenerating them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing final directory.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def format_value(value: Any, digits: int = 4) -> str:
    number = float_or_none(value)
    if number is not None:
        return f"{number:.{digits}f}"
    return "" if value is None else str(value)


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


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def boundaries(path: Path) -> dict[tuple[str, str, str], list[tuple[int, int]]]:
    grouped: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    for row in read_csv(path):
        key = (row["video_id"], row["action"], row["method"])
        grouped[key].append((int(row["start_frame"]), int(row["end_frame"])))
    for values in grouped.values():
        values.sort()
    return dict(grouped)


def rep_counts(path: Path) -> dict[tuple[str, str], dict[str, int]]:
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in read_csv(path):
        counts[(row["video_id"], row["action"])][row["method"]] += 1
    return counts


def blind_rows(path: Path) -> dict[tuple[str, str], dict[str, float | None]]:
    rows: dict[tuple[str, str], dict[str, float | None]] = {}
    for row in read_csv(path):
        key = (row["method"], f"{float(row['iou_threshold']):g}")
        rows[key] = {name: float_or_none(value) for name, value in row.items()}
    return rows


def smoothing_rows(path: Path) -> list[dict[str, Any]]:
    by_name = {row["name"]: row for row in read_csv(path)}
    rows: list[dict[str, Any]] = []
    for name in SMOOTHING_ORDER:
        source = by_name.get(name)
        if source is None:
            continue
        rows.append(
            {
                "family": "smoothing",
                "setting": name,
                "sigma": None,
                "sigma_unit": "",
                "trials": 1,
                "p2a_count_ok_rate": float_or_none(source["p2a_count_ok"]),
                "p2a_success_rate": float_or_none(source["p2a_success"]),
                "aifit_reps_mean": float_or_none(source["aifit_reps"]),
                "aifit_reps_std": 0.0,
                "peak_reps_mean": float_or_none(source["peak_reps"]),
                "target_0218_air_jump_rope_reps": float_or_none(
                    source["target_0218_air_jump_rope_reps"]
                ),
                "unchanged_0173_air_jump_rope_reps": float_or_none(
                    source["unchanged_0173_air_jump_rope_reps"]
                ),
                "blind_03_micro_f1_mean": float_or_none(source["blind_03_micro_f1"]),
                "blind_03_micro_f1_std": 0.0,
                "blind_03_micro_iou_mean": float_or_none(source["blind_03_micro_iou"]),
                "blind_03_p95_boundary_error_frames_mean": float_or_none(
                    source["blind_03_p95_boundary_error_frames"]
                ),
                "blind_05_micro_f1_mean": float_or_none(source["blind_05_micro_f1"]),
                "walking_lunge_stride_f1_mean": float_or_none(
                    source["walking_lunge_stride_f1"]
                ),
                "mean_changed_segment_methods": float_or_none(
                    source["changed_segment_methods"]
                ),
                "selected_default": name == "w7p2",
            }
        )
    return rows


def summary_noise_rows(
    summary: dict[str, dict[str, Any]], family: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(summary, key=lambda item: float(item)):
        entry = summary[key]
        rows.append(
            {
                "family": family,
                "setting": f"sigma_{float(entry['sigma']):g}{entry['sigma_unit']}",
                "sigma": float(entry["sigma"]),
                "sigma_unit": entry["sigma_unit"],
                "trials": int(entry["trials"]),
                "p2a_count_ok_rate": float(entry["p2a_count_ok_rate"]),
                "p2a_success_rate": float(entry["p2a_success_rate"]),
                "aifit_reps_mean": entry["aifit_reps"]["mean"],
                "aifit_reps_std": entry["aifit_reps"]["std"],
                "peak_reps_mean": entry["peak_reps"]["mean"],
                "target_0218_air_jump_rope_reps": None,
                "unchanged_0173_air_jump_rope_reps": None,
                "blind_03_micro_f1_mean": entry["blind_03_micro_f1"]["mean"],
                "blind_03_micro_f1_std": entry["blind_03_micro_f1"]["std"],
                "blind_03_micro_iou_mean": entry["blind_03_micro_iou"]["mean"],
                "blind_03_p95_boundary_error_frames_mean": entry[
                    "blind_03_p95_boundary_error_frames"
                ]["mean"],
                "blind_05_micro_f1_mean": entry["blind_05_micro_f1"]["mean"],
                "walking_lunge_stride_f1_mean": entry["walking_lunge_stride_f1"]["mean"],
                "mean_changed_segment_methods": entry["mean_changed_segment_methods"],
                "selected_default": False,
            }
        )
    return rows


def markdown_table(
    path: Path,
    title: str,
    rows: list[dict[str, Any]],
    columns: list[tuple[str, str]],
    digits: int = 4,
    column_digits: dict[str, int] | None = None,
) -> None:
    column_digits = column_digits or {}
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(label for _key, label in columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        values = []
        row_digits = int(row.get("decimals", digits))
        for key, _label in columns:
            value = row.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)):
                values.append(format_value(value, column_digits.get(key, row_digits)))
            elif isinstance(value, bool):
                values.append("yes" if value else "no")
            else:
                values.append("" if value is None else str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def save_figure(figure: plt.Figure, base: Path, pdf: bool = True) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base.with_suffix(".png"), dpi=190, bbox_inches="tight")
    if pdf:
        figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def prepare_output(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"output directory already exists and is not empty: {output}\n"
            "pass --overwrite to refresh known files"
        )
    for directory in ("config", "data", "tables", "figures", "checks"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    return output


def copy_configs(args: argparse.Namespace, output: Path) -> None:
    for source, name in (
        (args.golden, "p2a_golden.json"),
        (args.rep_metadata, "rep_metadata.json"),
        (args.phase_config, "phase_guidance.json"),
    ):
        if source.exists():
            copy_file(source, output / "config" / name)


def copy_segmentation_data(args: argparse.Namespace, output: Path) -> Path:
    destination = output / "data" / "segmentation"
    destination.mkdir(parents=True, exist_ok=True)
    for name in SEGMENTATION_FILES:
        source = args.baseline_dir / name
        if source.exists():
            copy_file(source, destination / name)
    blind_destination = destination / "annotation_blind_eval"
    blind_destination.mkdir(parents=True, exist_ok=True)
    for name in BLIND_FILES:
        source = args.baseline_dir / "annotation_blind_eval" / name
        if source.exists():
            copy_file(source, blind_destination / name)
    return destination


def copy_robustness_data(args: argparse.Namespace, output: Path) -> None:
    destination = output / "data" / "robustness"
    summary_destination = destination / "summary"
    summary_destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "smoothing_ablation.csv",
        "smoothing_ablation.json",
        "noise_robustness.csv",
        "noise_robustness.json",
        "robustness_summary.json",
    ):
        source = args.robustness_dir / name
        if source.exists():
            copy_file(source, summary_destination / name)

    patterns = (
        "smoothing/*/robustness_config.json",
        "angle_noise/*/trial_*/robustness_config.json",
        "landmark_noise/*/trial_*/robustness_config.json",
    )
    for pattern in patterns:
        for config_path in sorted(args.robustness_dir.glob(pattern)):
            run_dir = config_path.parent
            relative = run_dir.relative_to(args.robustness_dir)
            run_destination = destination / "runs" / relative
            for name in ROBUSTNESS_FILES:
                source = run_dir / name
                if source.exists():
                    copy_file(source, run_destination / name)
            for name in ("blind_eval_summary.csv",):
                source = run_dir / "annotation_blind_eval" / name
                if source.exists():
                    copy_file(source, run_destination / "annotation_blind_eval" / name)
            source = run_dir / "p1b1_walking_lunges_eval.json"
            if source.exists():
                copy_file(source, run_destination / source.name)


def generate_qualitative_plots(
    args: argparse.Namespace, output: Path
) -> tuple[list[dict[str, Any]], list[Path]]:
    plotted_dir = output / "figures" / "qualitative"
    plotted_dir.mkdir(parents=True, exist_ok=True)
    index_path = output / "figures" / "qualitative_index.csv"
    if args.skip_qualitative and index_path.exists():
        rows = read_csv(index_path)
        paths = [output / row["path"] for row in rows]
        return rows, paths

    with tempfile.TemporaryDirectory(prefix="vls-final-plots-") as temporary:
        temporary_dir = Path(temporary) / "segmentation"
        result = run_command(
            [
                sys.executable,
                str(SEGMENT_SCRIPT),
                "--output-dir",
                str(temporary_dir),
                "--angle-scoring",
                "legacy",
                "--p2a-tail-filter",
                "guarded",
                "--smooth-window",
                "7",
                "--smooth-polyorder",
                "2",
            ],
            "qualitative segmentation",
            output / "checks" / "qualitative_segmentation.stdout.txt",
        )
        if result.returncode != 0:
            raise RuntimeError("qualitative segmentation failed")
        actual = boundaries(temporary_dir / "reps.csv")
        expected = boundaries(args.baseline_dir / "reps.csv")
        if actual != expected:
            raise RuntimeError("qualitative run does not reproduce the frozen P2-A boundaries")
        for source in sorted((temporary_dir / "plots").glob("*.png")):
            copy_file(source, plotted_dir / source.name)

    counts = rep_counts(args.baseline_dir / "reps.csv")
    rejected: dict[tuple[str, str], int] = defaultdict(int)
    for row in read_csv(args.baseline_dir / "rejected_reps.csv"):
        rejected[(row["video_id"], row["action"])] += 1
    rows: list[dict[str, Any]] = []
    for plot_path in sorted(plotted_dir.glob("*.png")):
        stem = plot_path.stem.removesuffix("_segmentation")
        video_id, action = stem.split("_", 1)
        key = (video_id, action)
        rows.append(
            {
                "video_id": video_id,
                "action": action,
                "aifit_reps": counts.get(key, {}).get("aifit", 0),
                "peak_reps": counts.get(key, {}).get("peak", 0),
                "rejected_reps": rejected.get(key, 0),
                "path": str(plot_path.relative_to(output)),
            }
        )
    write_csv(
        index_path,
        rows,
        ["video_id", "action", "aifit_reps", "peak_reps", "rejected_reps", "path"],
    )
    return rows, [plotted_dir / Path(row["path"]).name for row in rows]


def build_main_table(
    args: argparse.Namespace, output: Path
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, float | None]]]:
    previous = blind_rows(
        args.reference_dir / "annotation_blind_eval" / "blind_eval_summary.csv"
    )
    current = blind_rows(
        args.baseline_dir / "annotation_blind_eval" / "blind_eval_summary.csv"
    )
    previous_counts = rep_counts(args.reference_dir / "reps.csv")
    current_counts = rep_counts(args.baseline_dir / "reps.csv")
    previous_aifit = sum(values.get("aifit", 0) for values in previous_counts.values())
    current_aifit = sum(values.get("aifit", 0) for values in current_counts.values())
    previous_peak = sum(values.get("peak", 0) for values in previous_counts.values())
    current_peak = sum(values.get("peak", 0) for values in current_counts.values())
    walking = json.loads(
        (args.baseline_dir / "p1b1_walking_lunges_eval.json").read_text()
    )["acceptance"]["aifit_stride_combined"]

    def metric_rows(
        label: str,
        description: str,
        old: float | None,
        new: float | None,
        digits: int = 4,
    ) -> dict[str, Any]:
        delta = None if old is None or new is None else float(new) - float(old)
        return {
            "metric": label,
            "description": description,
            "previous_p1d": old,
            "final_p2a": new,
            "delta": delta,
            "decimals": digits,
        }

    rows = [
        metric_rows("aifit_reps", "AIFit reps across all 22 action segments", previous_aifit, current_aifit, 0),
        metric_rows("peak_reps", "Peak-baseline reps across all action segments", previous_peak, current_peak, 0),
        metric_rows("blind_gt_reps", "Blind GT reps in 5 evaluated segments", 66, 66, 0),
        metric_rows(
            "blind_pred_iou030", "AIFit predicted reps at IoU=0.30",
            previous[("aifit", "0.3")].get("pred_reps"), current[("aifit", "0.3")].get("pred_reps"), 0,
        ),
        metric_rows(
            "blind_tp_iou030", "True positives at IoU=0.30",
            previous[("aifit", "0.3")].get("matched"), current[("aifit", "0.3")].get("matched"), 0,
        ),
        metric_rows(
            "blind_fp_iou030", "False positives at IoU=0.30",
            previous[("aifit", "0.3")].get("false_positives"), current[("aifit", "0.3")].get("false_positives"), 0,
        ),
        metric_rows(
            "blind_fn_iou030", "False negatives at IoU=0.30",
            previous[("aifit", "0.3")].get("false_negatives"), current[("aifit", "0.3")].get("false_negatives"), 0,
        ),
        metric_rows(
            "blind_micro_f1_iou030", "Micro F1 at IoU=0.30",
            previous[("aifit", "0.3")].get("micro_f1"), current[("aifit", "0.3")].get("micro_f1"),
        ),
        metric_rows(
            "blind_raw_micro_iou", "Raw micro IoU, all matched boundaries",
            previous[("aifit", "0.3")].get("micro_iou"), current[("aifit", "0.3")].get("micro_iou"),
        ),
        metric_rows(
            "blind_mean_abs_start_frames", "Mean absolute start-boundary error (frames)",
            previous[("aifit", "0.3")].get("mean_abs_start_error_frames"), current[("aifit", "0.3")].get("mean_abs_start_error_frames"), 2,
        ),
        metric_rows(
            "blind_mean_abs_end_frames", "Mean absolute end-boundary error (frames)",
            previous[("aifit", "0.3")].get("mean_abs_end_error_frames"), current[("aifit", "0.3")].get("mean_abs_end_error_frames"), 2,
        ),
        metric_rows(
            "blind_p95_boundary_frames", "p95 boundary error (frames)",
            previous[("aifit", "0.3")].get("p95_boundary_error_frames"), current[("aifit", "0.3")].get("p95_boundary_error_frames"), 2,
        ),
        metric_rows(
            "blind_pred_iou050", "AIFit predicted reps at IoU=0.50",
            previous[("aifit", "0.5")].get("pred_reps"), current[("aifit", "0.5")].get("pred_reps"), 0,
        ),
        metric_rows(
            "blind_fp_iou050", "False positives at IoU=0.50",
            previous[("aifit", "0.5")].get("false_positives"), current[("aifit", "0.5")].get("false_positives"), 0,
        ),
        metric_rows(
            "blind_fn_iou050", "False negatives at IoU=0.50",
            previous[("aifit", "0.5")].get("false_negatives"), current[("aifit", "0.5")].get("false_negatives"), 0,
        ),
        metric_rows(
            "blind_micro_f1_iou050", "Micro F1 at IoU=0.50",
            previous[("aifit", "0.5")].get("micro_f1"), current[("aifit", "0.5")].get("micro_f1"),
        ),
        metric_rows("walking_stride_gt", "Walking-lunge GT strides (combined)", None, walking["gt_reps"], 0),
        metric_rows("walking_stride_pred", "Walking-lunge predicted strides (combined)", None, walking["pred_reps"], 0),
        metric_rows("walking_stride_f1", "Walking-lunge stride F1", None, walking["f1"]),
        metric_rows("target_0218_air_jump_rope_reps", "0218 air-jump-rope AIFit reps", 40, current_counts[("0218", "air_jump_rope")].get("aifit", 0), 0),
        metric_rows("unchanged_0173_air_jump_rope_reps", "0173 air-jump-rope AIFit reps", 24, current_counts[("0173", "air_jump_rope")].get("aifit", 0), 0),
    ]
    fields = ["metric", "description", "previous_p1d", "final_p2a", "delta"]
    write_csv(output / "tables" / "main_results.csv", rows, fields)
    markdown_table(
        output / "tables" / "main_results.md",
        "task1-3 final main results: P1-D vs P2-A",
        rows,
        [
            ("metric", "Metric"),
            ("previous_p1d", "P1-D"),
            ("final_p2a", "P2-A final"),
            ("delta", "Delta"),
        ],
    )
    return rows, current


def build_segment_table(args: argparse.Namespace, output: Path) -> list[dict[str, Any]]:
    previous = rep_counts(args.reference_dir / "reps.csv")
    current = rep_counts(args.baseline_dir / "reps.csv")
    segments = [
        (row["video_id"], row["action"])
        for row in read_csv(args.baseline_dir / "action_segments.csv")
    ]
    rows: list[dict[str, Any]] = []
    for video_id, action in sorted(segments):
        key = (video_id, action)
        old_aifit = previous.get(key, {}).get("aifit", 0)
        new_aifit = current.get(key, {}).get("aifit", 0)
        old_peak = previous.get(key, {}).get("peak", 0)
        new_peak = current.get(key, {}).get("peak", 0)
        rows.append(
            {
                "video_id": video_id,
                "action": action,
                "aifit_p1d": old_aifit,
                "aifit_p2a": new_aifit,
                "aifit_delta": new_aifit - old_aifit,
                "peak_p1d": old_peak,
                "peak_p2a": new_peak,
                "peak_delta": new_peak - old_peak,
            }
        )
    write_csv(
        output / "tables" / "segment_counts.csv",
        rows,
        ["video_id", "action", "aifit_p1d", "aifit_p2a", "aifit_delta", "peak_p1d", "peak_p2a", "peak_delta"],
    )
    markdown_table(
        output / "tables" / "segment_counts.md",
        "Per-segment rep counts: P1-D vs P2-A",
        rows,
        [
            ("video_id", "Video"),
            ("action", "Action"),
            ("aifit_p1d", "AIFit P1-D"),
            ("aifit_p2a", "AIFit P2-A"),
            ("aifit_delta", "Delta"),
            ("peak_p2a", "Peak P2-A"),
        ],
        column_digits={
            "aifit_p1d": 0,
            "aifit_p2a": 0,
            "aifit_delta": 0,
            "peak_p2a": 0,
        },
    )
    return rows


def build_ablation_tables(
    args: argparse.Namespace, output: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    smoothing = smoothing_rows(args.robustness_dir / "smoothing_ablation.csv")
    summary = json.loads((args.robustness_dir / "robustness_summary.json").read_text())
    angle = summary_noise_rows(summary["angle_noise_summary"], "angle_noise")
    landmark = summary_noise_rows(summary["landmark_noise_summary"], "landmark_noise")
    combined = smoothing + angle + landmark
    common_fields = [
        "family",
        "setting",
        "sigma",
        "sigma_unit",
        "trials",
        "p2a_count_ok_rate",
        "p2a_success_rate",
        "aifit_reps_mean",
        "aifit_reps_std",
        "peak_reps_mean",
        "target_0218_air_jump_rope_reps",
        "unchanged_0173_air_jump_rope_reps",
        "blind_03_micro_f1_mean",
        "blind_03_micro_f1_std",
        "blind_03_micro_iou_mean",
        "blind_03_p95_boundary_error_frames_mean",
        "blind_05_micro_f1_mean",
        "walking_lunge_stride_f1_mean",
        "mean_changed_segment_methods",
        "selected_default",
    ]
    write_csv(output / "tables" / "ablation_smoothing.csv", smoothing, common_fields)
    write_csv(output / "tables" / "ablation_angle_noise.csv", angle, common_fields)
    write_csv(output / "tables" / "ablation_landmark_noise.csv", landmark, common_fields)
    write_csv(output / "tables" / "ablation_all.csv", combined, common_fields)

    columns = [
        ("setting", "Setting"),
        ("sigma", "Sigma"),
        ("trials", "Trials"),
        ("p2a_success_rate", "P2-A success"),
        ("aifit_reps_mean", "AIFit reps"),
        ("blind_03_micro_f1_mean", "F1@0.30"),
        ("blind_03_micro_iou_mean", "raw IoU"),
        ("blind_03_p95_boundary_error_frames_mean", "p95 boundary"),
        ("blind_05_micro_f1_mean", "F1@0.50"),
        ("mean_changed_segment_methods", "Changed segments"),
    ]
    column_digits = {
        "sigma": 3,
        "trials": 0,
        "p2a_success_rate": 3,
        "aifit_reps_mean": 1,
        "blind_03_p95_boundary_error_frames_mean": 2,
        "mean_changed_segment_methods": 1,
    }
    markdown_table(
        output / "tables" / "ablation_smoothing.md",
        "Smoothing ablation",
        smoothing,
        columns,
        column_digits=column_digits,
    )
    markdown_table(
        output / "tables" / "ablation_angle_noise.md",
        "Angle-noise robustness",
        angle,
        columns,
        column_digits=column_digits,
    )
    markdown_table(
        output / "tables" / "ablation_landmark_noise.md",
        "Landmark-phase-noise robustness",
        landmark,
        columns,
        column_digits=column_digits,
    )
    (output / "tables" / "README.md").write_text(
        "# Final tables\n\n"
        "- `main_results.csv/.md`: P1-D vs frozen P2-A main metrics.\n"
        "- `segment_counts.csv/.md`: all 22 segment-method counts.\n"
        "- `ablation_smoothing.csv/.md`: six Savitzky-Golay settings.\n"
        "- `ablation_angle_noise.csv/.md`: angle-noise sigma sweep.\n"
        "- `ablation_landmark_noise.csv/.md`: nose-phase noise sigma sweep.\n"
        "- `ablation_all.csv`: uniform machine-readable ablation table.\n\n"
        "Noise rows are aggregated over three fixed-seed trials; sigma=0 uses one run.\n"
    )
    return smoothing, angle, landmark, combined


def plot_main_metrics(
    current: dict[tuple[str, str], dict[str, float | None]], output: Path
) -> None:
    thresholds = ["0.1", "0.3", "0.5"]
    labels = ["0.10", "0.30", "0.50"]
    x = np.arange(len(thresholds))
    width = 0.34
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.8))
    axis = axes[0, 0]
    for offset, method, color, label in ((-width / 2, "aifit", COLORS["aifit"], "AIFit"), (width / 2, "peak", COLORS["peak"], "Peak")):
        values = [current[(method, threshold)]["micro_f1"] for threshold in thresholds]
        axis.bar(x + offset, values, width, color=color, label=label)
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("micro F1")
    axis.set_title("Blind QEVD-FIT micro F1")
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.6)

    axis = axes[0, 1]
    metrics = ("matched", "false_positives", "false_negatives")
    metric_labels = ("TP", "FP", "FN")
    colors = ("#4c9a63", "#d18a36", "#9c4f4f")
    bottom = np.zeros(2)
    for metric, metric_label, color in zip(metrics, metric_labels, colors):
        values = np.asarray([current[("aifit", threshold)][metric] for threshold in ("0.3", "0.5")])
        axis.bar(["IoU 0.30", "IoU 0.50"], values, bottom=bottom, color=color, label=metric_label)
        bottom += values
    axis.set_ylabel("reps")
    axis.set_title("AIFit prediction outcomes")
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.6)

    axis = axes[1, 0]
    error_metrics = ("mean_abs_start_error_frames", "mean_abs_end_error_frames", "p95_boundary_error_frames")
    error_labels = ("mean |start|", "mean |end|", "p95 boundary")
    for offset, method, color, label in ((-width / 2, "aifit", COLORS["aifit"], "AIFit"), (width / 2, "peak", COLORS["peak"], "Peak")):
        values = [current[(method, "0.3")][metric] for metric in error_metrics]
        axis.bar(np.arange(3) + offset, values, width, color=color, label=label)
    axis.set_xticks(np.arange(3), error_labels)
    axis.set_ylabel("frames")
    axis.set_title("Boundary error at IoU=0.30")
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.6)

    axis = axes[1, 1]
    for offset, method, color, label in ((-width / 2, "aifit", COLORS["aifit"], "AIFit"), (width / 2, "peak", COLORS["peak"], "Peak")):
        values = [current[(method, "0.3")]["micro_iou"], current[(method, "0.3")]["phase_aligned_iou"]]
        axis.bar(np.arange(2) + offset, values, width, color=color, label=label)
    axis.set_xticks(np.arange(2), ["raw IoU", "phase-aligned IoU"])
    axis.set_ylim(0, 1.0)
    axis.set_title("IoU at IoU threshold 0.30")
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    figure.suptitle("task1-3 final: blind metrics on five QEVD-FIT segments", fontsize=13)
    handles = [
        Patch(color=COLORS["aifit"], label="AIFit"),
        Patch(color=COLORS["peak"], label="Peak"),
        Patch(color="#4c9a63", label="TP"),
        Patch(color="#d18a36", label="FP"),
        Patch(color="#9c4f4f", label="FN"),
    ]
    figure.legend(handles=handles, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.01))
    figure.tight_layout(rect=(0, 0.055, 1, 0.95))
    save_figure(figure, output / "figures" / "fig01_main_metrics")


def plot_smoothing(rows: list[dict[str, Any]], output: Path) -> None:
    labels = [row["setting"] for row in rows]
    x = np.arange(len(rows))
    reps = [row["aifit_reps_mean"] for row in rows]
    target = [row["target_0218_air_jump_rope_reps"] for row in rows]
    f1 = [row["blind_03_micro_f1_mean"] for row in rows]
    p95 = [row["blind_03_p95_boundary_error_frames_mean"] for row in rows]
    colors = [COLORS["default"] if row["selected_default"] else COLORS["aifit"] for row in rows]

    figure, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    axis = axes[0]
    axis.bar(x, reps, color=colors, width=0.62)
    axis.axhline(365, color=COLORS["muted"], linestyle="--", linewidth=1, label="final baseline 365")
    for index, (rep, count) in enumerate(zip(reps, target)):
        axis.text(index, rep + 2, f"{rep:.0f}\n0218={count:.0f}", ha="center", va="bottom", fontsize=8)
    axis.set_ylabel("AIFit reps")
    axis.set_title("Smoothing ablation: rep count and 0218 target")
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    axis.legend(frameon=False)

    axis = axes[1]
    axis.plot(x, f1, marker="o", color=COLORS["aifit"], label="micro F1 @ IoU 0.30")
    axis.set_ylabel("micro F1")
    axis.set_ylim(0.62, 1.03)
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    twin = axis.twinx()
    twin.plot(x, p95, marker="s", color=COLORS["peak"], label="p95 boundary (frames)")
    twin.set_ylabel("p95 boundary error (frames)")
    lines = axis.get_lines() + twin.get_lines()
    axis.legend(lines, [line.get_label() for line in lines], frameon=False, loc="lower left")
    axis.set_xticks(x, labels)
    axis.set_title("Smoothing ablation: blind metrics")
    figure.tight_layout()
    save_figure(figure, output / "figures" / "fig02_smoothing_ablation")


def plot_angle_noise(rows: list[dict[str, Any]], output: Path) -> None:
    sigmas = [row["sigma"] for row in rows]
    f1_03 = [row["blind_03_micro_f1_mean"] for row in rows]
    f1_03_std = [row["blind_03_micro_f1_std"] for row in rows]
    f1_05 = [row["blind_05_micro_f1_mean"] for row in rows]
    p95 = [row["blind_03_p95_boundary_error_frames_mean"] for row in rows]
    success = [100.0 * row["p2a_count_ok_rate"] for row in rows]
    trigger = [100.0 * row["p2a_success_rate"] for row in rows]

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    axis = axes[0]
    axis.errorbar(sigmas, f1_03, yerr=f1_03_std, marker="o", capsize=3, color=COLORS["aifit"], label="F1 @ 0.30")
    axis.plot(sigmas, f1_05, marker="s", color=COLORS["peak"], label="F1 @ 0.50")
    axis.set_xlabel("angle noise sigma (deg)")
    axis.set_ylabel("micro F1")
    axis.set_ylim(0.25, 1.05)
    axis.set_title("Blind micro F1")
    axis.grid(color=COLORS["grid"], linewidth=0.6)
    axis.legend(frameon=False)

    axis = axes[1]
    axis.plot(sigmas, p95, marker="o", color=COLORS["rejected"])
    axis.set_xlabel("angle noise sigma (deg)")
    axis.set_ylabel("p95 boundary error (frames)")
    axis.set_title("Boundary p95")
    axis.grid(color=COLORS["grid"], linewidth=0.6)

    axis = axes[2]
    axis.plot(sigmas, success, marker="o", color=COLORS["default"], label="0218 count correct")
    axis.plot(sigmas, trigger, marker="s", color=COLORS["aifit"], label="P2-A tail trigger")
    axis.set_xlabel("angle noise sigma (deg)")
    axis.set_ylabel("success rate (%)")
    axis.set_ylim(-5, 108)
    axis.set_title("P2-A target stability")
    axis.grid(color=COLORS["grid"], linewidth=0.6)
    axis.legend(frameon=False, loc="lower left")
    for sigma, value in zip(sigmas, f1_03):
        axes[0].annotate(f"{value:.2f}", (sigma, value), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
    figure.suptitle("Angle-noise robustness (three fixed-seed trials)", fontsize=12)
    figure.tight_layout()
    save_figure(figure, output / "figures" / "fig03_angle_noise")


def plot_landmark_noise(
    args: argparse.Namespace, summary: dict[str, Any], output: Path
) -> None:
    rows = summary_noise_rows(summary["landmark_noise_summary"], "landmark_noise")
    sigmas = [row["sigma"] for row in rows]
    count_ok = [100.0 * row["p2a_count_ok_rate"] for row in rows]
    trigger = [100.0 * row["p2a_success_rate"] for row in rows]
    f1 = [row["blind_03_micro_f1_mean"] for row in rows]

    raw_rows = [
        row
        for row in read_csv(args.robustness_dir / "noise_robustness.csv")
        if row["kind"] == "landmark_noise"
    ]
    by_sigma: dict[float, list[int]] = defaultdict(list)
    for row in raw_rows:
        by_sigma[float(row["sigma"])].append(int(row["unchanged_0173_air_jump_rope_reps"]))
    unchanged_rates = [100.0 * sum(value == 24 for value in by_sigma[sigma]) / len(by_sigma[sigma]) for sigma in sigmas]

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    axis = axes[0]
    x = np.arange(len(rows))
    width = 0.25
    axis.bar(x - width, count_ok, width, color=COLORS["default"], label="0218 count correct")
    axis.bar(x, trigger, width, color=COLORS["aifit"], label="P2-A tail trigger")
    axis.bar(x + width, unchanged_rates, width, color=COLORS["peak"], label="0173 unchanged")
    axis.set_xticks(x, [f"{sigma:g}" for sigma in sigmas])
    axis.set_xlabel("nose phase noise sigma (m)")
    axis.set_ylabel("rate (%)")
    axis.set_ylim(0, 112)
    axis.set_title("Target and non-target stability")
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.6)

    axis = axes[1]
    for sigma, values in sorted(by_sigma.items()):
        offsets = np.linspace(-0.05, 0.05, len(values))
        axis.scatter([sigma] * len(values) + offsets, values, color=COLORS["aifit"], s=24)
    axis.axhline(24, color=COLORS["muted"], linestyle="--", label="0173 baseline 24")
    axis.set_xlabel("nose phase noise sigma (m)")
    axis.set_ylabel("0173 air-jump-rope reps")
    axis.set_title("Non-target count per trial")
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    handles = [
        Patch(color=COLORS["default"], label="0218 count correct"),
        Patch(color=COLORS["aifit"], label="P2-A tail trigger"),
        Patch(color=COLORS["peak"], label="0173 unchanged"),
        plt.Line2D([], [], color=COLORS["muted"], linestyle="--", label="0173 baseline 24"),
    ]
    figure.legend(handles=handles, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.01))
    figure.suptitle("Landmark phase-noise robustness (nose only)", fontsize=12)
    figure.tight_layout(rect=(0, 0.08, 1, 0.93))
    save_figure(figure, output / "figures" / "fig04_landmark_noise")


def plot_target_case(args: argparse.Namespace, output: Path) -> None:
    run_payload = json.loads((args.baseline_dir / "run.json").read_text())
    segment_payload = next(
        segment
        for segment in run_payload["segments"]
        if segment["video_id"] == "0218" and segment["action"] == "air_jump_rope"
    )
    angle = segment_payload["angles"]["aifit"]["angle"]
    window = segment_payload["window"]
    angle_path = args.angles_dir / "0218" / "angles_wide.csv"
    values = np.asarray(
        [float(row[angle]) for row in read_csv(angle_path)], dtype=np.float64
    )
    frames = np.arange(len(values))
    valid = np.isfinite(values)
    filled = np.interp(frames, frames[valid], values[valid])
    window_start = max(window["start_frame"] - 40, 1560)
    window_end = min(window["end_frame"] + 20, 1710)
    slice_values = filled[window_start:window_end]
    smooth = savgol_filter(slice_values, 7, 2, mode="interp")
    slice_frames = np.arange(window_start, window_end)

    aifit_reps = [
        (int(row["start_frame"]), int(row["end_frame"]))
        for row in read_csv(args.baseline_dir / "reps.csv")
        if row["video_id"] == "0218"
        and row["action"] == "air_jump_rope"
        and row["method"] == "aifit"
    ]
    rejected = next(
        row
        for row in read_csv(args.baseline_dir / "rejected_reps.csv")
        if row["video_id"] == "0218" and row["action"] == "air_jump_rope"
    )
    rejected_interval = (int(rejected["start_frame"]), int(rejected["end_frame"]))
    evidence = ast.literal_eval(rejected["evidence"])
    fit_end = int(evidence["fit_end_frame"])

    figure, axis = plt.subplots(figsize=(11.5, 4.2))
    axis.plot(slice_frames, slice_values, color="#999999", linewidth=0.9, label="raw angle")
    axis.plot(slice_frames, smooth, color="#222222", linewidth=1.5, label="SG 7/2")
    for index, (start, end) in enumerate(aifit_reps):
        if start < window_start or end > window_end:
            continue
        axis.axvspan(start, end, color=COLORS["aifit"], alpha=0.08, linewidth=0)
        if index >= len(aifit_reps) - 4:
            axis.axvline(start, color=COLORS["aifit"], linewidth=0.9, linestyle="--")
    axis.axvspan(
        rejected_interval[0],
        rejected_interval[1],
        facecolor=COLORS["rejected"],
        edgecolor="#9b3d0f",
        hatch="//",
        alpha=0.18,
        linewidth=0.8,
        label=f"P2-A rejected {rejected_interval[0]}-{rejected_interval[1]}",
    )
    axis.axvline(fit_end, color=COLORS["default"], linewidth=1.2, linestyle=":", label=f"fit end {fit_end}")
    axis.set_xlim(window_start, window_end)
    axis.set_xlabel("decoded frame index")
    axis.set_ylabel(f"{angle} (deg)")
    axis.set_title("0218 air jump rope: P2-A removes the transition tail")
    axis.grid(color=COLORS["grid"], linewidth=0.6)
    axis.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        fontsize=8,
    )
    axis.text(
        0.01,
        0.04,
        f"support=3/3 | shape {evidence['shape_similarity']:.3f} < {evidence['shape_limit']:.3f} | "
        f"motion {evidence['motion_evidence']:.4f} < {evidence['motion_limit']:.3f}",
        transform=axis.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    figure.tight_layout(rect=(0, 0, 0.79, 1))
    save_figure(figure, output / "figures" / "fig05_0218_tail_case")


def build_contact_sheet(paths: list[Path], output: Path) -> None:
    if not paths:
        return
    thumb_width = 520
    columns = 3
    margin = 16
    title_height = 36
    boxes = []
    for path in paths:
        with Image.open(path) as image:
            ratio = image.height / image.width
            boxes.append((path, thumb_width, int(thumb_width * ratio)))
    rows = (len(boxes) + columns - 1) // columns
    cell_height = max(height for _path, _width, height in boxes) + 26
    sheet = Image.new(
        "RGB",
        (columns * thumb_width + (columns + 1) * margin, rows * cell_height + title_height + margin),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((margin, 10), "task1-3 final qualitative segmentation plots", fill="black", font=font)
    for index, (path, width, height) in enumerate(boxes):
        column = index % columns
        row = index // columns
        left = margin + column * (thumb_width + margin)
        top = title_height + margin + row * cell_height
        with Image.open(path) as image:
            thumbnail = image.convert("RGB")
            thumbnail.thumbnail((width, height))
            sheet.paste(thumbnail, (left, top + 18))
        draw.text((left, top), path.stem.replace("_segmentation", ""), fill="#333333", font=font)
    sheet.save(output / "figures" / "fig06_qualitative_contact_sheet.png")
    sheet.save(output / "figures" / "fig06_qualitative_contact_sheet.pdf", "PDF", resolution=150)


def build_figure_index(output: Path) -> None:
    rows = [
        {
            "figure": "fig01_main_metrics",
            "description": "Blind P/R/F1, prediction outcomes, boundary error and IoU.",
        },
        {
            "figure": "fig02_smoothing_ablation",
            "description": "Smoothing ablation: rep count, 0218 target and blind metrics.",
        },
        {
            "figure": "fig03_angle_noise",
            "description": "Angle-noise robustness: F1, p95 boundary and P2-A target stability.",
        },
        {
            "figure": "fig04_landmark_noise",
            "description": "Nose phase-noise robustness and non-target rep count.",
        },
        {
            "figure": "fig05_0218_tail_case",
            "description": "Annotated 0218 air-jump-rope tail rejection case.",
        },
        {
            "figure": "fig06_qualitative_contact_sheet",
            "description": "Contact sheet of all final qualitative segmentation plots.",
        },
    ]
    write_csv(output / "figures" / "figure_index.csv", rows, ["figure", "description"])


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    args: argparse.Namespace,
    output: Path,
    validation: dict[str, Any],
    current: dict[tuple[str, str], dict[str, float | None]],
) -> None:
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "file_checksums.sha256"}:
            continue
        relative = path.relative_to(output)
        files.append(
            {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": checksum(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "name": "task1-3-final-v0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": "Canonical task1-3 V0 output with frozen P2-A defaults, final figures and ablations.",
        "sources": {
            "baseline_segmentation": str(args.baseline_dir),
            "reference_segmentation": str(args.reference_dir),
            "robustness": str(args.robustness_dir),
            "angles": str(args.angles_dir),
            "golden": str(args.golden),
        },
        "parameters": json.loads((args.baseline_dir / "run.json").read_text())["parameters"],
        "key_metrics": {
            "aifit_reps": 365,
            "peak_reps": 341,
            "blind_iou_030": {
                "pred": current[("aifit", "0.3")]["pred_reps"],
                "tp": current[("aifit", "0.3")]["matched"],
                "fp": current[("aifit", "0.3")]["false_positives"],
                "fn": current[("aifit", "0.3")]["false_negatives"],
                "micro_f1": current[("aifit", "0.3")]["micro_f1"],
            },
            "walking_lunge_stride_f1": 1.0,
        },
        "validation": validation,
        "files": files,
    }
    write_json(output / "manifest.json", manifest)
    lines = [f"{entry['sha256']}  {entry['path']}" for entry in files]
    (output / "checks" / "file_checksums.sha256").write_text("\n".join(lines) + "\n")


def write_readme(
    args: argparse.Namespace,
    output: Path,
    validation: dict[str, Any],
    current: dict[tuple[str, str], dict[str, float | None]],
) -> None:
    blind = current[("aifit", "0.3")]
    readme = f"""# task1-3 final output (V0)\n\nCanonical result snapshot after P2-A closure. This directory is generated from frozen data and is the entry point for final tables and figures.\n\n## Canonical status\n\n- Parameters: `angle_scoring=legacy`, `p2a_tail_filter=guarded`, smoothing `7/2`.\n- AIFit reps: `365`; Peak reps: `341`.\n- `0218/air_jump_rope`: `39` reps, tail `1663-1679` rejected; `0173/air_jump_rope`: `24` reps.\n- Blind QEVD-FIT, IoU=0.30: Pred/TP/FP/FN `{blind['pred_reps']:.0f}/{blind['matched']:.0f}/{blind['false_positives']:.0f}/{blind['false_negatives']:.0f}`, micro F1 `{blind['micro_f1']:.4f}`.\n- Walking-lunge stride: `12/12`, F1 `1.0`.\n- P2-A validation: `{validation.get('status', 'unknown')}`.\n\n## Layout\n\n```text\nsegmentation_final/\n├── README.md\n├── manifest.json\n├── config/                  # golden and metadata snapshots\n├── data/\n│   ├── segmentation/        # canonical P2-A tables and blind evaluation\n│   └── robustness/          # summary tables and per-run audit data\n├── tables/                  # main and ablation tables (CSV + Markdown)\n├── figures/\n│   ├── fig01_main_metrics.*\n│   ├── fig02_smoothing_ablation.*\n│   ├── fig03_angle_noise.*\n│   ├── fig04_landmark_noise.*\n│   ├── fig05_0218_tail_case.*\n│   ├── fig06_qualitative_contact_sheet.*\n│   └── qualitative/         # per-segment final plots\n└── checks/                  # regression report and SHA-256 checksums\n```\n\n## Key files\n\n- `tables/main_results.csv`: P1-D vs P2-A main metrics.\n- `tables/ablation_all.csv`: uniform smoothing/noise ablation table.\n- `data/segmentation/reps.csv`: final rep boundaries.\n- `data/segmentation/annotation_blind_eval/blind_eval_summary.csv`: blind metrics.\n- `figures/fig05_0218_tail_case.png`: annotated P2-A target case.\n- `checks/p2a_regression.json`: golden validation of this snapshot.\n\n## Reproduce\n\n```bash\n.venv/bin/python VLS/V0/scripts/build_final_outputs.py --overwrite\n.venv/bin/python VLS/V0/scripts/check_p2a_regression.py \\\n  --segmentation-dir VLS/V0/results/segmentation_final/data/segmentation \\\n  --report VLS/V0/results/segmentation_final/checks/p2a_regression.json\n```\n\n## Scope\n\nQEVD-FIT is an internal blind benchmark, not Fit3D. The numbers are not the official AIFit paper metrics. The independent second annotator remains pending.\n"""
    (output / "README.md").write_text(readme)


def validate_final(args: argparse.Namespace, output: Path, segmentation_dir: Path) -> dict[str, Any]:
    report_path = output / "checks" / "p2a_regression.json"
    result = run_command(
        [
            sys.executable,
            str(CHECK_SCRIPT),
            "--golden",
            str(args.golden),
            "--segmentation-dir",
            str(segmentation_dir),
            "--reference-dir",
            str(args.reference_dir),
            "--report",
            str(report_path),
        ],
        "final P2-A regression",
        output / "checks" / "p2a_regression.stdout.txt",
    )
    payload = json.loads(report_path.read_text())
    if result.returncode != 0 or payload.get("status") != "pass":
        raise RuntimeError(f"final P2-A validation failed: {report_path}")
    return payload


def main() -> int:
    args = parse_args()
    for name in (
        "baseline_dir",
        "reference_dir",
        "robustness_dir",
        "angles_dir",
        "golden",
        "rep_metadata",
        "phase_config",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    output = prepare_output(args)
    copy_configs(args, output)
    segmentation_dir = copy_segmentation_data(args, output)
    copy_robustness_data(args, output)
    qualitative_rows, qualitative_paths = generate_qualitative_plots(args, output)
    _, current = build_main_table(args, output)
    segment_rows = build_segment_table(args, output)
    smoothing, angle, landmark, combined = build_ablation_tables(args, output)
    summary = json.loads((args.robustness_dir / "robustness_summary.json").read_text())
    plot_main_metrics(current, output)
    plot_smoothing(smoothing, output)
    plot_angle_noise(angle, output)
    plot_landmark_noise(args, summary, output)
    plot_target_case(args, output)
    build_contact_sheet(qualitative_paths, output)
    build_figure_index(output)
    validation = validate_final(args, output, segmentation_dir)
    write_readme(args, output, validation, current)
    write_manifest(args, output, validation, current)
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "segments": len(segment_rows),
                "qualitative_plots": len(qualitative_rows),
                "smoothing_rows": len(smoothing),
                "angle_noise_rows": len(angle),
                "landmark_noise_rows": len(landmark),
                "ablation_rows": len(combined),
                "validation": validation.get("status"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
