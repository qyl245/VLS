#!/usr/bin/env python3
"""Plot representative angle curves for the task1-2 outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import DATA_ROOT


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANGLES_DIR = ROOT / "outputs" / "angles"
DEFAULT_FEEDBACK = DATA_ROOT / "feedbacks_long_range.json"

PLOT_PLAN = {
    "0153": ["squat", "jumping_jack"],
    "0161": ["pushup", "plank"],
    "0162": ["lunge", "jumping_jack"],
    "0173": ["squat", "armcrosschest"],
    "0218": ["lunge"],
}
PATTERNS = {
    "squat": ("squat",),
    "jumping_jack": ("jumping jack",),
    "pushup": ("pushup", "push-up", "push up"),
    "lunge": ("lunge",),
    "plank": ("plank",),
    "armcrosschest": ("armcrosschest", "arm across"),
}
ANGLE_SETS = {
    "squat": ("left_knee_flexion", "right_knee_flexion", "torso_flexion"),
    "jumping_jack": (
        "left_shoulder_abduction_coronal",
        "right_shoulder_abduction_coronal",
        "left_knee_flexion",
    ),
    "pushup": ("left_elbow_flexion", "right_elbow_flexion", "torso_flexion"),
    "lunge": ("left_knee_flexion", "right_knee_flexion", "torso_flexion"),
    "plank": ("torso_flexion", "spine_stability"),
    "armcrosschest": (
        "left_shoulder_horizontal_abduction_transverse",
        "right_shoulder_horizontal_abduction_transverse",
        "torso_flexion",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angles-dir", type=Path, default=DEFAULT_ANGLES_DIR)
    parser.add_argument("--feedback-json", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_ANGLES_DIR / "plots")
    parser.add_argument("--pad-frames", type=int, default=15)
    args = parser.parse_args()
    args.angles_dir = args.angles_dir.resolve()
    args.feedback_json = args.feedback_json.resolve()
    args.plots_dir = args.plots_dir.resolve()
    return args


def load_feedback(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text())
    return {
        Path(item["long_range_video_file"]).name: item.get("feedbacks", [])
        for item in payload
    }


def matching_segments(texts: list[str], patterns: tuple[str, ...]) -> list[tuple[int, int]]:
    matches = np.array([any(pattern in text.lower() for pattern in patterns) for text in texts])
    segments = []
    start = None
    for index, matched in enumerate(np.append(matches, False)):
        if matched and start is None:
            start = index
        elif not matched and start is not None:
            segments.append((start, index - 1))
            start = None
    return segments


def select_segment(
    texts: list[str],
    patterns: tuple[str, ...],
    wide: dict[str, np.ndarray],
    score_angles: tuple[str, ...],
) -> tuple[int, int] | None:
    segments = matching_segments(texts, patterns)
    if not segments:
        return None
    scored = []
    for start, end in segments:
        if end - start + 1 < 5:
            continue
        series = np.vstack([wide[name][start : end + 1] for name in score_angles])
        finite = np.isfinite(series)
        counts = finite.sum(axis=0)
        finite_sum = np.where(finite, series, 0.0).sum(axis=0)
        mean_series = np.divide(
            finite_sum,
            counts,
            out=np.full(series.shape[1], np.nan, dtype=np.float64),
            where=counts > 0,
        )
        valid = mean_series[np.isfinite(mean_series)]
        if len(valid) < 5:
            continue
        low, high = np.percentile(valid, [5, 95])
        scored.append((float(high - low), end - start + 1, start, end))
    if not scored:
        start, end = max(segments, key=lambda item: item[1] - item[0])
        return start, end
    _, _, start, end = max(scored)
    return start, end


def load_wide(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        values = {name: [] for name in columns}
        for row in reader:
            for name in columns:
                values[name].append(float(row[name]))
    return columns, {name: np.asarray(value, dtype=np.float64) for name, value in values.items()}


def plot_action(
    video_id: str,
    action: str,
    texts: list[str],
    wide: dict[str, np.ndarray],
    output_path: Path,
    pad_frames: int,
) -> dict:
    score_angles = (ANGLE_SETS[action][0],)
    segment = select_segment(texts, PATTERNS[action], wide, score_angles)
    if segment is None:
        raise ValueError(f"no feedback segment for {video_id} {action}")
    start = max(0, segment[0] - pad_frames)
    end = min(len(texts) - 1, segment[1] + pad_frames)
    timestamps = wide["timestamp_ms"][start : end + 1] / 1000.0

    figure, axis = plt.subplots(figsize=(12, 4.5))
    colors = ["#d1495b", "#00798c", "#edae49", "#30638e"]
    primary = ANGLE_SETS[action]
    secondary_axis = None
    if action == "plank":
        secondary_axis = axis.twinx()

    for index, name in enumerate(primary):
        series = wide[name][start : end + 1]
        target = secondary_axis if name == "spine_stability" and secondary_axis is not None else axis
        target.plot(
            timestamps,
            series,
            color=colors[index],
            linewidth=1.8,
            label=name + (" (variance)" if name == "spine_stability" else ""),
        )

    segment_start_time = wide["timestamp_ms"][segment[0]] / 1000.0
    segment_end_time = wide["timestamp_ms"][segment[1]] / 1000.0
    axis.axvspan(segment_start_time, segment_end_time, color="#444444", alpha=0.08)
    axis.set_title(f"{video_id} | {action} | feedback frames {segment[0]}-{segment[1]}")
    axis.set_xlabel("video playback time (s)")
    axis.set_ylabel("angle (deg)")
    axis.set_ylim(0, 180)
    axis.grid(alpha=0.25)
    if secondary_axis is not None:
        secondary_axis.set_ylabel("spine variance")
        secondary_axis.set_ylim(bottom=0)
    lines = []
    labels = []
    for target in [axis, secondary_axis]:
        if target is None:
            continue
        line, label = target.get_legend_handles_labels()
        lines.extend(line)
        labels.extend(label)
    axis.legend(lines, labels, loc="upper right", fontsize=8, ncol=2)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return {
        "video_id": video_id,
        "action": action,
        "feedback_start_frame": segment[0],
        "feedback_end_frame": segment[1],
        "plot_start_frame": start,
        "plot_end_frame": end,
        "angles": list(primary),
        "path": str(output_path),
    }


def main() -> int:
    args = parse_args()
    feedback = load_feedback(args.feedback_json)
    plotted = []
    for video_id, actions in PLOT_PLAN.items():
        video_dir = args.angles_dir / video_id
        wide_path = video_dir / "angles_wide.csv"
        if not wide_path.exists():
            raise FileNotFoundError(wide_path)
        _, wide = load_wide(wide_path)
        for action in actions:
            output_path = args.plots_dir / f"{video_id}_{action}.png"
            record = plot_action(
                video_id,
                action,
                feedback[f"{video_id}.mp4"],
                wide,
                output_path,
                args.pad_frames,
            )
            plotted.append(record)
            print(f"plotted {output_path.name}", flush=True)
    (args.plots_dir / "index.json").write_text(
        json.dumps({"task": "task1-2", "plots": plotted}, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"generated {len(plotted)} plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
