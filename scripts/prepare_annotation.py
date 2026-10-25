#!/usr/bin/env python3
"""Create an editable rep-annotation template from task1-3 AIFit output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPS = ROOT / "results" / "segmentation" / "reps.csv"
DEFAULT_SUMMARY = ROOT / "results" / "segmentation" / "summary.csv"
DEFAULT_RUN = ROOT / "results" / "segmentation" / "run.json"
DEFAULT_OUTPUT = ROOT / "results" / "segmentation" / "annotation"
DEFAULT_PLOTS = ROOT / "results" / "segmentation" / "plots"

DEFAULT_SEGMENTS = (
    ("0153", "squat"),
    ("0153", "pushup"),
    ("0161", "lunge_jumps"),
    ("0161", "pushup"),
    ("0173", "good_morning"),
    ("0173", "boxing_squat_punches"),
    ("0218", "walking_lunges"),
)

ANNOTATION_FIELDS = [
    "segment_id",
    "video_id",
    "action",
    "rep_id",
    "draft_start_frame",
    "draft_end_frame",
    "gt_start_frame",
    "gt_end_frame",
    "review_status",
    "phase_landmark",
    "boundary_confidence",
    "annotator",
    "reviewed_at",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=Path, default=DEFAULT_REPS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--segments",
        nargs="*",
        default=None,
        help="Segments as video_id:action. Defaults to the 7-segment starter set.",
    )
    args = parser.parse_args()
    if args.segments:
        parsed = []
        for item in args.segments:
            if ":" not in item:
                parser.error("--segments entries must use video_id:action")
            video_id, action = item.split(":", 1)
            parsed.append((video_id, action))
        args.selected_segments = parsed
    else:
        args.selected_segments = list(DEFAULT_SEGMENTS)
    return args


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    template_path = args.output_dir / "annotation_template.csv"
    if template_path.exists() and not args.overwrite:
        raise SystemExit(
            f"{template_path} already exists; use --overwrite only after backing up reviewed annotations"
        )
    reps = read_csv(args.reps)
    summary = read_csv(args.summary)
    run = json.loads(args.run.read_text())
    run_segments = {
        (item["video_id"], item["action"]): item
        for item in run.get("segments", [])
    }
    selected = {(str(video_id).zfill(4), action): (str(video_id).zfill(4), action) for video_id, action in args.selected_segments}

    annotation_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    for (video_id, action) in selected.values():
        key = (video_id, action)
        if key not in run_segments:
            raise ValueError(f"segment not found in run.json: {video_id}/{action}")
        segment_reps = [
            row for row in reps
            if row["video_id"].zfill(4) == video_id
            and row["action"] == action
            and row["method"] == "aifit"
        ]
        if not segment_reps:
            raise ValueError(f"no AIFit reps for {video_id}/{action}")
        segment_reps.sort(key=lambda row: int(row["rep_id"]))
        segment_id = f"{video_id}_{action}"
        for row in segment_reps:
            annotation_rows.append(
                {
                    "segment_id": segment_id,
                    "video_id": video_id,
                    "action": action,
                    "rep_id": int(row["rep_id"]),
                    "draft_start_frame": int(row["start_frame"]),
                    "draft_end_frame": int(row["end_frame"]),
                    "gt_start_frame": int(row["start_frame"]),
                    "gt_end_frame": int(row["end_frame"]),
                    "review_status": "pending",
                    "phase_landmark": "",
                    "boundary_confidence": "",
                    "annotator": "",
                    "reviewed_at": "",
                    "notes": "",
                }
            )
        summary_rows = [
            row for row in summary
            if row["video_id"].zfill(4) == video_id and row["action"] == action
        ]
        by_method = {row["method"]: row for row in summary_rows}
        aifit = by_method.get("aifit", {})
        peak = by_method.get("peak", {})
        plot_path = args.plots_dir / f"{video_id}_{action}_segmentation.png"
        segment_rows.append(
            {
                "segment_id": segment_id,
                "video_id": video_id,
                "action": action,
                "window_start_frame": run_segments[key]["window"]["start_frame"],
                "window_end_frame": run_segments[key]["window"]["end_frame"],
                "main_angle": aifit.get("angle", ""),
                "aifit_reps": aifit.get("num_reps", len(segment_reps)),
                "peak_reps": peak.get("num_reps", ""),
                "method_agreement_iou": aifit.get("agreement_iou", ""),
                "plot_path": str(plot_path) if plot_path.exists() else "",
                "draft_source": "AIFit",
                "notes": "",
            }
        )

    annotation_rows.sort(key=lambda row: (row["video_id"], row["action"], row["rep_id"]))
    write_csv(template_path, annotation_rows, ANNOTATION_FIELDS)
    write_csv(
        args.output_dir / "annotation_segments.csv",
        segment_rows,
        [
            "segment_id",
            "video_id",
            "action",
            "window_start_frame",
            "window_end_frame",
            "main_angle",
            "aifit_reps",
            "peak_reps",
            "method_agreement_iou",
            "plot_path",
            "draft_source",
            "notes",
        ],
    )
    print(f"segments={len(segment_rows)} draft_reps={len(annotation_rows)}")
    print(f"template={args.output_dir / 'annotation_template.csv'}")
    print(f"segments={args.output_dir / 'annotation_segments.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
