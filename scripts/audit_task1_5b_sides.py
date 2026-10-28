#!/usr/bin/env python3
"""Audit GT near-side depth, manifest camera side, and MediaPipe side choice."""

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

import build_task1_5_fit3d_manifest as manifest_builder
import task1_5_metrics as metrics


DEFAULT_MANIFEST = ROOT / "configs" / "task1_5_fit3d_clips.json"
DEFAULT_ANALYSIS_DIR = ROOT / "results" / "task1_5" / "clip_analysis"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "task1_5" / "manifest" / "side_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def view_side(view: str) -> str | None:
    if view.endswith("_left"):
        return "left"
    if view.endswith("_right"):
        return "right"
    return None


def mediapipe_side_visibility(
    exercise: str, landmarks_path: Path
) -> dict[str, float]:
    specs = metrics.SIDE_SIGNAL_SPECS.get(exercise, {})
    if not specs or not landmarks_path.is_file():
        return {}
    values: dict[str, list[float]] = {side: [] for side in specs}
    for record in json.loads(landmarks_path.read_text()):
        if not record.get("detected"):
            continue
        visibility = np.asarray(record.get("visibility") or [], dtype=np.float64)
        for side, spec in specs.items():
            indices = spec["mediapipe_joints"]
            if visibility.size <= max(indices):
                continue
            values[side].append(float(np.min(visibility[list(indices)])))
    return {
        side: float(np.median(samples)) if samples else 0.0
        for side, samples in values.items()
    }


def load_signal_selection(analysis_dir: Path, clip_id: str) -> dict[str, Any]:
    path = analysis_dir / clip_id / "analysis.json"
    if not path.is_file():
        return {}
    analysis = json.loads(path.read_text())
    return (analysis.get("segmentation") or {}).get("signal_selection") or {}


def audit_clip(
    clip: dict[str, Any],
    analysis_dir: Path,
) -> dict[str, Any]:
    joints = np.asarray(
        json.loads(Path(clip["gt"]["joints3d_25"]).read_text())["joints3d_25"],
        dtype=np.float64,
    )
    camera = manifest_builder.load_camera_parameters(
        Path(clip["gt"]["camera_parameters"])
    )
    rotation = camera["rotation"]
    translation = camera["translation"]
    rep_frames = [int(value) for value in clip["gt"]["rep_frames"]]
    camera_points = joints @ rotation.T + translation
    depth = metrics.near_side_from_camera_points(
        camera_points, clip["exercise"], rep_frames
    )

    geometry = clip.get("view_geometry") or {}
    manifest_side = geometry.get("camera_side")
    view = str(clip.get("view", "unknown"))
    label_side = view_side(view)
    selection = load_signal_selection(analysis_dir, clip["clip_id"])
    candidates = selection.get("candidates") or {}
    near = candidates.get(depth["near_side"]) or {}
    selected_side = selection.get("selected_side")
    selected_angle = selection.get("selected_angle")
    selection_reason = selection.get("selection_reason")

    return {
        "clip_id": clip["clip_id"],
        "subject_id": clip["subject_id"],
        "exercise": clip["exercise"],
        "camera_id": clip["camera_id"],
        "split": clip.get("split"),
        "view": view,
        "view_label_side": label_side or "",
        "manifest_camera_side": manifest_side,
        "gt_near_side": depth["near_side"],
        "gt_left_depth": depth["left_depth"],
        "gt_right_depth": depth["right_depth"],
        "gt_signed_depth_margin_right_minus_left": depth["signed_depth_margin"],
        "geometry_side_matches_depth": str(manifest_side == depth["near_side"]).lower(),
        "view_suffix_matches_depth": (
            "" if label_side is None else str(label_side == depth["near_side"]).lower()
        ),
        "near_side_visibility_median": near.get("visibility_median"),
        "selected_side": selected_side or "",
        "selected_angle": selected_angle or "",
        "selection_reason": selection_reason or "",
        "selection_low_confidence": (
            "" if not selection else str(bool(selection.get("low_confidence"))).lower()
        ),
        "selected_matches_gt_near_side": (
            ""
            if label_side is None
            else str(selected_side == depth["near_side"]).lower()
        ),
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    clips = [
        clip
        for clip in manifest["clips"]
        if clip.get("has_gt") and not clip.get("smoke_only")
    ]
    rows = [audit_clip(clip, args.analysis_dir.resolve()) for clip in clips]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "side_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    side_rows = [row for row in rows if row["view"] in metrics.SIDE_VIEWS]
    selected_rows = [row for row in side_rows if row["selected_side"]]
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["exercise"], row["view"])].append(row)
    summary = {
        "schema_version": "task1-5b-side-audit-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "analysis_dir": str(args.analysis_dir.resolve()),
        "clip_count": len(rows),
        "geometry_side_mismatch_count": sum(
            row["geometry_side_matches_depth"] == "false" for row in rows
        ),
        "view_suffix_mismatch_count": sum(
            row["view_suffix_matches_depth"] == "false" for row in rows
        ),
        "near_side_counts": dict(Counter(row["gt_near_side"] for row in rows)),
        "manifest_camera_side_counts": dict(
            Counter(str(row["manifest_camera_side"]) for row in rows)
        ),
        "side_view_clips": len(side_rows),
        "side_view_with_selection": len(selected_rows),
        "selected_side_mismatch_count": sum(
            row["selected_matches_gt_near_side"] == "false" for row in side_rows
        ),
        "selection_reason_counts": dict(
            Counter(row["selection_reason"] for row in side_rows if row["selection_reason"])
        ),
        "by_exercise_view": {
            exercise: {
                view: {
                    "clip_count": len(group),
                    "near_side_counts": dict(
                        Counter(row["gt_near_side"] for row in group)
                    ),
                    "selected_side_counts": dict(
                        Counter(row["selected_side"] for row in group if row["selected_side"])
                    ),
                }
                for (item_exercise, view), group in sorted(grouped.items())
                if item_exercise == exercise
            }
            for exercise in sorted({row["exercise"] for row in rows})
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "clip_count",
                    "geometry_side_mismatch_count",
                    "view_suffix_mismatch_count",
                    "side_view_with_selection",
                    "selected_side_mismatch_count",
                    "selection_reason_counts",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
