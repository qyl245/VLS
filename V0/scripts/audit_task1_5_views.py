#!/usr/bin/env python3
"""Audit Fit3D camera views and generate contact sheets for manual review."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import build_task1_5_fit3d_manifest as manifest_builder


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "task1_5_fit3d_clips.json"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "task1_5" / "manifest"
TILE_SIZE = 240
TEXT_HEIGHT = 48
TITLE_HEIGHT = 54


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overrides", type=Path, default=None)
    parser.add_argument("--sheet-frame-offset", type=int, default=30)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"cannot decode frame {frame_index}: {video_path}")
        return frame
    finally:
        cap.release()


def audit_frame(clip: dict[str, Any], offset: int) -> int:
    rep_frames = (clip.get("gt") or {}).get("rep_frames") or []
    if not rep_frames:
        return 0
    base = int(rep_frames[len(rep_frames) // 2])
    return min(int(clip["trim"]["end_frame"]), max(0, base + offset))


def make_tile(clip: dict[str, Any], frame: np.ndarray, subject_label: str | None = None) -> np.ndarray:
    image = cv2.resize(frame, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_AREA)
    tile = np.full((TILE_SIZE + TEXT_HEIGHT, TILE_SIZE, 3), 24, dtype=np.uint8)
    tile[:TILE_SIZE] = image
    geometry = clip.get("view_geometry") or {}
    angle = geometry.get("angle_from_frontal_normal_deg")
    angle_text = "unknown" if angle is None else f"{float(angle):.1f} deg"
    primary = subject_label or clip["subject_id"]
    lines = [
        f"{primary} {clip['exercise']} | {clip['camera_id']}",
        f"{clip['view']} | {angle_text} | {clip.get('view_source', '')}",
    ]
    for index, text in enumerate(lines):
        cv2.putText(
            tile,
            text,
            (8, TILE_SIZE + 17 + index * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return tile


def group_clips(clips: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for clip in clips:
        grouped[(clip["subject_id"], clip["exercise"])].append(clip)
    for values in grouped.values():
        values.sort(key=lambda item: manifest_builder.CAMERA_IDS.index(item["camera_id"]))
    return dict(grouped)


def write_group_sheet(
    group: list[dict[str, Any]],
    frame_index: int,
    output_path: Path,
) -> None:
    tiles = []
    for clip in group:
        frame = read_frame(Path(clip["video_path"]), frame_index)
        tiles.append(make_tile(clip, frame))
    sheet = np.hstack(tiles)
    title = (
        f"{group[0]['subject_id']} {group[0]['exercise']} | "
        f"4 cameras | frame {frame_index}"
    )
    canvas = np.full((sheet.shape[0] + TITLE_HEIGHT, sheet.shape[1], 3), 16, dtype=np.uint8)
    canvas[TITLE_HEIGHT:] = sheet
    cv2.putText(
        canvas,
        title,
        (14, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def write_exercise_sheet(
    exercise: str,
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    frame_offsets: dict[str, int],
    output_path: Path,
) -> None:
    subjects = sorted(
        subject for subject, group_exercise in grouped if group_exercise == exercise
    )
    rows = []
    for subject in subjects:
        group = grouped[(subject, exercise)]
        frame_index = frame_offsets[group[0]["clip_id"]]
        tiles = [
            make_tile(clip, read_frame(Path(clip["video_path"]), frame_index), subject)
            for clip in group
        ]
        rows.append(np.hstack(tiles))
    sheet = np.vstack(rows)
    canvas = np.full((sheet.shape[0] + TITLE_HEIGHT, sheet.shape[1], 3), 16, dtype=np.uint8)
    canvas[TITLE_HEIGHT:] = sheet
    cv2.putText(
        canvas,
        f"Fit3D {exercise} | rows=subjects | columns=cameras",
        (14, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def angle_range(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min_deg": 0.0, "median_deg": 0.0, "max_deg": 0.0}
    return {
        "min_deg": round(min(values), 3),
        "median_deg": round(float(np.median(values)), 3),
        "max_deg": round(max(values), 3),
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    overrides = manifest_builder.load_view_overrides(args.overrides)
    train = [clip for clip in manifest["clips"] if clip.get("has_gt")]
    grouped = group_clips(train)
    sheet_dir = args.output_dir / "contact_sheets"
    frame_offsets: dict[str, int] = {}
    rows = []

    for (subject, exercise), group in sorted(grouped.items()):
        frame_index = audit_frame(group[0], args.sheet_frame_offset)
        for clip in group:
            frame_offsets[clip["clip_id"]] = frame_index
        sheet_path = sheet_dir / f"{subject}_{exercise}.jpg"
        write_group_sheet(group, frame_index, sheet_path)
        for clip in group:
            manual_view = overrides.get(clip["clip_id"], "")
            rows.append(
                {
                    "clip_id": clip["clip_id"],
                    "subject_id": clip["subject_id"],
                    "exercise": clip["exercise"],
                    "action_variant": clip["action_variant"],
                    "split": clip["split"],
                    "camera_id": clip["camera_id"],
                    "auto_view": clip["auto_view"],
                    "final_view": clip["view"],
                    "view_angle_deg": clip["view_geometry"]["angle_from_frontal_normal_deg"],
                    "view_angle_iqr_deg": clip["view_geometry"]["angle_iqr_deg"],
                    "front_rear": clip["view_geometry"]["front_rear"],
                    "camera_side": clip["view_geometry"]["camera_side"],
                    "projection_in_frame_rate": clip["quality"]["projection_in_frame_rate"],
                    "manual_view": manual_view,
                    "manual_match": "" if not manual_view else str(manual_view == clip["view"]).lower(),
                    "manual_review_status": "reviewed" if manual_view else "pending",
                    "contact_sheet": str(sheet_path.relative_to(args.output_dir)),
                }
            )

    for exercise in sorted({key[1] for key in grouped}):
        write_exercise_sheet(
            exercise,
            grouped,
            frame_offsets,
            sheet_dir / f"all_{exercise}_views.jpg",
        )

    fieldnames = list(rows[0])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "view_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    view_counts = Counter((clip["exercise"], clip["view"]) for clip in train)
    angles: defaultdict[str, list[float]] = defaultdict(list)
    for clip in train:
        angles[clip["exercise"]].append(
            float(clip["view_geometry"]["angle_from_frontal_normal_deg"])
        )
    reviewed = sum(1 for row in rows if row["manual_review_status"] == "reviewed")
    matches = sum(row["manual_match"] == "true" for row in rows)
    summary = {
        "schema_version": "task1-5a-view-audit-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "clip_count": len(train),
        "action_instance_count": len(grouped),
        "view_counts_by_exercise": {
            exercise: {
                view: count
                for (item_exercise, view), count in sorted(view_counts.items())
                if item_exercise == exercise
            }
            for exercise in sorted(angles)
        },
        "angle_ranges_by_exercise": {
            exercise: angle_range(values) for exercise, values in sorted(angles.items())
        },
        "manual_review": {
            "reviewed_clips": reviewed,
            "pending_clips": len(rows) - reviewed,
            "matches": matches,
        },
        "contact_sheets": {
            "groups": sorted(str(path.relative_to(args.output_dir)) for path in sheet_dir.glob("*.jpg") if not path.name.startswith("all_")),
            "combined": sorted(str(path.relative_to(args.output_dir)) for path in sheet_dir.glob("all_*.jpg")),
        },
    }
    write_json(args.output_dir / "view_audit_summary.json", summary)
    print(
        f"wrote {args.output_dir / 'view_audit.csv'}; "
        f"groups={len(grouped)} clips={len(train)} reviewed={reviewed}"
    )


if __name__ == "__main__":
    main()
