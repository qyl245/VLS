#!/usr/bin/env python3
"""Extract image-space pose signals to help locate blind annotation phases."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTION_SEGMENTS = ROOT / "results" / "segmentation" / "action_segments.csv"
DEFAULT_VIDEOS = ROOT.parent / "dataset" / "QEVD-FIT" / "QEVD-FIT-COACH" / "long_range_videos"
DEFAULT_MODEL = ROOT / "models" / "pose_landmarker_full.task"
DEFAULT_OUTPUT = ROOT / "results" / "segmentation" / "annotation_blind" / "motion_signals.csv"

DEFAULT_SEGMENTS = (
    ("0161", "pushup"),
    ("0173", "boxing_squat_punches"),
    ("0161", "lunge_jumps"),
    ("0218", "walking_lunges"),
    ("0218", "air_jump_rope"),
    ("0162", "mountain_climbers"),
)

FIELDS = [
    "segment_id",
    "video_id",
    "action",
    "frame_id",
    "head_y",
    "shoulder_y",
    "hip_y",
    "left_ankle_y",
    "right_ankle_y",
    "left_foot_y",
    "right_foot_y",
    "left_wrist_y",
    "right_wrist_y",
    "left_knee_y",
    "right_knee_y",
    "hip_x",
    "left_ankle_x",
    "right_ankle_x",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-segments", type=Path, default=DEFAULT_ACTION_SEGMENTS)
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--segments", nargs="*", default=None)
    args = parser.parse_args()
    if args.segments:
        selected = []
        for item in args.segments:
            if ":" not in item:
                parser.error("--segments entries must use video_id:action")
            video_id, action = item.split(":", 1)
            selected.append((video_id.zfill(4), action))
        args.selected_segments = selected
    else:
        args.selected_segments = list(DEFAULT_SEGMENTS)
    return args


def read_windows(path: Path) -> dict[tuple[str, str], tuple[int, int]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["video_id"].zfill(4), row["action"]): (
            int(row["start_frame"]),
            int(row["end_frame"]),
        )
        for row in rows
    }


def make_image(frame: np.ndarray) -> Any:
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def landmark_values(landmarks: list[Any]) -> dict[str, float]:
    def point(index: int) -> Any:
        return landmarks[index]

    def mean(values: list[float]) -> float:
        return float(np.mean(values))

    return {
        "head_y": float(point(0).y),
        "shoulder_y": mean([point(11).y, point(12).y]),
        "hip_y": mean([point(23).y, point(24).y]),
        "left_ankle_y": float(point(27).y),
        "right_ankle_y": float(point(28).y),
        "left_foot_y": float(point(31).y),
        "right_foot_y": float(point(32).y),
        "left_wrist_y": float(point(15).y),
        "right_wrist_y": float(point(16).y),
        "left_knee_y": float(point(25).y),
        "right_knee_y": float(point(26).y),
        "hip_x": mean([point(23).x, point(24).x]),
        "left_ankle_x": float(point(27).x),
        "right_ankle_x": float(point(28).x),
    }


def main() -> int:
    args = parse_args()
    windows = read_windows(args.action_segments)
    selected_by_video: dict[str, list[tuple[str, int, int]]] = {}
    for video_id, action in args.selected_segments:
        key = (video_id, action)
        if key not in windows:
            raise ValueError(f"segment not found: {video_id}/{action}")
        start, end = windows[key]
        selected_by_video.setdefault(video_id, []).append((action, start, end))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for video_id, segments in selected_by_video.items():
            video_path = args.videos_dir / f"{video_id}.mp4"
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise RuntimeError(f"cannot open {video_path}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=1,
            )
            landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
            try:
                for action, start, end in segments:
                    segment_id = f"{video_id}_{action}"
                    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
                    for frame_id in range(start, end + 1):
                        ok, frame = capture.read()
                        if not ok:
                            raise RuntimeError(f"cannot decode {video_id} frame {frame_id}")
                        timestamp_ms = int(round(frame_id * 1000.0 / fps))
                        result = landmarker.detect_for_video(make_image(frame), timestamp_ms)
                        if not result.pose_landmarks:
                            continue
                        row = {
                            "segment_id": segment_id,
                            "video_id": video_id,
                            "action": action,
                            "frame_id": frame_id,
                        }
                        row.update(landmark_values(result.pose_landmarks[0]))
                        writer.writerow(row)
                    print(f"processed {segment_id}: {start}-{end}")
            finally:
                landmarker.close()
                capture.release()
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
