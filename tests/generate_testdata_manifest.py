#!/usr/bin/env python3
"""为 rawdata/testdata 下的视频生成 task1-4 格式的测试清单。

文件名需遵循：
    {squat|pushup}_{front|side}{index}[_{condition}]_{rotation}.mp4

脚本用 OpenCV 探测视频元数据；对侧视片段，额外用 MediaPipe 姿态可见度
在抽样帧上判定 side_left / side_right。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_task1_4_manifest import classify_view, pose_view_features  # noqa: E402
DEFAULT_INPUT_DIR = ROOT / "rawdata" / "testdata"
DEFAULT_OUTPUT = ROOT / "results" / "testdata" / "testdata_clips.json"
DEFAULT_MODEL = ROOT / "models" / "pose_landmarker_full.task"

BASE_METRICS = ("rep_duration_s", "rep_duration_cv", "spine_stability_bottom")
SQUAT_SIDE_METRICS = (
    "knee_flexion_rom",
    "bottom_knee_angle",
    "hip_flexion_rom",
    "torso_lean_bottom",
    "hip_depth_ratio",
)
SQUAT_FRONTAL_METRICS = (
    "knee_valgus_bottom",
    "knee_asymmetry_bottom",
    "hip_asymmetry_bottom",
)
PUSHUP_SIDE_METRICS = (
    "elbow_flexion_rom",
    "bottom_elbow_angle",
    "body_line_angle",
    "hip_sag_ratio",
)
PUSHUP_FRONTAL_METRICS = ("elbow_asymmetry_bottom", "shoulder_asymmetry_bottom")
EXERCISES = {"squat": ("squats", SQUAT_SIDE_METRICS, SQUAT_FRONTAL_METRICS), "pushup": ("pushups", PUSHUP_SIDE_METRICS, PUSHUP_FRONTAL_METRICS)}
SIDE_VIEWS = {"side_left", "side_right", "oblique_left", "oblique_right"}
FRONTAL_VIEWS = {"front", "rear", "oblique_left", "oblique_right"}
SIDE_JOINT_INDICES = {
    "left": (23, 25, 27),
    "right": (24, 26, 28),
}
CONDITION_NOTES = {
    "two": "two_person",
    "ubstable": "unstable_camera",
    "intrefered": "interference",
    "wrong": "wrong_form",
}
NAME_RE = re.compile(r"^(?P<view>front|side)(?P<index>\d+)$")


def metrics_for_view(exercise: str, view: str) -> tuple[list[str], list[str]]:
    _, side_metrics, frontal_metrics = EXERCISES[exercise]
    usable = list(BASE_METRICS)
    if view in SIDE_VIEWS:
        usable.extend(side_metrics)
    if view in FRONTAL_VIEWS:
        usable.extend(frontal_metrics)
    disabled = [m for m in (*side_metrics, *frontal_metrics) if m not in usable]
    return usable, disabled


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"invalid fps/frame_count: {path.name}")
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


def view_analysis(path: Path, meta: dict, model: Path, sample_count: int = 24) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, frame_count // sample_count)
        values: dict[str, list[float]] = {"left": [], "right": []}
        features: list[dict[str, float]] = []
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
            frame_index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % step != 0:
                    frame_index += 1
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                result = landmarker.detect(image)
                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    for side, indices in SIDE_JOINT_INDICES.items():
                        values[side].append(float(min(landmarks[i].visibility for i in indices)))
                    item = pose_view_features(result, meta["width"], meta["height"])
                    if item is not None:
                        features.append(item)
                frame_index += 1
    finally:
        cap.release()

    scores = {
        side: (float(np.median(items)) if items else 0.0)
        for side, items in values.items()
    }
    classification = classify_view(features)
    return {
        "scores": scores,
        "detections": max(len(values["left"]), len(values["right"])),
        "classification": classification,
    }


def parse_name(path: Path) -> dict:
    parts = path.stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"unexpected filename: {path.name}")
    exercise = parts[0]
    match = NAME_RE.fullmatch(parts[1])
    if not match:
        raise ValueError(f"unexpected view token: {path.name}")
    rotation = parts[-1]
    conditions = parts[2:-1]
    if exercise not in EXERCISES:
        raise ValueError(f"unsupported exercise: {path.name}")
    if rotation != "none":
        raise ValueError(f"unsupported rotation: {path.name}")
    return {
        "exercise": exercise,
        "view_kind": match.group("view"),
        "index": int(match.group("index")),
        "conditions": conditions,
    }


def build_entry(path: Path, meta: dict, args: argparse.Namespace) -> dict:
    parsed = parse_name(path)
    side_scores: dict[str, float] = {}
    side_detections = 0
    classification: dict = {"view": "unknown", "confidence": 0.0}
    auto_view = "unknown"
    if parsed["view_kind"] == "front":
        view = "front"
        view_source = "filename"
        if not args.no_side_detect:
            analysis = view_analysis(path, meta, args.model)
            classification = analysis["classification"]
            auto_view = classification["view"]
            side_scores = analysis["scores"]
            side_detections = analysis["detections"]
    else:
        if args.force_side:
            side = args.force_side
            view = f"side_{side}"
            view_source = "forced"
        elif args.no_side_detect:
            view = "side_right"
            view_source = "default_without_detection"
        else:
            analysis = view_analysis(path, meta, args.model)
            classification = analysis["classification"]
            auto_view = classification["view"]
            side_scores = analysis["scores"]
            side_detections = analysis["detections"]
            if auto_view in SIDE_VIEWS or auto_view in {"front", "rear", "oblique_left", "oblique_right"}:
                view = auto_view
                view_source = "auto_pose_view"
            else:
                side = "left" if side_scores.get("left", 0.0) >= side_scores.get("right", 0.0) else "right"
                view = f"side_{side}"
                view_source = "auto_side_visibility"

    usable, disabled = metrics_for_view(parsed["exercise"], view)
    action_display = EXERCISES[parsed["exercise"]][0]
    conditions = [CONDITION_NOTES.get(c, c) for c in parsed["conditions"]]
    notes = ["testdata auto-generated"]
    if conditions:
        notes.append("condition=" + "+".join(conditions))
    if parsed["view_kind"] == "side" and view == "front":
        notes.append("filename_side_but_pose_frontal")
    if side_scores:
        notes.append(
            "side_visibility="
            + ",".join(f"{k}:{v:.3f}" for k, v in sorted(side_scores.items()))
        )
        notes.append(f"side_samples={side_detections}")
    if classification.get("width_ratio") is not None:
        notes.append(
            "pose_view="
            f"{auto_view}@{float(classification.get('confidence') or 0.0):.3f}"
        )
        notes.append(f"width_ratio={float(classification['width_ratio']):.3f}")

    return {
        "clip_id": path.stem,
        "video_id": path.stem,
        "video_path": str(path.resolve()),
        "exercise": parsed["exercise"],
        "action_display": action_display,
        "trim": {"start_frame": 0, "end_frame": meta["frame_count"] - 1},
        "fps": meta["fps"],
        "frame_count": meta["frame_count"],
        "source_size": {"width": meta["width"], "height": meta["height"]},
        "rotation": "none",
        "view": view,
        "auto_view": auto_view,
        "view_source": view_source,
        "split": "testdata",
        "phase_role": "testdata",
        "selected_for_phase": view in SIDE_VIEWS,
        "usable_metrics": usable,
        "disabled_metrics": disabled,
        "quality": {},
        "notes": "; ".join(notes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", default="testdata")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--only", default="", help="comma-separated filename stems or prefixes")
    parser.add_argument("--force-side", choices=("left", "right"), default="")
    parser.add_argument("--no-side-detect", action="store_true")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"input dir not found: {args.input_dir}")
    if args.force_side:
        args.no_side_detect = False
    if not args.no_side_detect and not args.force_side and not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")

    paths = sorted(p for p in args.input_dir.glob("*.mp4") if p.is_file())
    if args.only:
        prefixes = tuple(x.strip() for x in args.only.split(",") if x.strip())
        paths = [p for p in paths if p.stem.startswith(prefixes)]
    if not paths:
        raise SystemExit("no matching videos")

    clips = []
    for path in paths:
        meta = probe_video(path)
        entry = build_entry(path, meta, args)
        clips.append(entry)
        print(
            f"{entry['clip_id']}: view={entry['view']} "
            f"fps={meta['fps']:.3f} frames={meta['frame_count']} "
            f"size={meta['width']}x{meta['height']} source={entry['view_source']}"
        )

    payload = {
        "schema_version": "task1-4-v1",
        "task": "task1-4a-testdata",
        "dataset": args.dataset,
        "clips": clips,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(clips)} clips -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
