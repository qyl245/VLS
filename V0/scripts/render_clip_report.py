#!/usr/bin/env python3
"""Render annotated evidence frames and a static task1-4 clip report."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEEDBACK_DIR = ROOT / "results" / "feedback_v1"
DEFAULT_ANALYSIS_DIR = ROOT / "results" / "clip_analysis_v1"
DEFAULT_TEMPLATES = ROOT / "configs" / "task1_4_feedback_templates.json"
DEFAULT_MODEL = ROOT / "models" / "pose_landmarker_full.task"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports_v1"

REPORT_SCHEMA_VERSION = "task1-4-report-v1"
SUMMARY_SCHEMA_VERSION = "task1-4-report-summary-v1"

ROTATION_CODES = {
    "none": None,
    "90cw": cv2.ROTATE_90_CLOCKWISE,
    "90ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180": cv2.ROTATE_180,
}

JOINT_INDEX = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

SKELETON_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
)

COLOR_SKELETON = (86, 220, 120)
COLOR_JOINT = (40, 205, 245)
COLOR_RED = (42, 42, 238)
COLOR_ORANGE = (36, 150, 249)
COLOR_GREY = (150, 150, 150)
COLOR_WHITE = (245, 245, 245)
COLOR_DARK = (18, 24, 38)

SEVERITY_LABELS = {"high": "高", "medium": "中", "low": "低", "none": "无"}
SEVERITY_COLORS = {"high": "#dc2626", "medium": "#ea580c", "low": "#ca8a04", "none": "#64748b"}
EXERCISE_LABELS = {"squat": "深蹲", "pushup": "俯卧撑"}
POSE_KEY_JOINTS = {
    "pushup": (
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
    ),
    "squat": (
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ),
}
MIN_KEY_JOINT_VISIBILITY = 0.5
MIN_PUSHUP_SHOULDER_WRIST_SPAN = 0.05
MIN_SQUAT_SEGMENT_SPAN = 0.03
VIEW_LABELS = {
    "front": "正面",
    "rear": "背面",
    "side_left": "左侧",
    "side_right": "右侧",
    "oblique_left": "左斜侧",
    "oblique_right": "右斜侧",
    "unknown": "未知",
}
REASON_LABELS = {
    "view_unsupported": "当前视角不支持",
    "insufficient_reps": "完整 rep 数不足",
    "low_confidence": "置信度不足",
    "metric_unavailable": "指标不可用",
    "quality_gate_failed": "质量门控失败",
}

QUALITY_WARNING_LABELS = {
    "exercise_posture_reps_rejected": "已剔除不符合动作姿态的热身/起身 rep",
    "no_complete_reps": "没有可评估的完整有效 rep",
    "fewer_than_three_complete_reps": "完整有效 rep 少于 3 个，组级指标不评估",
    "pose_detection_rate_below_pass_threshold": "姿态检出率低于通过阈值",
    "key_visibility_q10_below_pass_threshold": "关键关节可见性偏低",
    "body_in_frame_rate_below_pass_threshold": "人体完整入镜率偏低",
}

FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-dir", type=Path, default=DEFAULT_FEEDBACK_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clip-id", nargs="+", default=None)
    parser.add_argument("--split", default="dev", help="Use 'all' to disable split filtering.")
    parser.add_argument(
        "--skip-frames",
        action="store_true",
        help="Regenerate HTML only and reuse existing annotated frames.",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Regenerate HTML only and reuse existing browser-compatible clip videos.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    minutes = int(value // 60)
    seconds = value - minutes * 60
    return f"{minutes:02d}:{seconds:04.1f}"


def clip_origin_s(analysis: dict[str, Any]) -> float:
    return float(analysis["trim"]["start_frame"]) / float(analysis["fps"])


def clip_duration_s(analysis: dict[str, Any]) -> float:
    trim = analysis["trim"]
    return (int(trim["end_frame"]) - int(trim["start_frame"]) + 1) / float(analysis["fps"])


def clip_relative_s(absolute_s: float, analysis: dict[str, Any]) -> float:
    return max(0.0, float(absolute_s) - clip_origin_s(analysis))


def load_bundle(clip_id: str, feedback_dir: Path, analysis_dir: Path) -> dict[str, Any]:
    issue_path = feedback_dir / clip_id / "issues.json"
    feature_path = feedback_dir / clip_id / "features.csv"
    analysis_path = analysis_dir / clip_id / "analysis.json"
    clip_path = analysis_dir / clip_id / "clip.json"
    missing = [str(path) for path in (issue_path, feature_path, analysis_path, clip_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing report input: " + ", ".join(missing))
    issue_output = read_json(issue_path)
    analysis = read_json(analysis_path)
    clip = read_json(clip_path)
    features = read_csv(feature_path)
    if issue_output.get("clip_id") != clip_id or analysis.get("clip_id") != clip_id:
        raise ValueError(f"clip_id mismatch for {clip_id}")
    wide = {}
    wide_path = analysis_dir / clip_id / "angles" / "angles_wide.csv"
    if wide_path.exists():
        for row in read_csv(wide_path):
            wide[int(row["frame_id"])] = row
    return {
        "clip_id": clip_id,
        "issue_output": issue_output,
        "analysis": analysis,
        "clip": clip,
        "features": features,
        "wide": wide,
        "feedback_dir": feedback_dir / clip_id,
        "analysis_dir": analysis_dir / clip_id,
        "video_path": Path(clip["video_path"]),
    }


def select_clip_ids(args: argparse.Namespace) -> list[str]:
    available = {}
    for path in sorted(args.feedback_dir.iterdir()):
        issue_path = path / "issues.json"
        analysis_path = args.analysis_dir / path.name / "analysis.json"
        if not path.is_dir() or not issue_path.exists() or not analysis_path.exists():
            continue
        analysis = read_json(analysis_path)
        if args.split != "all" and analysis.get("split") != args.split:
            continue
        available[path.name] = True
    if args.clip_id:
        missing = sorted(set(args.clip_id) - set(available))
        if missing:
            raise ValueError("unknown or filtered clip_id: " + ", ".join(missing))
        return list(args.clip_id)
    return sorted(available)


def rotate_frame(frame: np.ndarray, rotation: str) -> np.ndarray:
    code = ROTATION_CODES.get(rotation)
    if code is None:
        return frame
    return cv2.rotate(frame, code)


def prune_stale_annotated_frames(frame_dir: Path, keep_names: set[str]) -> int:
    """Remove evidence frames left behind by suppressed or renumbered issues."""
    removed = 0
    if not frame_dir.exists():
        return removed
    for path in frame_dir.glob("*.png"):
        if path.name not in keep_names:
            path.unlink()
            removed += 1
    return removed


class VideoFrameReader:
    def __init__(self, video_path: Path, rotation: str):
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        self.rotation = rotation

    def read(self, source_frame: int) -> np.ndarray:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(source_frame))
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"cannot read source frame {source_frame}")
        return rotate_frame(frame, self.rotation)

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> "VideoFrameReader":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def build_web_video(bundle: dict[str, Any], output_dir: Path) -> Path:
    """Encode the clip window to a browser-compatible VP8/WebM proxy."""
    analysis = bundle["analysis"]
    trim = analysis["trim"]
    start = int(trim["start_frame"])
    end = int(trim["end_frame"])
    fps = float(analysis["fps"])
    rotation = analysis.get("rotation", "none")
    target = output_dir / "media" / f"{bundle['clip_id']}_clip.webm"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.tmp.webm")

    cap = cv2.VideoCapture(str(bundle["video_path"]))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {bundle['video_path']}")
    writer = None
    written = 0
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _source_frame in range(start, end + 1):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(
                    f"cannot read video frame in [{start}, {end}]: {bundle['video_path']}"
                )
            frame = rotate_frame(frame, rotation)
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(temporary),
                    cv2.VideoWriter_fourcc(*"VP80"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError("OpenCV VP8/WebM encoder is unavailable")
                if hasattr(cv2, "VIDEOWRITER_PROP_QUALITY"):
                    writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 92)
            writer.write(frame)
            written += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if written == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"no frames written for {bundle['clip_id']}")
    temporary.replace(target)
    return target


def create_landmarker(model_path: Path) -> Any:
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def normalized_landmarks_from_result(result: Any) -> list[dict[str, float]] | None:
    if not result.pose_landmarks:
        return None
    return [
        {
            "x": float(point.x),
            "y": float(point.y),
            "z": float(point.z),
            "visibility": float(point.visibility),
        }
        for point in result.pose_landmarks[0]
    ]


def make_mediapipe_image(frame: np.ndarray) -> Any:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


def detect_normalized_landmarks(landmarker: Any, frame: np.ndarray) -> list[dict[str, float]] | None:
    return normalized_landmarks_from_result(landmarker.detect(make_mediapipe_image(frame)))


def detect_video_tracked_landmarks(
    video_path: Path,
    rotation: str,
    target_frame: int,
    fps: float,
    model_path: Path,
    warmup_frames: int = 30,
) -> list[dict[str, float]] | None:
    start = max(0, target_frame - warmup_frames)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for source_frame in range(start, target_frame + 1):
            ok, frame = cap.read()
            if not ok:
                return None
            rotated = rotate_frame(frame, rotation)
            timestamp_ms = int(round((source_frame - start) * 1000.0 / fps)) if fps > 0 else source_frame - start
            result = landmarker.detect_for_video(make_mediapipe_image(rotated), timestamp_ms)
            if source_frame == target_frame:
                return normalized_landmarks_from_result(result)
    finally:
        cap.release()
        landmarker.close()
    return None


def normalized_to_pixels(landmarks: list[dict[str, float]], width: int, height: int) -> np.ndarray:
    return np.asarray(
        [[point["x"] * width, point["y"] * height] for point in landmarks],
        dtype=np.float64,
    )


def visibility_array(landmarks: list[dict[str, float]]) -> np.ndarray:
    return np.asarray([point["visibility"] for point in landmarks], dtype=np.float64)


def landmarks_are_plausible(
    landmarks: list[dict[str, float]] | None,
    exercise: str,
) -> bool:
    """Reject collapsed or low-visibility poses before drawing an overlay."""
    if not landmarks:
        return False
    names = POSE_KEY_JOINTS.get(exercise)
    if not names:
        return True
    for name in names:
        if float(landmarks[JOINT_INDEX[name]].get("visibility", 0.0)) < MIN_KEY_JOINT_VISIBILITY:
            return False
    if exercise == "pushup":
        for side in ("left", "right"):
            shoulder = landmarks[JOINT_INDEX[f"{side}_shoulder"]]
            wrist = landmarks[JOINT_INDEX[f"{side}_wrist"]]
            span = math.hypot(shoulder["x"] - wrist["x"], shoulder["y"] - wrist["y"])
            if span < MIN_PUSHUP_SHOULDER_WRIST_SPAN:
                return False
    elif exercise == "squat":
        for side in ("left", "right"):
            hip = landmarks[JOINT_INDEX[f"{side}_hip"]]
            knee = landmarks[JOINT_INDEX[f"{side}_knee"]]
            ankle = landmarks[JOINT_INDEX[f"{side}_ankle"]]
            if math.hypot(hip["x"] - knee["x"], hip["y"] - knee["y"]) < MIN_SQUAT_SEGMENT_SPAN:
                return False
            if math.hypot(knee["x"] - ankle["x"], knee["y"] - ankle["y"]) < MIN_SQUAT_SEGMENT_SPAN:
                return False
    return True


def draw_dashed_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash: int = 9,
    gap: int = 6,
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    distance = 0.0
    while distance < length:
        segment_end = min(distance + dash, length)
        p1 = (int(round(x1 + dx * distance)), int(round(y1 + dy * distance)))
        p2 = (int(round(x1 + dx * segment_end)), int(round(y1 + dy * segment_end)))
        cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)
        distance += dash + gap


def draw_arc(
    image: np.ndarray,
    center: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    radius: int,
    color: tuple[int, int, int],
    thickness: int = 3,
    dashed: bool = False,
) -> None:
    cx, cy = float(center[0]), float(center[1])
    angle_a = math.atan2(float(first[1]) - cy, float(first[0]) - cx)
    angle_b = math.atan2(float(second[1]) - cy, float(second[0]) - cx)
    delta = angle_b - angle_a
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    segments = 28
    for index in range(segments):
        if dashed and index % 2:
            continue
        t1 = index / segments
        t2 = (index + 1) / segments
        a1 = angle_a + delta * t1
        a2 = angle_a + delta * t2
        p1 = (int(round(cx + radius * math.cos(a1))), int(round(cy + radius * math.sin(a1))))
        p2 = (int(round(cx + radius * math.cos(a2))), int(round(cy + radius * math.sin(a2))))
        cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)


def draw_skeleton(
    image: np.ndarray,
    points: np.ndarray,
    visibility: np.ndarray,
    highlight_joints: Iterable[str],
    visibility_threshold: float = 0.5,
) -> None:
    highlight_indices = {JOINT_INDEX[name] for name in highlight_joints if name in JOINT_INDEX}
    low_overlay = image.copy()
    for a, b in SKELETON_CONNECTIONS:
        if a >= len(points) or b >= len(points):
            continue
        visible = visibility[a] >= visibility_threshold and visibility[b] >= visibility_threshold
        p1 = tuple(np.round(points[a]).astype(int))
        p2 = tuple(np.round(points[b]).astype(int))
        if visible:
            cv2.line(image, p1, p2, COLOR_SKELETON, 3, cv2.LINE_AA)
        else:
            draw_dashed_line(image, p1, p2, COLOR_GREY, 2)
    for index, point in enumerate(points):
        center = tuple(np.round(point).astype(int))
        if visibility[index] < visibility_threshold:
            cv2.circle(low_overlay, center, 9, COLOR_GREY, -1, cv2.LINE_AA)
            cv2.circle(image, center, 4, COLOR_GREY, -1, cv2.LINE_AA)
        elif index in highlight_indices:
            cv2.circle(image, center, 10, COLOR_RED, 2, cv2.LINE_AA)
            cv2.circle(image, center, 6, COLOR_RED, -1, cv2.LINE_AA)
        else:
            cv2.circle(image, center, 3, COLOR_JOINT, -1, cv2.LINE_AA)
    if np.any(visibility < visibility_threshold):
        cv2.addWeighted(low_overlay, 0.28, image, 0.72, 0.0, dst=image)


def project_to_line(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    vector = end - start
    denominator = float(np.dot(vector, vector))
    if denominator < 1e-8:
        return start.copy()
    t = float(np.dot(point - start, vector) / denominator)
    return start + t * vector


def draw_arrow(image: np.ndarray, start: np.ndarray, end: np.ndarray, color: tuple[int, int, int]) -> None:
    p1 = tuple(np.round(start).astype(int))
    p2 = tuple(np.round(end).astype(int))
    cv2.arrowedLine(image, p1, p2, color, 3, cv2.LINE_AA, tipLength=0.25)


def draw_stability_halo(image: np.ndarray, torso_center: np.ndarray, size: float) -> None:
    overlay = image.copy()
    center = tuple(np.round(torso_center).astype(int))
    axes = (max(20, int(size * 0.33)), max(30, int(size * 0.52)))
    cv2.ellipse(overlay, center, axes, 0, 0, 360, COLOR_RED, 10, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.30, image, 0.70, 0.0, dst=image)


def draw_issue_geometry(
    image: np.ndarray,
    points: np.ndarray,
    visibility: np.ndarray,
    issue: dict[str, Any],
    wide_row: dict[str, str],
) -> None:
    code = issue["issue_code"]
    primary = issue.get("measurements", {}).get("primary_metric", "")
    radius = max(24, min(44, int(min(image.shape[:2]) * 0.08)))

    if code == "squat_knee_valgus":
        for side in ("left", "right"):
            hip = points[JOINT_INDEX[f"{side}_hip"]]
            knee = points[JOINT_INDEX[f"{side}_knee"]]
            ankle = points[JOINT_INDEX[f"{side}_ankle"]]
            if min(
                visibility[JOINT_INDEX[f"{side}_hip"]],
                visibility[JOINT_INDEX[f"{side}_knee"]],
                visibility[JOINT_INDEX[f"{side}_ankle"]],
            ) < 0.5:
                continue
            draw_dashed_line(
                image,
                tuple(np.round(hip).astype(int)),
                tuple(np.round(ankle).astype(int)),
                COLOR_JOINT,
                2,
            )
            projection = project_to_line(knee, hip, ankle)
            if np.linalg.norm(projection - knee) > 4:
                draw_arrow(image, knee, projection, COLOR_RED)
            draw_arc(image, knee, hip, ankle, radius, COLOR_RED, 3)
            draw_arc(image, knee, hip, ankle, radius + 9, COLOR_JOINT, 2, dashed=True)
        return

    if primary.startswith("knee_asymmetry"):
        for side in ("left", "right"):
            hip = points[JOINT_INDEX[f"{side}_hip"]]
            knee = points[JOINT_INDEX[f"{side}_knee"]]
            ankle = points[JOINT_INDEX[f"{side}_ankle"]]
            draw_arc(image, knee, hip, ankle, radius, COLOR_RED, 3)
        return

    if primary.startswith("hip_asymmetry"):
        for side in ("left", "right"):
            shoulder = points[JOINT_INDEX[f"{side}_shoulder"]]
            hip = points[JOINT_INDEX[f"{side}_hip"]]
            knee = points[JOINT_INDEX[f"{side}_knee"]]
            draw_arc(image, hip, shoulder, knee, radius, COLOR_RED, 3)
        return

    if primary.startswith("elbow_asymmetry"):
        for side in ("left", "right"):
            shoulder = points[JOINT_INDEX[f"{side}_shoulder"]]
            elbow = points[JOINT_INDEX[f"{side}_elbow"]]
            wrist = points[JOINT_INDEX[f"{side}_wrist"]]
            draw_arc(image, elbow, shoulder, wrist, radius, COLOR_RED, 3)
            value = to_float(wide_row.get(f"{side}_elbow_flexion"))
            if value is not None:
                label_point = tuple(np.round(elbow + np.array([8, -12])).astype(int))
                cv2.putText(
                    image,
                    f"{value:.1f}",
                    label_point,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    COLOR_RED,
                    2,
                    cv2.LINE_AA,
                )
        return

    if primary.startswith("shoulder_asymmetry"):
        for side in ("left", "right"):
            hip = points[JOINT_INDEX[f"{side}_hip"]]
            shoulder = points[JOINT_INDEX[f"{side}_shoulder"]]
            elbow = points[JOINT_INDEX[f"{side}_elbow"]]
            draw_arc(image, shoulder, hip, elbow, radius, COLOR_RED, 3)
            value = to_float(wide_row.get(f"{side}_shoulder_flexion"))
            if value is not None:
                label_point = tuple(np.round(shoulder + np.array([8, -12])).astype(int))
                cv2.putText(
                    image,
                    f"{value:.1f}",
                    label_point,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    COLOR_RED,
                    2,
                    cv2.LINE_AA,
                )
        return

    if code.endswith("_stability_low"):
        shoulder_mid = (points[11] + points[12]) / 2.0
        hip_mid = (points[23] + points[24]) / 2.0
        cv2.line(
            image,
            tuple(np.round(shoulder_mid).astype(int)),
            tuple(np.round(hip_mid).astype(int)),
            COLOR_RED,
            4,
            cv2.LINE_AA,
        )
        draw_stability_halo(image, (shoulder_mid + hip_mid) / 2.0, np.linalg.norm(shoulder_mid - hip_mid))


def find_font_path() -> str:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return ""


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = find_font_path()
    if path:
        return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    if not text:
        return []
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines = []
    current = ""
    leading_punctuation = set("，。！？；：、）》】")
    for char in text:
        if not current and lines and char in leading_punctuation:
            lines[-1] += char
            continue
        candidate = current + char
        box = draw.textbbox((0, 0), candidate, font=font)
        if box[2] - box[0] <= max_width or not current:
            current = candidate
        else:
            if char in leading_punctuation:
                lines.append(current + char)
                current = ""
            else:
                lines.append(current)
                current = char
    if current:
        lines.append(current)
    return lines


def draw_text_panel(
    image: np.ndarray,
    issue: dict[str, Any],
    time_offset_s: float = 0.0,
) -> np.ndarray:
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    title_size = max(15, min(28, width // 24))
    body_size = max(12, min(19, width // 34))
    title_font = load_font(title_size)
    body_font = load_font(body_size)
    padding = max(10, width // 42)
    panel_width = width - 2 * padding
    metric = issue.get("measurements", {})
    primary_metric = metric.get("primary_metric", "")
    value = metric.get("primary_value")
    unit = metric.get("unit", "")
    thresholds = metric.get("thresholds", {})
    severity = issue.get("severity", "none")
    threshold = to_float(thresholds.get(severity)) if severity in thresholds else None
    value_text = format_metric_value(value, unit)
    threshold_text = format_metric_value(threshold, unit)
    detail_lines = wrap_text(issue["text"].get("detail", ""), body_font, panel_width - 2 * padding)
    cue_lines = wrap_text("提示：" + issue["text"].get("cue", ""), body_font, panel_width - 2 * padding)
    lines = []
    lines.append((issue["text"].get("title", issue["issue_code"]), title_font, COLOR_WHITE))
    lines.extend((line, body_font, COLOR_WHITE) for line in detail_lines[:3])
    meta_line = (
        f"rep {issue.get('representative_rep_id')} · frame {issue.get('anchor_frame')} · "
        f"{fmt_seconds(max(0.0, float(issue.get('anchor_time_s') or 0.0) - time_offset_s))} · "
        f"conf {issue.get('confidence') or 0:.2f}"
    )
    lines.append((meta_line, body_font, (218, 226, 238)))
    if value is not None:
        lines.append((f"{primary_metric}: {value_text} / 阈值 {threshold_text}", body_font, (255, 210, 210)))

    heights = []
    for text, font, _ in lines:
        box = draw.textbbox((0, 0), text, font=font)
        heights.append(box[3] - box[1] + 5)
    panel_height = sum(heights) + 2 * padding
    outline_hex = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["none"]).lstrip("#")
    outline = tuple(int(outline_hex[index : index + 2], 16) for index in (0, 2, 4))
    draw.rounded_rectangle(
        (padding, padding, padding + panel_width, padding + panel_height),
        radius=10,
        fill=(15, 23, 42, 218),
        outline=(*outline, 255),
        width=3,
    )
    y = padding * 2
    for index, (text, font, color) in enumerate(lines):
        draw.text((padding * 2, y), text, font=font, fill=(*color, 255))
        y += heights[index]

    footer_height = padding * 2 + sum(
        draw.textbbox((0, 0), line, font=body_font)[3] - draw.textbbox((0, 0), line, font=body_font)[1] + 4
        for line in cue_lines[:3]
    )
    draw.rounded_rectangle(
        (padding, height - footer_height - padding, width - padding, height - padding),
        radius=10,
        fill=(15, 23, 42, 210),
    )
    y = height - footer_height
    for line in cue_lines[:3]:
        draw.text((padding * 2, y), line, font=body_font, fill=(236, 241, 248, 255))
        box = draw.textbbox((0, 0), line, font=body_font)
        y += box[3] - box[1] + 4
    composited = Image.alpha_composite(canvas, overlay).convert("RGB")
    return cv2.cvtColor(np.asarray(composited), cv2.COLOR_RGB2BGR)


def format_metric_value(value: float | None, unit: str) -> str:
    if value is None:
        return "-"
    if unit == "deg":
        return f"{value:.1f}°"
    if unit == "ratio":
        return f"{value:.2f}"
    if unit == "variance":
        return f"{value:.3f}"
    if unit == "s":
        return f"{value:.2f}s"
    return f"{value:.2f} {unit}".strip()


def add_zoom_inset(image: np.ndarray, points: np.ndarray, zoom: float) -> None:
    height, width = image.shape[:2]
    if width < 680 or height < 420:
        return
    inset_size = min(250, width // 3)
    center_x = float(np.median(points[:, 0]))
    center_y = float(np.median(points[:, 1]))
    crop_w = int(width / max(zoom, 1.0))
    crop_h = int(crop_w * 0.75)
    x1 = max(0, min(width - crop_w, int(center_x - crop_w / 2)))
    y1 = max(0, min(height - crop_h, int(center_y - crop_h / 2)))
    crop = image[y1 : y1 + crop_h, x1 : x1 + crop_w]
    if crop.size == 0:
        return
    inset = cv2.resize(crop, (inset_size, int(inset_size * 0.75)), interpolation=cv2.INTER_CUBIC)
    x = width - inset.shape[1] - 16
    y = 16
    cv2.rectangle(image, (x - 3, y - 3), (x + inset.shape[1] + 3, y + inset.shape[0] + 3), COLOR_WHITE, 2)
    image[y : y + inset.shape[0], x : x + inset.shape[1]] = inset
    cv2.putText(image, "ZOOM", (x + 6, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 2, cv2.LINE_AA)


def render_issue_frame(
    frame: np.ndarray,
    landmarks: list[dict[str, float]] | None,
    issue: dict[str, Any],
    wide_row: dict[str, str],
    time_offset_s: float = 0.0,
) -> tuple[np.ndarray, bool]:
    output = frame.copy()
    if landmarks:
        points = normalized_to_pixels(landmarks, output.shape[1], output.shape[0])
        visibility = visibility_array(landmarks)
        draw_skeleton(output, points, visibility, issue["visual_overlay"].get("highlight_joints", []))
        draw_issue_geometry(output, points, visibility, issue, wide_row)
        focus_indices = [
            JOINT_INDEX[name]
            for name in issue["visual_overlay"].get("highlight_joints", [])
            if name in JOINT_INDEX
        ]
        if focus_indices:
            add_zoom_inset(
                output,
                points[focus_indices],
                float(issue["visual_overlay"].get("camera_zoom", 1.0)),
            )
    output = draw_text_panel(output, issue, time_offset_s)
    return output, landmarks is not None


def timeline_position(absolute_s: float, analysis: dict[str, Any]) -> float:
    duration = clip_duration_s(analysis)
    if duration <= 0:
        return 0.0
    relative = absolute_s - clip_origin_s(analysis)
    return max(0.0, min(100.0, relative / duration * 100.0))


def html_escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def card_html(
    issue: dict[str, Any],
    image_path: str,
    featured: bool,
    fps: float,
    origin_s: float = 0.0,
) -> str:
    severity = issue.get("severity", "none")
    color = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["none"])
    label = SEVERITY_LABELS.get(severity, severity)
    confidence = issue.get("confidence")
    confidence_text = f"{confidence:.0%}" if confidence is not None else "-"
    threshold = issue.get("measurements", {}).get("thresholds", {}).get(severity)
    unit = issue.get("measurements", {}).get("unit", "")
    threshold_text = format_metric_value(to_float(threshold), unit)
    value_text = format_metric_value(to_float(issue.get("measurements", {}).get("primary_value")), unit)
    anchor_time = max(0.0, float(issue.get("anchor_time_s") or 0.0) - origin_s)
    evidence = issue.get("evidence_interval") or [None, None]
    evidence_start = max(0.0, (evidence[0] or 0) / fps - origin_s)
    evidence_end = max(0.0, (evidence[1] or 0) / fps - origin_s)
    featured_class = " featured" if featured else ""
    return f"""
    <article class="issue-card{featured_class}" data-issue-id="{html_escape(issue['issue_id'])}" style="--severity:{color}">
      <button class="card-hit" type="button" aria-label="回放 {html_escape(issue['text']['title'])}"
        onclick="selectIssue('{html_escape(issue['issue_id'])}', {anchor_time:.6f}, {evidence_start:.6f}, {evidence_end:.6f})"></button>
      <div class="issue-head">
        <span class="severity">{label}</span>
        <span class="issue-frequency">{html_escape(issue.get('frequency', '-'))}</span>
        {'<span class="default-tag">默认展示</span>' if featured else ''}
      </div>
      <h3>{html_escape(issue['text']['title'])}</h3>
      <p>{html_escape(issue['text']['detail'])}</p>
      <p class="cue">{html_escape(issue['text']['cue'])}</p>
      <img src="{html_escape(image_path)}" alt="{html_escape(issue['text']['title'])}标注帧" loading="lazy">
      <div class="issue-meta">
        <span>{html_escape(issue.get('measurements', {}).get('primary_metric', '-'))}</span>
        <strong>{value_text}</strong>
        <span>阈值 {threshold_text}</span>
        <span>rep {html_escape(issue.get('representative_rep_id', '-'))}</span>
        <span>{fmt_seconds(anchor_time)}</span>
        <span>置信度 {confidence_text}</span>
      </div>
    </article>
    """


def rep_rows_html(bundle: dict[str, Any], active_issues: list[dict[str, Any]]) -> str:
    rows = []
    analysis = bundle["analysis"]
    exercise = analysis["exercise"]
    angle_label = "底部膝角" if exercise == "squat" else "底部肘角"
    for rep in analysis["segmentation"]["reps"]:
        rep_id = int(rep["rep_id"])
        start_s = clip_relative_s(float(rep["start_time_s"]), analysis)
        end_s = clip_relative_s(float(rep["end_time_s"]), analysis)
        bottom_s = clip_relative_s(float(rep["bottom_time_s"]), analysis)
        titles = [
            issue["text"]["title"]
            for issue in active_issues
            if rep_id in issue.get("rep_ids", [])
        ]
        issues_text = "、".join(dict.fromkeys(titles)) or "未触发活动问题"
        rows.append(
            f"""
            <tr onclick="seekRelative({bottom_s:.6f})">
              <td>{rep_id}</td>
              <td>{fmt_seconds(start_s)} – {fmt_seconds(end_s)}</td>
              <td>{float(rep['duration_s']):.2f}s</td>
              <td>{int(rep['bottom_frame'])}</td>
              <td>{float(rep['bottom_angle_deg']):.1f}°</td>
              <td>{float(rep['confidence']):.0%}</td>
              <td>{html_escape(issues_text)}</td>
            </tr>
            """
        )
    header = f"<tr><th>rep</th><th>clip 时间区间</th><th>时长</th><th>bottom 帧</th><th>{angle_label}</th><th>分割置信度</th><th>问题</th></tr>"
    return f"<table class=rep-table><thead>{header}</thead><tbody>{''.join(rows)}</tbody></table>"


def timeline_html(bundle: dict[str, Any], active_issues: list[dict[str, Any]]) -> str:
    analysis = bundle["analysis"]
    reps = analysis["segmentation"]["reps"]
    rep_bars = []
    bottom_markers = []
    for rep in reps:
        left = timeline_position(float(rep["start_time_s"]), analysis)
        right = timeline_position(float(rep["end_time_s"]), analysis)
        width = max(0.2, right - left)
        rep_id = int(rep["rep_id"])
        rep_start = clip_relative_s(float(rep["start_time_s"]), analysis)
        rep_bottom = clip_relative_s(float(rep["bottom_time_s"]), analysis)
        rep_bars.append(
            f'<button class="rep-bar" style="left:{left:.3f}%;width:{width:.3f}%" '
            f'title="rep {rep_id}: {float(rep["duration_s"]):.2f}s" '
            f'onclick="seekRelative({rep_start:.6f})">{rep_id}</button>'
        )
        bottom = timeline_position(float(rep["bottom_time_s"]), analysis)
        bottom_markers.append(
            f'<button class="bottom-marker" style="left:{bottom:.3f}%" '
            f'title="rep {rep_id} bottom · {fmt_seconds(rep_bottom)}" '
            f'onclick="seekRelative({rep_bottom:.6f})"></button>'
        )
    issue_rows = []
    for issue in active_issues:
        evidence = issue.get("evidence_interval")
        if not evidence:
            continue
        start = timeline_position(float(evidence[0]) / float(analysis["fps"]), analysis)
        end = timeline_position(float(evidence[1]) / float(analysis["fps"]), analysis)
        anchor = clip_relative_s(float(issue["anchor_time_s"]), analysis)
        evidence_start = clip_relative_s(float(evidence[0]) / float(analysis["fps"]), analysis)
        evidence_end = clip_relative_s(float(evidence[1]) / float(analysis["fps"]), analysis)
        width = max(0.3, end - start)
        color = SEVERITY_COLORS.get(issue.get("severity"), SEVERITY_COLORS["none"])
        issue_rows.append(
            f"""
            <div class="timeline-row issue-row">
              <div class="row-label">{html_escape(issue['text']['title'])}</div>
              <div class="track">
                <button class="issue-band" style="left:{start:.3f}%;width:{width:.3f}%;background:{color}"
                  title="{html_escape(issue['issue_id'])} · {html_escape(issue['frequency'])}"
                  onclick="selectIssue('{html_escape(issue['issue_id'])}', {anchor:.6f}, {evidence_start:.6f}, {evidence_end:.6f})"></button>
              </div>
            </div>
            """
        )
    ticks = []
    duration = clip_duration_s(analysis)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        ticks.append(
            f'<span style="left:{fraction * 100:.1f}%">{fmt_seconds(duration * fraction)}</span>'
        )
    return f"""
    <section class="panel timeline-panel">
      <div class="panel-title"><h2>时间线与证据区间</h2><span>点击 rep、bottom 或问题色带可定位视频</span></div>
      <div class="timeline-axis">{''.join(ticks)}</div>
      <div class="timeline-row"><div class="row-label">rep</div><div class="track rep-track">{''.join(rep_bars)}<div class="bottom-layer">{''.join(bottom_markers)}</div></div></div>
      {''.join(issue_rows)}
    </section>
    """


def suppressed_html(issue_output: dict[str, Any]) -> str:
    suppressed = [issue for issue in issue_output.get("issues", []) if issue.get("suppressed")]
    if not suppressed:
        return '<p class="muted">无质量或视角屏蔽项。</p>'
    items = []
    for issue in suppressed:
        reason = issue.get("suppression_reason", "unknown")
        items.append(
            f'<li><strong>{html_escape(issue["text"]["title"])}</strong>'
            f'<span>{html_escape(REASON_LABELS.get(reason, reason))}</span>'
            f'<small>{html_escape(issue["text"].get("detail", ""))}</small></li>'
        )
    return "<ul class=suppressed-list>" + "".join(items) + "</ul>"


def build_report_html(
    bundle: dict[str, Any],
    templates: dict[str, Any],
    issue_images: dict[str, str],
    video_url: str,
    source_video_url: str,
    output_dir: Path,
) -> str:
    analysis = bundle["analysis"]
    issue_output = bundle["issue_output"]
    active_issues = [issue for issue in issue_output["issues"] if not issue.get("suppressed")]
    featured = [issue for issue in active_issues if issue.get("show_by_default")][:3]
    others = [issue for issue in active_issues if issue not in featured]
    reps = analysis["segmentation"]["reps"]
    average_duration = sum(float(rep["duration_s"]) for rep in reps) / len(reps) if reps else 0.0
    quality = analysis.get("quality", {})
    disabled_metrics = quality.get("view_support", {}).get("disabled_metrics", [])
    metric_labels = templates.get("metric_labels", {})
    disabled_text = "、".join(metric_labels.get(metric, metric) for metric in disabled_metrics) or "无"
    warnings = analysis.get("quality", {}).get("warnings", [])
    failures = analysis.get("quality", {}).get("failures", [])
    quality_items = [
        QUALITY_WARNING_LABELS.get(item, item) for item in failures + warnings
    ]
    quality_text = "；".join(quality_items) if quality_items else "无质量警告"
    feedback_rel = Path(os.path.relpath(bundle["feedback_dir"], output_dir)).as_posix()
    analysis_rel = Path(os.path.relpath(bundle["analysis_dir"], output_dir)).as_posix()
    featured_html = "".join(
        card_html(
            issue,
            issue_images.get(issue["issue_id"], ""),
            True,
            float(analysis["fps"]),
            clip_origin_s(analysis),
        )
        for issue in featured
    )
    others_html = "".join(
        card_html(
            issue,
            issue_images.get(issue["issue_id"], ""),
            False,
            float(analysis["fps"]),
            clip_origin_s(analysis),
        )
        for issue in others
    )
    more_block = (
        f'<details class="more-issues"><summary>其余活动问题卡（{len(others)}）</summary><div class="issue-grid">{others_html}</div></details>'
        if others
        else ""
    )
    if not active_issues and not reps:
        no_issue_block = (
            '<p class="no-issue">本 clip 没有识别到符合目标动作姿态的完整 rep；'
            '热身、起身和过渡动作已从点评中剔除，因此本段不生成动作问题卡。</p>'
        )
    elif not active_issues:
        no_issue_block = '<p class="no-issue">本段未检测到超过初版阈值的活动问题；未评估项仍不表示动作正常。</p>'
    else:
        no_issue_block = ""
    exercise_label = EXERCISE_LABELS.get(analysis["exercise"], analysis["exercise"])
    view_label = VIEW_LABELS.get(analysis.get("view"), analysis.get("view", "未知"))
    title = f"QEVD-FIT · {bundle['clip_id']} · 动作反馈报告"
    poster = issue_images.get(featured[0]["issue_id"], "") if featured else ""
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_escape(title)}</title>
  <style>{REPORT_CSS}</style>
</head>
<body>
  <header class="hero">
    <div>
      <p class="eyebrow">TASK 1-4D · 离线训练后分析</p>
      <h1>{html_escape(title)}</h1>
      <p class="subtitle">动作：{html_escape(exercise_label)} · 视角：{html_escape(view_label)} · split：{html_escape(analysis.get('split', '-'))}</p>
    </div>
    <div class="quality-badge">{html_escape(quality.get('status', '-'))}</div>
  </header>

  <div class="notice">初版工程阈值，尚未用专家问题级标注校准。报告用于回放定位与工程验证，不构成医学或动作标准结论。</div>

  <main>
    <section class="summary-grid">
      <div class="metric"><span>完整 rep</span><strong>{len(reps)}</strong></div>
      <div class="metric"><span>总时长</span><strong>{clip_duration_s(analysis):.1f}s</strong></div>
      <div class="metric"><span>平均 rep 时长</span><strong>{average_duration:.2f}s</strong></div>
      <div class="metric"><span>活动问题</span><strong>{len(active_issues)}</strong></div>
      <div class="metric"><span>屏蔽/未评估</span><strong>{issue_output.get('summary', {}).get('suppressed_issue_count', 0)}</strong></div>
    </section>

    <section class="viewer panel">
      <div class="panel-title"><h2>clip 回放</h2><span id="player-time">00:00.0</span></div>
      <video id="player" controls preload="auto" poster="{html_escape(poster)}" src="{html_escape(video_url)}"></video>
      <div class="viewer-actions">
        <button type="button" onclick="seekRelative(0)">回到 clip 起点</button>
        <button type="button" id="play-evidence" disabled>播放当前证据区间</button>
        <span id="selection-note" class="muted">点击问题卡定位代表帧</span>
      </div>
      <p class="muted">视频已裁剪并转为浏览器兼容的 VP8/WebM，播放器时间与时间线均为 clip 相对时间；问题卡中的 frame 编号仍对应原始视频。若跳转或拖动失败，请通过 <code>scripts/serve_reports.py</code> 启动服务；<code>python -m http.server</code> 不支持 Range。</p>
    </section>

    <section class="panel">
      <div class="panel-title"><h2>问题卡</h2><span>默认最多展示 3 张，全部活动问题可展开</span></div>
      {no_issue_block}
      <div class="issue-grid">{featured_html}</div>
      {more_block}
    </section>

    {timeline_html(bundle, active_issues)}

    <section class="panel">
      <div class="panel-title"><h2>逐 rep 明细</h2><span>点击行定位该 rep 的 bottom</span></div>
      {rep_rows_html(bundle, active_issues)}
    </section>

    <section class="split-grid">
      <div class="panel">
        <h2>质量与未评估项</h2>
        <p><strong>质量门控：</strong>{html_escape(quality.get('status', '-'))} · {html_escape(quality_text)}</p>
        <p><strong>相机运动：</strong>{html_escape((quality.get('checks', {}).get('camera_motion') or {}).get('status', 'not_evaluated'))}</p>
        <p><strong>当前视角禁用指标：</strong>{html_escape(disabled_text)}</p>
        {suppressed_html(issue_output)}
      </div>
      <div class="panel">
        <h2>原始数据</h2>
        <ul class="download-list">
          <li><a href="{html_escape(feedback_rel)}/issues.json" download>issues.json</a></li>
          <li><a href="{html_escape(feedback_rel)}/features.csv" download>features.csv</a></li>
          <li><a href="{html_escape(analysis_rel)}/analysis.json" download>analysis.json</a></li>
          <li><a href="{html_escape(analysis_rel)}/reps.csv" download>reps.csv</a></li>
          <li><a href="{html_escape(analysis_rel)}/angles/angles_wide.csv" download>angles_wide.csv</a></li>
          <li><a href="{html_escape(source_video_url)}" download>原始长视频（MP4V，仅下载）</a></li>
        </ul>
        <p class="muted">请使用 <code>VLS/V0/scripts/serve_reports.py</code> 启动支持 HTTP Range 的本地服务后打开报告。</p>
      </div>
    </section>
  </main>
  <script>{REPORT_JS}</script>
</body>
</html>
"""
    return page


def build_index_html(summary: dict[str, Any]) -> str:
    rows = []
    for result in summary["results"]:
        severity = result["severity_counts"]
        rows.append(
            f"""
            <tr>
              <td><a href="{html_escape(result['clip_id'])}/report.html">{html_escape(result['clip_id'])}</a></td>
              <td>{html_escape(EXERCISE_LABELS.get(result['exercise'], result['exercise']))}</td>
              <td>{result['rep_count']}</td>
              <td>{result['active_issue_count']}</td>
              <td><span class="mini high">{severity.get('high', 0)}</span><span class="mini medium">{severity.get('medium', 0)}</span><span class="mini low">{severity.get('low', 0)}</span></td>
              <td>{result['frame_count']}</td>
              <td>{html_escape(result['quality_status'])}</td>
            </tr>
            """
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>QEVD-FIT · T1-4D 报告索引</title>
  <style>{REPORT_CSS}</style>
</head>
<body>
  <header class="hero"><div><p class="eyebrow">TASK 1-4D</p><h1>动作反馈报告索引</h1><p class="subtitle">生成于 {html_escape(summary['generated_at'])} · {summary['clip_count']} 个 clip · {summary['frame_count']} 张标注帧</p></div></header>
  <div class="notice">阈值仍为 provisional；点击 clip 进入可回放问题卡、时间线和原始数据链接。</div>
  <main><section class="panel"><table class=rep-table><thead><tr><th>clip</th><th>动作</th><th>rep</th><th>活动问题</th><th>严重度 H/M/L</th><th>标注帧</th><th>质量</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section></main>
</body>
</html>
"""


REPORT_CSS = """
:root{color-scheme:light;--ink:#172033;--muted:#64748b;--line:#dbe3ee;--panel:#ffffff;--bg:#f3f6fa;--red:#dc2626;--orange:#ea580c;--yellow:#ca8a04}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:"Noto Sans CJK SC","Microsoft YaHei",system-ui,sans-serif;line-height:1.55}
.hero{display:flex;justify-content:space-between;gap:24px;align-items:flex-end;padding:36px max(24px,calc((100vw - 1180px)/2));background:linear-gradient(125deg,#0f172a,#1e3a5f);color:white}
.hero h1{font-size:clamp(24px,3vw,38px);margin:4px 0}.eyebrow{letter-spacing:.16em;color:#93c5fd;font-size:12px;margin:0}.subtitle{margin:4px 0 0;color:#cbd5e1}
.quality-badge{border:1px solid #93c5fd;padding:8px 14px;border-radius:999px;text-transform:uppercase;font-weight:700}
.notice{margin:20px auto 0;max-width:1180px;padding:12px 16px;border-left:4px solid #f59e0b;background:#fffbeb;color:#78350f;border-radius:8px}
main{max-width:1180px;margin:20px auto 60px;padding:0 24px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 8px 24px rgba(15,23,42,.05);margin-bottom:20px}
.panel h2,.panel-title h2{margin:0;font-size:19px}.panel-title{display:flex;justify-content:space-between;gap:16px;align-items:baseline;margin-bottom:14px}.panel-title span,.muted{color:var(--muted);font-size:13px}
.summary-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:20px}.metric{background:white;border:1px solid var(--line);border-radius:14px;padding:14px 16px}.metric span{display:block;color:var(--muted);font-size:12px}.metric strong{font-size:26px}
.viewer video{width:100%;max-height:68vh;background:#020617;border-radius:12px;display:block}.viewer-actions{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:12px}
button{border:1px solid #cbd5e1;background:white;color:var(--ink);border-radius:999px;padding:8px 13px;cursor:pointer;font:inherit}button:hover{border-color:#2563eb;color:#1d4ed8}button:disabled{opacity:.45;cursor:not-allowed}
.issue-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.issue-card{position:relative;border:1px solid var(--line);border-top:5px solid var(--severity);border-radius:14px;padding:16px;background:#fff;transition:.16s;overflow:hidden}.issue-card.featured{box-shadow:0 8px 22px rgba(15,23,42,.08)}.issue-card.selected{outline:3px solid #2563eb;outline-offset:2px}.card-hit{position:absolute;inset:0;background:transparent;border:0;border-radius:0;z-index:1}.issue-head,.issue-card h3,.issue-card p,.issue-card img,.issue-meta{position:relative;z-index:2;pointer-events:none}.issue-card h3{margin:8px 0 4px;font-size:18px}.issue-card p{margin:4px 0;color:#334155}.issue-card .cue{color:var(--muted);font-size:13px}.issue-card img{width:100%;margin:12px 0;border-radius:10px;background:#020617;max-height:420px;object-fit:contain}.issue-meta{display:flex;gap:8px;flex-wrap:wrap;font-size:12px;color:var(--muted)}.issue-meta strong{color:var(--ink)}
.severity{background:var(--severity);color:white;border-radius:999px;padding:2px 9px;font-size:12px;font-weight:700}.issue-frequency,.default-tag{font-size:12px;color:var(--muted);margin-left:6px}.default-tag{color:#1d4ed8}
.more-issues{margin-top:18px}.more-issues summary{cursor:pointer;color:#1d4ed8;font-weight:600;margin-bottom:12px}
.timeline-panel{overflow:hidden}.timeline-axis{position:relative;height:24px;margin-left:132px;color:var(--muted);font-size:11px}.timeline-axis span{position:absolute;transform:translateX(-50%)}.timeline-row{display:grid;grid-template-columns:120px 1fr;gap:12px;align-items:center;margin:7px 0}.row-label{font-size:12px;color:#475569;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.track{height:28px;position:relative;background:#eef2f7;border-radius:999px;overflow:hidden}.rep-bar{position:absolute;top:0;height:28px;border-radius:0;border:0;border-right:1px solid white;background:#bfdbfe;color:#1e3a8a;padding:0;font-size:10px}.rep-bar:hover{background:#93c5fd}.bottom-layer{position:absolute;inset:0;pointer-events:none}.bottom-marker{position:absolute;bottom:0;width:7px;height:9px;padding:0;border:0;border-radius:3px 3px 0 0;background:#1d4ed8;transform:translateX(-50%);pointer-events:auto}.issue-row .track{height:18px}.issue-band{position:absolute;top:3px;height:12px;border:0;border-radius:999px;opacity:.72;padding:0;min-width:3px}.issue-band:hover{opacity:1}
.rep-table{width:100%;border-collapse:collapse;font-size:13px}.rep-table th,.rep-table td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line)}.rep-table tbody tr{cursor:pointer}.rep-table tbody tr:hover{background:#f8fafc}
.split-grid{display:grid;grid-template-columns:1.2fr .8fr;gap:20px}.suppressed-list{list-style:none;padding:0;margin:12px 0 0}.suppressed-list li{display:grid;gap:2px;padding:10px;border-radius:10px;background:#f8fafc;margin-bottom:8px}.suppressed-list span{color:#b45309;font-size:12px}.suppressed-list small{color:var(--muted)}.download-list{list-style:none;padding:0}.download-list li{margin:8px 0}.download-list a{color:#1d4ed8;text-decoration:none}.download-list a:hover{text-decoration:underline}
.no-issue{padding:12px 14px;background:#ecfdf5;border:1px solid #a7f3d0;border-radius:10px;color:#065f46}.mini{display:inline-block;min-width:24px;text-align:center;border-radius:999px;padding:1px 6px;color:white;margin-right:4px}.mini.high{background:var(--red)}.mini.medium{background:var(--orange)}.mini.low{background:var(--yellow)}
@media(max-width:850px){.summary-grid{grid-template-columns:repeat(2,1fr)}.issue-grid,.split-grid{grid-template-columns:1fr}.timeline-row{grid-template-columns:86px 1fr}.timeline-axis{margin-left:98px}.hero{flex-direction:column;align-items:flex-start}}
"""


REPORT_JS = """
const player = document.getElementById('player');
const playerTime = document.getElementById('player-time');
const selectionNote = document.getElementById('selection-note');
const playEvidenceButton = document.getElementById('play-evidence');
let evidenceWindow = null;
let selectedIssueId = null;
let evidencePlaying = false;
let evidenceSeekToken = 0;

function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return '00:00.0';
  const minutes = Math.floor(seconds / 60);
  const rest = (seconds - minutes * 60).toFixed(1).padStart(4, '0');
  return `${String(minutes).padStart(2, '0')}:${rest}`;
}

function seekRelative(time, pause = true) {
  if (!player) return;
  const target = Math.max(0, Number(time));
  const applySeek = () => {
    player.currentTime = target;
    if (pause) player.pause();
  };
  if (player.readyState >= 1) {
    applySeek();
  } else {
    player.addEventListener('loadedmetadata', applySeek, {once: true});
  }
  player.scrollIntoView({behavior: 'smooth', block: 'center'});
}

function selectIssue(issueId, anchorTime, evidenceStart, evidenceEnd) {
  selectedIssueId = issueId;
  const start = Number(evidenceStart);
  const end = Number(evidenceEnd);
  evidenceWindow = Number.isFinite(start) && Number.isFinite(end) && end > start
    ? [start, end]
    : null;
  evidencePlaying = false;
  evidenceSeekToken += 1;
  document.querySelectorAll('.issue-card').forEach(card => {
    card.classList.toggle('selected', card.dataset.issueId === issueId);
  });
  selectionNote.textContent = `已定位 ${issueId}`;
  playEvidenceButton.disabled = !evidenceWindow;
  seekRelative(anchorTime, true);
}

function playEvidence() {
  if (!evidenceWindow || !player) return;
  const start = evidenceWindow[0];
  const end = evidenceWindow[1];
  evidencePlaying = true;
  evidenceSeekToken += 1;
  const token = evidenceSeekToken;
  player.pause();
  selectionNote.textContent = `正在定位证据区间 ${formatTime(start)} – ${formatTime(end)}`;

  const begin = () => {
    if (token !== evidenceSeekToken || !evidenceWindow) return;
    selectionNote.textContent = `正在回放 ${selectedIssueId} · ${formatTime(start)} – ${formatTime(end)}`;
    const playPromise = player.play();
    if (playPromise && typeof playPromise.catch === 'function') {
      playPromise.catch(() => {
        evidencePlaying = false;
        selectionNote.textContent = '浏览器阻止了播放，请再次点击“播放当前证据区间”。';
      });
    }
  };

  if (Math.abs(player.currentTime - start) <= 0.04 && player.readyState >= 1) {
    begin();
    return;
  }
  const onSeeked = () => {
    player.removeEventListener('seeked', onSeeked);
    begin();
  };
  player.addEventListener('seeked', onSeeked);
  player.currentTime = start;
}

if (player) {
  player.addEventListener('loadedmetadata', () => {
    player.currentTime = 0;
    if (playerTime) playerTime.textContent = formatTime(0);
  });
  player.addEventListener('timeupdate', () => {
    if (playerTime) playerTime.textContent = formatTime(player.currentTime);
    if (evidencePlaying && evidenceWindow && !player.paused && player.currentTime >= evidenceWindow[1] - 0.01) {
      player.pause();
      evidencePlaying = false;
      playEvidenceButton.disabled = false;
      selectionNote.textContent = '证据区间回放结束，可再次播放。';
    }
  });
  player.addEventListener('error', () => {
    evidencePlaying = false;
    playEvidenceButton.disabled = true;
    selectionNote.textContent = '视频加载失败，请确认 clip 代理视频存在，并通过支持 Range 的本地 HTTP 服务打开报告。';
  });
}
if (playEvidenceButton) playEvidenceButton.addEventListener('click', playEvidence);
"""


def render_clip(
    bundle: dict[str, Any],
    templates: dict[str, Any],
    output_root: Path,
    landmarker: Any | None,
    model_path: Path,
    skip_frames: bool,
    skip_video: bool,
    previous_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = output_root / bundle["clip_id"]
    frame_dir = output_dir / "annotated_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    issue_output = bundle["issue_output"]
    active_issues = [issue for issue in issue_output["issues"] if not issue.get("suppressed")]
    expected_frame_names = {
        f"{issue['issue_id']}_frame_{int(issue['anchor_frame'])}.png"
        for issue in active_issues
    }
    issue_images = {}
    pose_failures = 0
    pose_fallbacks = 0
    generated_frames = 0

    if skip_frames:
        for issue in active_issues:
            filename = f"{issue['issue_id']}_frame_{int(issue['anchor_frame'])}.png"
            if (frame_dir / filename).exists():
                issue_images[issue["issue_id"]] = f"annotated_frames/{filename}"
        generated_frames = len(issue_images)
        if previous_result:
            pose_fallbacks = int(previous_result.get("pose_fallback_count", 0))
            pose_failures = int(previous_result.get("pose_detection_failures", 0))
    elif active_issues:
        if landmarker is None:
            raise ValueError("landmarker is required when frames are rendered")
        with VideoFrameReader(
            bundle["video_path"], bundle["analysis"].get("rotation", "none")
        ) as reader:
            for issue in active_issues:
                anchor = int(issue["anchor_frame"])
                frame = reader.read(anchor)
                landmarks = detect_normalized_landmarks(landmarker, frame)
                exercise = bundle["analysis"]["exercise"]
                if not landmarks_are_plausible(landmarks, exercise):
                    tracked = detect_video_tracked_landmarks(
                        bundle["video_path"],
                        bundle["analysis"].get("rotation", "none"),
                        anchor,
                        float(bundle["analysis"]["fps"]),
                        model_path,
                    )
                    if tracked is not None and landmarks_are_plausible(tracked, exercise):
                        landmarks = tracked
                        pose_fallbacks += 1
                    else:
                        landmarks = None
                wide_row = bundle["wide"].get(
                    anchor - int(bundle["analysis"]["trim"]["start_frame"]), {}
                )
                annotated, detected = render_issue_frame(
                    frame,
                    landmarks,
                    issue,
                    wide_row,
                    clip_origin_s(bundle["analysis"]),
                )
                filename = f"{issue['issue_id']}_frame_{anchor}.png"
                path = frame_dir / filename
                cv2.imwrite(str(path), annotated)
                issue_images[issue["issue_id"]] = f"annotated_frames/{filename}"
                generated_frames += 1
                pose_failures += int(not detected)

    stale_frames_removed = prune_stale_annotated_frames(frame_dir, expected_frame_names)
    web_video_path = output_dir / "media" / f"{bundle['clip_id']}_clip.webm"
    if not skip_video or not web_video_path.exists():
        web_video_path = build_web_video(bundle, output_dir)
    video_url = Path(os.path.relpath(web_video_path, output_dir)).as_posix()
    video_url = quote(video_url, safe="/:")
    source_video_url = Path(os.path.relpath(bundle["video_path"], output_dir)).as_posix()
    source_video_url = quote(source_video_url, safe="/:")
    page = build_report_html(
        bundle,
        templates,
        issue_images,
        video_url,
        source_video_url,
        output_dir,
    )
    (output_dir / "report.html").write_text(page)
    severity_counts = Counter(
        issue["severity"] for issue in active_issues if issue.get("severity") in SEVERITY_LABELS
    )
    return {
        "clip_id": bundle["clip_id"],
        "exercise": bundle["analysis"]["exercise"],
        "view": bundle["analysis"].get("view"),
        "quality_status": bundle["analysis"].get("quality_status"),
        "rep_count": len(bundle["analysis"]["segmentation"]["reps"]),
        "active_issue_count": len(active_issues),
        "suppressed_issue_count": issue_output.get("summary", {}).get("suppressed_issue_count", 0),
        "severity_counts": {
            severity: severity_counts.get(severity, 0) for severity in ("high", "medium", "low")
        },
        "frame_count": generated_frames,
        "pose_fallback_count": pose_fallbacks,
        "pose_detection_failures": pose_failures,
        "stale_frames_removed": stale_frames_removed,
        "video": str(web_video_path.resolve()),
        "video_codec": "vp8/webm",
        "report": str((output_dir / "report.html").resolve()),
        "annotated_frames": str(frame_dir.resolve()),
    }


def main() -> None:
    args = parse_args()
    templates = read_json(args.templates)
    clip_ids = select_clip_ids(args)
    if not clip_ids:
        raise ValueError("no clip matched")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundles = [load_bundle(clip_id, args.feedback_dir, args.analysis_dir) for clip_id in clip_ids]
    previous_results = {}
    previous_summary_path = args.output_dir / "summary.json"
    if previous_summary_path.exists():
        previous_summary = read_json(previous_summary_path)
        previous_results = {
            result["clip_id"]: result for result in previous_summary.get("results", [])
        }
    has_frames = not args.skip_frames and any(
        not issue["suppressed"] for bundle in bundles for issue in bundle["issue_output"]["issues"]
    )
    landmarker = create_landmarker(args.model) if has_frames else None
    results = []
    try:
        for bundle in bundles:
            result = render_clip(
                bundle,
                templates,
                args.output_dir,
                landmarker,
                args.model,
                args.skip_frames,
                args.skip_video,
                previous_results.get(bundle["clip_id"]),
            )
            results.append(result)
            print(
                f"{result['clip_id']}: {result['active_issue_count']} issues, "
                f"{result['frame_count']} annotated frames",
                flush=True,
            )
    finally:
        if landmarker is not None:
            landmarker.close()

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "task": "task1-4d",
        "feedback_dir": str(args.feedback_dir.resolve()),
        "analysis_dir": str(args.analysis_dir.resolve()),
        "clip_count": len(results),
        "frame_count": sum(result["frame_count"] for result in results),
        "pose_fallback_count": sum(result["pose_fallback_count"] for result in results),
        "pose_detection_failures": sum(result["pose_detection_failures"] for result in results),
        "active_issue_count": sum(result["active_issue_count"] for result in results),
        "results": results,
        "index": str((args.output_dir / "index.html").resolve()),
    }
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "index.html").write_text(build_index_html(summary))
    print(
        f"wrote {summary['clip_count']} reports and {summary['frame_count']} frames to {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
