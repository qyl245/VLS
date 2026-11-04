#!/usr/bin/env python3
"""Render Fit3D GT skeletons, rep annotations, and joint-angle curves."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

import build_task1_5_fit3d_manifest as manifest_builder


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "task1_5_fit3d_clips.json"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "task1_5" / "manifest" / "gt_overlay"

JOINT_EDGES = (
    (0, 1),
    (0, 4),
    (1, 2),
    (2, 3),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (8, 14),
    (11, 12),
    (12, 13),
    (14, 15),
    (15, 16),
)

JOINT_GROUPS = {
    "squat": {
        "left": (1, 2, 3),
        "right": (4, 5, 6),
        "label": "knee_angle",
    },
    "pushup": {
        "left": (11, 12, 13),
        "right": (14, 15, 16),
        "label": "elbow_angle",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--clip-id", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overlay-height", type=int, default=640)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def three_point_angle(
    points: np.ndarray, indices: tuple[int, int, int]
) -> np.ndarray:
    first = points[:, indices[0]]
    center = points[:, indices[1]]
    last = points[:, indices[2]]
    upper = first - center
    lower = last - center
    denominator = np.linalg.norm(upper, axis=-1) * np.linalg.norm(lower, axis=-1)
    cosine = np.sum(upper * lower, axis=-1) / np.maximum(denominator, 1e-9)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def select_rep_frames(rep_frames: list[int]) -> list[int]:
    if not rep_frames:
        raise ValueError("rep frames are required for overlay rendering")
    if len(rep_frames) <= 3:
        return list(rep_frames)
    return [rep_frames[0], rep_frames[len(rep_frames) // 2], rep_frames[-1]]


def read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"cannot decode frame {frame_index}: {video_path}")
        return frame
    finally:
        cap.release()


def project_points(
    points: np.ndarray, camera: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    rotation = camera["rotation"]
    translation = camera["translation"]
    rvec, _ = cv2.Rodrigues(rotation)
    projected, _ = cv2.projectPoints(
        points, rvec, translation, camera["intrinsic_matrix"], None
    )
    projected = projected.reshape(-1, 2)
    camera_points = (rotation @ points.T).T + translation
    return projected, camera_points[:, 2] > 1e-6


def draw_skeleton(
    image: np.ndarray,
    projected: np.ndarray,
    visible: np.ndarray,
    highlight: tuple[int, int, int] | None = None,
) -> np.ndarray:
    canvas = image.copy()
    for start, end in JOINT_EDGES:
        if not (visible[start] and visible[end]):
            continue
        cv2.line(
            canvas,
            tuple(np.round(projected[start]).astype(int)),
            tuple(np.round(projected[end]).astype(int)),
            (40, 220, 40),
            3,
            cv2.LINE_AA,
        )
    for index in range(projected.shape[0]):
        if not visible[index]:
            continue
        color = (0, 165, 255) if highlight and index in highlight else (0, 80, 255)
        cv2.circle(
            canvas,
            tuple(np.round(projected[index]).astype(int)),
            5,
            color,
            -1,
            cv2.LINE_AA,
        )
    return canvas


def make_overlay_panel(
    clip: dict[str, Any],
    joints: np.ndarray,
    camera: dict[str, Any],
    frame_index: int,
    angle_name: str,
    angle_value: float,
    height: int,
) -> tuple[np.ndarray, int]:
    frame = read_frame(Path(clip["video_path"]), frame_index)
    projected, visible = project_points(joints[frame_index], camera)
    in_frame = (
        visible
        & (projected[:, 0] >= 0)
        & (projected[:, 0] < frame.shape[1])
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < frame.shape[0])
    )
    rendered = draw_skeleton(frame, projected, visible)
    rendered = cv2.resize(
        rendered,
        (height, height),
        interpolation=cv2.INTER_AREA,
    )
    panel = np.full((height + 58, height, 3), 20, dtype=np.uint8)
    panel[:height] = rendered
    lines = [
        f"frame {frame_index} | {clip['camera_id']} | {clip['view']}",
        f"GT {angle_name}={angle_value:.1f} deg | in-frame={in_frame.mean():.2f}",
    ]
    for row, text in enumerate(lines):
        cv2.putText(
            panel,
            text,
            (8, height + 20 + row * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return panel, int(in_frame.sum())


def render_clip_overlay(
    clip: dict[str, Any],
    joints: np.ndarray,
    output_path: Path,
    height: int,
) -> dict[str, Any]:
    camera = manifest_builder.load_camera_parameters(
        Path(clip["gt"]["camera_parameters"])
    )
    groups = JOINT_GROUPS[clip["exercise"]]
    left_angle = three_point_angle(joints, groups["left"])
    right_angle = three_point_angle(joints, groups["right"])
    mean_angle = (left_angle + right_angle) * 0.5
    rep_frames = [int(value) for value in clip["gt"]["rep_frames"]]
    selected = select_rep_frames(rep_frames)
    panels = []
    visible_joint_counts = []
    for frame_index in selected:
        panel, visible_count = make_overlay_panel(
            clip,
            joints,
            camera,
            frame_index,
            groups["label"],
            float(mean_angle[frame_index]),
            height,
        )
        panels.append(panel)
        visible_joint_counts.append(visible_count)
    sheet = np.hstack(panels)
    title = (
        f"{clip['clip_id']} | GT 3D projection | "
        f"rep frames={selected}"
    )
    canvas = np.full((sheet.shape[0] + 46, sheet.shape[1], 3), 12, dtype=np.uint8)
    canvas[46:] = sheet
    cv2.putText(
        canvas,
        title,
        (14, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"failed to write {output_path}")
    return {
        "frames": selected,
        "visible_joint_counts": visible_joint_counts,
        "angle_at_frames_deg": [
            round(float(mean_angle[frame]), 3) for frame in selected
        ],
    }


def rep_semantics(
    angles: np.ndarray, rep_frames: list[int]
) -> dict[str, Any]:
    labels = np.asarray([angles[frame] for frame in rep_frames], dtype=np.float64)
    interval_minima = []
    interval_min_frames = []
    for start, end in zip(rep_frames[:-1], rep_frames[1:]):
        window = angles[start : end + 1]
        local_index = int(np.argmin(window))
        interval_minima.append(float(window[local_index]))
        interval_min_frames.append(start + local_index)
    p95 = float(np.percentile(angles, 95.0))
    label_median = float(np.median(labels))
    delta_to_p95 = label_median - p95
    near_extension = delta_to_p95 >= -15.0
    return {
        "label_frame_count": len(rep_frames),
        "label_angle_median_deg": round(label_median, 3),
        "label_angle_min_deg": round(float(labels.min()), 3),
        "label_angle_max_deg": round(float(labels.max()), 3),
        "curve_p95_deg": round(p95, 3),
        "label_delta_to_p95_deg": round(delta_to_p95, 3),
        "interval_min_median_deg": round(float(np.median(interval_minima)), 3)
        if interval_minima
        else None,
        "interval_min_frames": interval_min_frames,
        "provisional_semantics": (
            "near_extension_top_or_boundary"
            if near_extension
            else "not_confirmed"
        ),
        "bottom_gt_status": "not_confirmed_do_not_use_as_bottom",
    }


def render_rep_curve(
    subject: str,
    exercise: str,
    joints: np.ndarray,
    rep_frames: list[int],
    output_path: Path,
) -> dict[str, Any]:
    groups = JOINT_GROUPS[exercise]
    left = three_point_angle(joints, groups["left"])
    right = three_point_angle(joints, groups["right"])
    mean_angle = (left + right) * 0.5
    frames = np.arange(len(mean_angle))
    figure, axis = plt.subplots(figsize=(13, 4.8))
    axis.plot(frames, left, color="#2a7fbf", linewidth=1.1, label="left")
    axis.plot(frames, right, color="#c44e52", linewidth=1.1, label="right")
    axis.plot(
        frames,
        mean_angle,
        color="#222222",
        linewidth=1.4,
        label="bilateral mean",
    )
    for index, frame in enumerate(rep_frames):
        axis.axvline(frame, color="#777777", linestyle="--", linewidth=0.9)
        axis.text(
            frame,
            0.98,
            f"rep {index + 1}",
            rotation=90,
            va="top",
            ha="right",
            transform=axis.get_xaxis_transform(),
            fontsize=7,
            color="#444444",
        )
    axis.set_title(
        f"{subject} {exercise} | Fit3D GT {groups['label']} | "
        "dashed lines = rep_ann frames"
    )
    axis.set_xlabel("frame index (50 FPS)")
    axis.set_ylabel("angle (deg)")
    axis.set_xlim(0, len(mean_angle) - 1)
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right", ncol=3, fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    result = rep_semantics(mean_angle, rep_frames)
    result["curve_path"] = str(output_path.name)
    result["angle_name"] = groups["label"]
    return result


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    clips = {clip["clip_id"]: clip for clip in manifest["clips"]}
    missing = sorted(set(args.clip_id) - set(clips))
    if missing:
        raise SystemExit(f"unknown clip ids: {missing}")

    overlay_dir = args.output_dir / "overlays"
    curve_dir = args.output_dir / "curves"
    joint_cache: dict[str, np.ndarray] = {}
    rep_cache: dict[str, list[int]] = {}
    summary: dict[str, Any] = {
        "schema_version": "task1-5a-gt-overlay-v1",
        "manifest": str(args.manifest.resolve()),
        "overlays": {},
        "rep_semantics": {},
    }

    for clip_id in args.clip_id:
        clip = clips[clip_id]
        gt = clip.get("gt") or {}
        joints_path = gt.get("joints3d_25")
        if not joints_path:
            raise ValueError(f"{clip_id}: GT is required for overlay rendering")
        cache_key = str(Path(joints_path).resolve())
        if cache_key not in joint_cache:
            joint_cache[cache_key] = np.asarray(
                json.loads(Path(joints_path).read_text())["joints3d_25"],
                dtype=np.float64,
            )
        joints = joint_cache[cache_key]
        rep_frames = [int(value) for value in gt.get("rep_frames") or []]
        rep_cache[clip_id] = rep_frames
        overlay_path = overlay_dir / f"{clip_id}.jpg"
        summary["overlays"][clip_id] = render_clip_overlay(
            clip, joints, overlay_path, args.overlay_height
        )
        summary["overlays"][clip_id]["path"] = str(
            overlay_path.relative_to(args.output_dir)
        )

    grouped: defaultdict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for clip_id in args.clip_id:
        clip = clips[clip_id]
        grouped[(clip["subject_id"], clip["exercise"])][clip_id] = clip

    for (subject, exercise), group in sorted(grouped.items()):
        representative = next(iter(group.values()))
        joints_path = str(Path(representative["gt"]["joints3d_25"]).resolve())
        joints = joint_cache[joints_path]
        rep_frames = rep_cache[representative["clip_id"]]
        curve_path = curve_dir / f"{subject}_{exercise}_gt_rep_curve.png"
        summary["rep_semantics"][f"{subject}_{exercise}"] = render_rep_curve(
            subject, exercise, joints, rep_frames, curve_path
        )

    write_json(args.output_dir / "summary.json", summary)
    print(
        f"wrote {len(summary['overlays'])} overlays and "
        f"{len(summary['rep_semantics'])} rep curves under {args.output_dir}"
    )


if __name__ == "__main__":
    main()
