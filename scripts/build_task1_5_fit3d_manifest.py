#!/usr/bin/env python3
"""Build and validate the task1-5 Fit3D clip manifest."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = ROOT.parent / "dataset" / "Fit3D"
DEFAULT_SPECS = ROOT / "configs" / "task1_4_exercise_specs.json"
DEFAULT_OUTPUT = ROOT / "configs" / "task1_5_fit3d_clips.json"
DEFAULT_RESULTS_DIR = ROOT / "results" / "task1_5" / "manifest"

ARTIFACT_SCHEMA_VERSION = "task1-5a-fit3d-clips-v1"
MANIFEST_SCHEMA_VERSION = "task1-4-v1"

TRAIN_DEV = ("s03", "s04", "s05", "s07")
TRAIN_HOLDOUT = ("s08", "s09", "s10", "s11")
TEST_SUBJECTS = ("s02", "s12", "s13")
CAMERA_IDS = ("50591643", "58860488", "60457274", "65906101")

ACTIONS = {
    "squat": {
        "action_display": "squats",
        "variant": "barbell_back_squat",
        "note": "Fit3D squat uses a barbell; it is not the bodyweight squat targeted by T1-4.",
    },
    "pushup": {
        "action_display": "pushups",
        "variant": "standard_floor_pushup",
        "note": "Fit3D pushup is a standard floor pushup.",
    },
}

VIEWS = {
    "front",
    "rear",
    "oblique_left",
    "oblique_right",
    "side_left",
    "side_right",
    "unknown",
}
FRONTAL_VIEWS = {"front", "rear", "oblique_left", "oblique_right"}
SIDE_VIEWS = {"side_left", "side_right"}
OBLIQUE_VIEWS = {"oblique_left", "oblique_right"}

FRONT_MAX_DEG = 30.0
SIDE_MIN_DEG = 60.0
MAX_VIEW_SPREAD_DEG = 25.0
MIN_PROJECTION_IN_FRAME_RATE = 0.80

KEY_PROJECTION_JOINTS = tuple(range(25))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--view-overrides", type=Path, default=None)
    parser.add_argument("--include-test", action="store_true", default=True)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return None
    return vector / norm


def split_for_subject(subject: str) -> str:
    if subject in TRAIN_DEV:
        return "dev"
    if subject in TRAIN_HOLDOUT:
        return "holdout"
    if subject in TEST_SUBJECTS:
        return "smoke"
    raise ValueError(f"unknown Fit3D subject: {subject}")


def classify_view(
    angle_deg: float,
    front_rear: str,
    camera_side: str,
    spread_deg: float,
) -> str:
    if not math.isfinite(angle_deg) or spread_deg > MAX_VIEW_SPREAD_DEG:
        return "unknown"
    if angle_deg <= FRONT_MAX_DEG:
        return front_rear
    if angle_deg >= SIDE_MIN_DEG:
        return f"side_{camera_side}"
    return f"oblique_{camera_side}"


def load_view_overrides(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    overrides = payload.get("overrides", payload) if isinstance(payload, dict) else {}
    if not isinstance(overrides, dict):
        raise ValueError("view overrides must be a JSON object")
    cleaned: dict[str, str] = {}
    for clip_id, value in overrides.items():
        view = value.get("view") if isinstance(value, dict) else value
        if view not in VIEWS:
            raise ValueError(f"invalid view override for {clip_id}: {view!r}")
        cleaned[str(clip_id)] = str(view)
    return cleaned


def load_camera_parameters(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    rotation = np.asarray(payload["extrinsics"]["R"], dtype=np.float64)
    camera_center = np.asarray(payload["extrinsics"]["T"], dtype=np.float64).reshape(3)
    intrinsics = payload.get("intrinsics_wo_distortion") or {}
    focal = np.asarray(intrinsics.get("f"), dtype=np.float64).reshape(-1)
    center = np.asarray(intrinsics.get("c"), dtype=np.float64).reshape(-1)
    if focal.size < 2 or center.size < 2:
        raise ValueError(f"invalid intrinsics: {path}")
    intrinsic_matrix = np.array(
        [
            [focal[0], 0.0, center[0]],
            [0.0, focal[1], center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return {
        "path": str(path.resolve()),
        "rotation": rotation,
        "camera_center": camera_center,
        "translation": -rotation @ camera_center,
        "intrinsic_matrix": intrinsic_matrix,
        "model": "opencv_world_to_camera_t_equals_minus_R_center",
    }


def video_metadata(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    try:
        return {
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


def projection_quality(
    joints: np.ndarray,
    frame_indices: list[int],
    camera: dict[str, Any],
    width: int,
    height: int,
) -> dict[str, Any]:
    rotation = camera["rotation"]
    translation = camera["translation"]
    intrinsic = camera["intrinsic_matrix"]
    rvec, _ = cv2.Rodrigues(rotation)
    in_frame_rates: list[float] = []
    depth_rates: list[float] = []
    spans: list[float] = []
    sample = frame_indices[: min(12, len(frame_indices))]
    for frame_index in sample:
        points = joints[frame_index]
        camera_points = (rotation @ points.T).T + translation
        valid_depth = camera_points[:, 2] > 1e-6
        projected, _ = cv2.projectPoints(points, rvec, translation, intrinsic, None)
        projected = projected.reshape(-1, 2)[list(KEY_PROJECTION_JOINTS)]
        valid_depth = valid_depth[list(KEY_PROJECTION_JOINTS)]
        in_frame = (
            valid_depth
            & (projected[:, 0] >= 0.0)
            & (projected[:, 0] < width)
            & (projected[:, 1] >= 0.0)
            & (projected[:, 1] < height)
        )
        in_frame_rates.append(float(in_frame.mean()))
        depth_rates.append(float(valid_depth.mean()))
        if in_frame.any():
            visible = projected[in_frame]
            spans.append(float(np.linalg.norm(visible.max(axis=0) - visible.min(axis=0))))
    return {
        "sample_frames": sample,
        "in_frame_rate": float(np.mean(in_frame_rates)) if in_frame_rates else 0.0,
        "depth_valid_rate": float(np.mean(depth_rates)) if depth_rates else 0.0,
        "median_visible_joint_span_px": float(np.median(spans)) if spans else 0.0,
    }


def sample_rep_range(rep_frames: list[int], radius: int = 3) -> list[int]:
    if not rep_frames:
        return []
    frames = {
        frame
        for rep_frame in rep_frames
        for frame in range(max(0, rep_frame - radius), rep_frame + radius + 1)
    }
    return sorted(frames)


def view_geometry(
    joints: np.ndarray,
    frame_indices: list[int],
    camera_center: np.ndarray,
) -> dict[str, Any]:
    angles: list[float] = []
    front_cosines: list[float] = []
    lateral_cosines: list[float] = []
    for frame_index in frame_indices:
        points = joints[frame_index]
        hip_mid = (points[1] + points[4]) * 0.5
        shoulder_mid = (points[11] + points[14]) * 0.5
        lateral = unit(points[4] - points[1])
        torso = unit(shoulder_mid - hip_mid)
        ray = unit(camera_center - hip_mid)
        if lateral is None or torso is None or ray is None:
            continue
        normal = unit(np.cross(lateral, torso))
        if normal is None:
            continue
        front_cosine = float(np.clip(np.dot(ray, normal), -1.0, 1.0))
        lateral_cosine = float(np.clip(np.dot(ray, lateral), -1.0, 1.0))
        angles.append(math.degrees(math.acos(min(1.0, abs(front_cosine)))))
        front_cosines.append(front_cosine)
        lateral_cosines.append(lateral_cosine)
    if not angles:
        raise ValueError("cannot compute view geometry from degenerate skeleton")

    angle_array = np.asarray(angles, dtype=np.float64)
    median_angle = float(np.median(angle_array))
    q25, q75 = np.percentile(angle_array, [25.0, 75.0])
    spread = float(q75 - q25)
    frontal_dot = float(np.median(front_cosines))
    lateral_dot = float(np.median(lateral_cosines))
    front_rear = "front" if frontal_dot < 0.0 else "rear"
    # lateral points from the subject's left hip (J1) to right hip (J4).
    # A camera on the positive-lateral side therefore views the subject's right.
    camera_side = "right" if lateral_dot >= 0.0 else "left"
    view = classify_view(median_angle, front_rear, camera_side, spread)
    return {
        "view": view,
        "angle_from_frontal_normal_deg": round(median_angle, 3),
        "angle_iqr_deg": round(spread, 3),
        "angle_min_deg": round(float(angle_array.min()), 3),
        "angle_max_deg": round(float(angle_array.max()), 3),
        "frame_count": len(angles),
        "frontal_axis_dot": round(frontal_dot, 6),
        "lateral_axis_dot": round(lateral_dot, 6),
        "front_rear": front_rear,
        "camera_side": camera_side,
        "view_confidence": round(max(0.0, min(1.0, 1.0 - spread / (2.0 * MAX_VIEW_SPREAD_DEG))), 4),
        "camera_center": [round(float(value), 6) for value in camera_center],
    }


def exercise_metric_groups(
    exercise: str, specs: dict[str, Any]
) -> tuple[list[str], list[str], list[str]]:
    metrics = specs["exercises"][exercise]["metrics"]
    base = sorted(name for name, config in metrics.items() if config.get("views") == "any")
    side = sorted(
        name for name, config in metrics.items() if config.get("views") == "side_or_oblique"
    )
    frontal = sorted(
        name for name, config in metrics.items() if config.get("views") == "frontal_or_oblique"
    )
    return base, side, frontal


def metric_policy(
    exercise: str,
    view: str,
    specs: dict[str, Any],
    side_metrics_enabled: bool = False,
) -> dict[str, list[str] | str | bool]:
    base, side, frontal = exercise_metric_groups(exercise, specs)
    usable = list(base)
    if view in FRONTAL_VIEWS:
        usable.extend(frontal)
    if side_metrics_enabled and view in SIDE_VIEWS | OBLIQUE_VIEWS:
        usable.extend(side)
    all_metrics = list(base) + list(side) + list(frontal)
    disabled = [metric for metric in all_metrics if metric not in usable]
    side_candidates = side if view in SIDE_VIEWS | OBLIQUE_VIEWS else []
    return {
        "usable_metrics": sorted(usable),
        "side_metric_candidates": sorted(side_candidates),
        "disabled_metrics": sorted(disabled),
        "side_metrics_enabled": bool(side_metrics_enabled),
        "side_metrics_status": (
            "enabled"
            if side_metrics_enabled and side_candidates
            else "pending_t1_5b_validation"
            if side_candidates
            else "view_unsupported"
        ),
    }


def build_clip_record(
    dataset_root: Path,
    subject: str,
    exercise: str,
    camera_id: str,
    specs: dict[str, Any],
    overrides: dict[str, str],
) -> dict[str, Any]:
    action = ACTIONS[exercise]
    subject_dir = dataset_root / "train" / subject
    video_path = subject_dir / "videos" / camera_id / f"{exercise}.mp4"
    joints_path = subject_dir / "joints3d_25" / f"{exercise}.json"
    rep_path = subject_dir / "rep_ann.json"
    camera_path = subject_dir / "camera_parameters" / camera_id / f"{exercise}.json"
    for path in (video_path, joints_path, rep_path, camera_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    joints = np.asarray(json.loads(joints_path.read_text())["joints3d_25"], dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (25, 3):
        raise ValueError(f"unexpected joints3d_25 shape: {joints.shape} in {joints_path}")
    rep_payload = json.loads(rep_path.read_text())
    rep_frames = [int(value) for value in rep_payload.get(exercise, [])]
    if not rep_frames:
        raise ValueError(f"missing rep annotation for {exercise}: {rep_path}")
    if rep_frames != sorted(set(rep_frames)) or min(rep_frames) < 0 or max(rep_frames) >= len(joints):
        raise ValueError(f"invalid rep frames for {subject}/{exercise}: {rep_frames}")

    video = video_metadata(video_path)
    camera = load_camera_parameters(camera_path)
    view = view_geometry(joints, sample_rep_range(rep_frames), camera["camera_center"])
    projection = projection_quality(
        joints,
        sample_rep_range(rep_frames),
        camera,
        video["width"],
        video["height"],
    )
    if projection["in_frame_rate"] < MIN_PROJECTION_IN_FRAME_RATE:
        raise ValueError(
            f"camera projection check failed for {subject}/{exercise}/{camera_id}: "
            f"{projection['in_frame_rate']:.3f}"
        )

    clip_id = f"fit3d_{subject}_{exercise}_{camera_id}"
    auto_view = str(view["view"])
    final_view = overrides.get(clip_id, auto_view)
    policy = metric_policy(exercise, final_view, specs)
    split = split_for_subject(subject)
    return {
        "clip_id": clip_id,
        "video_id": f"{subject}_{exercise}_{camera_id}",
        "subject_id": subject,
        "camera_id": camera_id,
        "video_path": str(video_path.resolve()),
        "exercise": exercise,
        "action_display": action["action_display"],
        "action_variant": action["variant"],
        "trim": {"start_frame": 0, "end_frame": len(joints) - 1},
        "fps": video["fps"],
        "frame_count": video["frame_count"],
        "annotation_frames": len(joints),
        "source_size": {"width": video["width"], "height": video["height"]},
        "rotation": "none",
        "view": final_view,
        "auto_view": auto_view,
        "view_source": "manual_override" if final_view != auto_view else "camera_geometry",
        "view_geometry": view,
        "split": split,
        "phase_role": split,
        "selected_for_phase": True,
        "smoke_only": False,
        "has_gt": True,
        "usable_metrics": policy["usable_metrics"],
        "side_metric_candidates": policy["side_metric_candidates"],
        "disabled_metrics": policy["disabled_metrics"],
        "metric_policy": {
            key: policy[key]
            for key in ("side_metrics_enabled", "side_metrics_status")
        },
        "gt": {
            "joints3d_25": str(joints_path.resolve()),
            "rep_frames": rep_frames,
            "rep_frame_unit": "frame_index",
            "camera_parameters": str(camera_path.resolve()),
            "camera_model": camera["model"],
        },
        "quality": {
            "video_minus_annotation_frames": video["frame_count"] - len(joints),
            "projection_in_frame_rate": round(projection["in_frame_rate"], 6),
            "projection_depth_valid_rate": round(projection["depth_valid_rate"], 6),
            "projection_median_visible_joint_span_px": round(
                projection["median_visible_joint_span_px"], 3
            ),
        },
        "notes": action["note"],
    }


def build_test_clip(
    dataset_root: Path,
    subject: str,
    exercise: str,
    specs: dict[str, Any],
) -> dict[str, Any]:
    action = ACTIONS[exercise]
    subject_dir = dataset_root / "test" / subject
    video_path = subject_dir / "videos" / f"{exercise}.mp4"
    camera_path = subject_dir / "camera_parameters" / f"{exercise}.json"
    for path in (video_path, camera_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    video = video_metadata(video_path)
    policy = metric_policy(exercise, "unknown", specs)
    return {
        "clip_id": f"fit3d_{subject}_{exercise}_smoke",
        "video_id": f"{subject}_{exercise}_smoke",
        "subject_id": subject,
        "camera_id": "test_single",
        "video_path": str(video_path.resolve()),
        "exercise": exercise,
        "action_display": action["action_display"],
        "action_variant": action["variant"],
        "trim": {"start_frame": 0, "end_frame": video["frame_count"] - 1},
        "fps": video["fps"],
        "frame_count": video["frame_count"],
        "annotation_frames": None,
        "source_size": {"width": video["width"], "height": video["height"]},
        "rotation": "none",
        "view": "unknown",
        "auto_view": "unknown",
        "view_source": "unavailable_no_gt",
        "view_geometry": None,
        "split": "smoke",
        "phase_role": "smoke",
        "selected_for_phase": False,
        "smoke_only": True,
        "has_gt": False,
        "usable_metrics": policy["usable_metrics"],
        "side_metric_candidates": [],
        "disabled_metrics": policy["disabled_metrics"],
        "metric_policy": {
            "side_metrics_enabled": False,
            "side_metrics_status": "smoke_only_no_gt",
        },
        "gt": None,
        "quality": {"projection_in_frame_rate": None},
        "notes": "Fit3D test has no 3D or rep GT; smoke test only.",
    }


def build_manifest(
    dataset_root: Path,
    specs_path: Path,
    overrides: dict[str, str],
    include_test: bool,
) -> dict[str, Any]:
    specs = json.loads(specs_path.read_text())
    clips: list[dict[str, Any]] = []
    for subject in TRAIN_DEV + TRAIN_HOLDOUT:
        for exercise in ACTIONS:
            for camera_id in CAMERA_IDS:
                clips.append(
                    build_clip_record(
                        dataset_root, subject, exercise, camera_id, specs, overrides
                    )
                )
    if include_test:
        for subject in TEST_SUBJECTS:
            for exercise in ACTIONS:
                clips.append(build_test_clip(dataset_root, subject, exercise, specs))

    train_clips = [clip for clip in clips if clip["has_gt"]]
    view_counts = Counter((clip["exercise"], clip["view"]) for clip in train_clips)
    angles: defaultdict[str, list[float]] = defaultdict(list)
    for clip in train_clips:
        angles[clip["exercise"]].append(
            float(clip["view_geometry"]["angle_from_frontal_normal_deg"])
        )
    view_ranges = {
        exercise: {
            "min_deg": round(min(values), 3),
            "median_deg": round(float(np.median(values)), 3),
            "max_deg": round(max(values), 3),
            "true_side_clip_count": sum(
                1
                for clip in train_clips
                if clip["exercise"] == exercise and clip["view"] in SIDE_VIEWS
            ),
        }
        for exercise, values in angles.items()
    }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "task": "task1-5a",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "Fit3D",
        "dataset_root": str(dataset_root.resolve()),
        "specs": str(specs_path.resolve()),
        "split_policy": {
            "unit": "subject",
            "dev": list(TRAIN_DEV),
            "holdout": list(TRAIN_HOLDOUT),
            "smoke": list(TEST_SUBJECTS) if include_test else [],
        },
        "screening_findings": {
            "train_action_instances": 16,
            "train_camera_clips": len(train_clips),
            "unique_rep_count": sum(
                len(clip["gt"]["rep_frames"])
                for clip in train_clips
                if clip["camera_id"] == CAMERA_IDS[0]
            ),
            "view_counts_by_exercise": {
                exercise: {
                    view: count
                    for (item_exercise, view), count in sorted(view_counts.items())
                    if item_exercise == exercise
                }
                for exercise in ACTIONS
            },
            "view_angle_ranges": view_ranges,
            "action_variants": {
                exercise: action["variant"] for exercise, action in ACTIONS.items()
            },
            "notes": [
                "Fit3D squat is a barbell back squat and does not replace the T1-4 bodyweight squat domain.",
                "Fit3D squat views are frontal/rear/shallow-oblique; no true side view was found.",
                "Fit3D pushup views are side/oblique and can support T1-5B side-view validation.",
                "Side metrics remain disabled in T1-5A pending T1-5B/C validation.",
            ],
        },
        "clips": clips,
    }


def validate_manifest(
    manifest: dict[str, Any],
    specs: dict[str, Any],
    check_files: bool = True,
    require_full_counts: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    clues: list[str] = []
    clips = manifest.get("clips", [])
    ids = [clip.get("clip_id") for clip in clips]
    if len(ids) != len(set(ids)):
        errors.append("clip_id is not unique")

    train = [clip for clip in clips if clip.get("has_gt")]
    smoke = [clip for clip in clips if clip.get("smoke_only")]
    if require_full_counts:
        if len(train) != 64:
            errors.append(f"expected 64 train camera clips, got {len(train)}")
        if len(smoke) != 6:
            errors.append(f"expected 6 smoke clips, got {len(smoke)}")

    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for clip in train:
        groups[(clip.get("subject_id"), clip.get("exercise"))].append(clip)
        trim = clip.get("trim", {})
        annotation_frames = clip.get("annotation_frames")
        if trim.get("start_frame") != 0:
            errors.append(f"{clip.get('clip_id')}: train trim must start at 0")
        if annotation_frames is not None and trim.get("end_frame") != annotation_frames - 1:
            errors.append(f"{clip.get('clip_id')}: trim.end_frame must equal N - 1")
        if clip.get("frame_count", -1) < int(annotation_frames or 0):
            errors.append(f"{clip.get('clip_id')}: video is shorter than annotation")
        if clip.get("view") not in VIEWS:
            errors.append(f"{clip.get('clip_id')}: invalid view {clip.get('view')!r}")
        if clip.get("split") != split_for_subject(str(clip.get("subject_id"))):
            errors.append(f"{clip.get('clip_id')}: split does not match subject policy")
        usable = set(clip.get("usable_metrics", []))
        disabled = set(clip.get("disabled_metrics", []))
        candidates = set(clip.get("side_metric_candidates", []))
        if usable & disabled:
            errors.append(f"{clip.get('clip_id')}: metric cannot be both usable and disabled")
        if not candidates <= disabled:
            errors.append(f"{clip.get('clip_id')}: side candidates must remain disabled in T1-5A")
        base, side, frontal = exercise_metric_groups(str(clip.get("exercise")), specs)
        if usable | disabled != set(base) | set(side) | set(frontal):
            errors.append(f"{clip.get('clip_id')}: metric policy does not cover all metrics")
        gt = clip.get("gt") or {}
        rep_frames = gt.get("rep_frames") or []
        if rep_frames != sorted(set(rep_frames)):
            errors.append(f"{clip.get('clip_id')}: rep frames must be strictly increasing")
        if rep_frames and (
            min(rep_frames) < 0 or max(rep_frames) >= int(annotation_frames or 0)
        ):
            errors.append(f"{clip.get('clip_id')}: rep frame outside annotation range")
        projection_rate = clip.get("quality", {}).get("projection_in_frame_rate")
        if projection_rate is None or projection_rate < MIN_PROJECTION_IN_FRAME_RATE:
            errors.append(f"{clip.get('clip_id')}: weak camera projection check")
        if check_files:
            for path in (
                clip.get("video_path"),
                gt.get("joints3d_25"),
                gt.get("camera_parameters"),
            ):
                if not path or not Path(path).is_file():
                    errors.append(f"{clip.get('clip_id')}: missing file {path}")

    for (subject, exercise), group in groups.items():
        if len(group) != 4:
            errors.append(f"{subject}/{exercise}: expected 4 camera clips, got {len(group)}")
        if {clip.get("camera_id") for clip in group} != set(CAMERA_IDS):
            errors.append(f"{subject}/{exercise}: camera ids do not match Fit3D cameras")
        annotation_counts = {clip.get("annotation_frames") for clip in group}
        if len(annotation_counts) != 1:
            errors.append(f"{subject}/{exercise}: annotation lengths differ across cameras")
        rep_sequences = {
            tuple((clip.get("gt") or {}).get("rep_frames", [])) for clip in group
        }
        if len(rep_sequences) != 1:
            errors.append(f"{subject}/{exercise}: rep frames differ across cameras")
    if require_full_counts and len(groups) != 16:
        errors.append(f"expected 16 train subject-action instances, got {len(groups)}")

    for clip in smoke:
        if clip.get("gt") is not None:
            errors.append(f"{clip.get('clip_id')}: smoke clip must not expose GT")
        if clip.get("selected_for_phase"):
            errors.append(f"{clip.get('clip_id')}: smoke clip must not enter phase selection")
        if clip.get("view") != "unknown":
            errors.append(f"{clip.get('clip_id')}: smoke view must remain unknown")

    if require_full_counts:
        dev_subjects = {clip.get("subject_id") for clip in train if clip.get("split") == "dev"}
        holdout_subjects = {
            clip.get("subject_id") for clip in train if clip.get("split") == "holdout"
        }
        if dev_subjects != set(TRAIN_DEV):
            errors.append(f"dev subjects mismatch: {sorted(dev_subjects)}")
        if holdout_subjects != set(TRAIN_HOLDOUT):
            errors.append(f"holdout subjects mismatch: {sorted(holdout_subjects)}")
        if dev_subjects & holdout_subjects:
            errors.append("dev and holdout subjects overlap")
        unique_reps = sum(
            len((clip.get("gt") or {}).get("rep_frames", []))
            for clip in train
            if clip.get("camera_id") == CAMERA_IDS[0]
        )
        if unique_reps != 96:
            errors.append(f"expected 96 unique reps, got {unique_reps}")

    view_counts = Counter(clip.get("view") for clip in train)
    clues.extend(
        [
            f"train_clips={len(train)}",
            f"smoke_clips={len(smoke)}",
            f"action_instances={len(groups)}",
            f"unique_reps={sum(len((clip.get('gt') or {}).get('rep_frames', [])) for clip in train if clip.get('camera_id') == CAMERA_IDS[0])}",
            f"view_counts={dict(sorted(view_counts.items()))}",
        ]
    )
    return {"passed": not errors, "errors": errors, "checks": clues}


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise ValueError(f"dataset root is not a directory: {dataset_root}")
    specs = json.loads(args.specs.read_text())
    overrides = load_view_overrides(args.view_overrides)
    manifest = build_manifest(dataset_root, args.specs.resolve(), overrides, args.include_test)
    validation = validate_manifest(manifest, specs, check_files=True, require_full_counts=True)
    validation.update(
        {
            "schema_version": "task1-5a-validation-v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_root": str(dataset_root),
            "view_overrides": str(args.view_overrides.resolve()) if args.view_overrides else None,
        }
    )
    write_json(args.output, manifest)
    write_json(args.results_dir / "validation.json", validation)
    if not validation["passed"]:
        raise SystemExit("manifest validation failed:\n- " + "\n- ".join(validation["errors"]))
    print(
        f"wrote {args.output} ({len(manifest['clips'])} clips); "
        f"validation={validation['checks']}"
    )


if __name__ == "__main__":
    main()
