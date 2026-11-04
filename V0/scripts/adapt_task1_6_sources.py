#!/usr/bin/env python3
"""Adapt Fit3D/QEVD sources into the frozen T1-6 evaluation contract.

This adapter only maps paths, frame coordinates, GT semantics and split roles.
It never invokes the rep segmenter and never edits segmentation outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FIT3D_MANIFEST = ROOT / "configs" / "task1_5_fit3d_clips.json"
DEFAULT_QEVD_MANIFEST = ROOT / "configs" / "task1_4_clips.json"
DEFAULT_FIRST_ROUND_ANNOTATIONS = (
    ROOT / "results" / "segmentation" / "annotation" / "annotation_template.csv"
)
DEFAULT_BLIND_ANNOTATIONS = (
    ROOT / "results" / "segmentation" / "annotation_blind" / "blind_reps.csv"
)
DEFAULT_FEEDBACK = (
    ROOT.parent
    / "dataset"
    / "QEVD-FIT"
    / "QEVD-FIT-COACH"
    / "feedbacks_long_range.json"
)
DEFAULT_BASELINE_SUMMARY = (
    ROOT / "results" / "task1_5" / "rep_evaluation" / "summary.json"
)
DEFAULT_SOURCES_OUTPUT = ROOT / "configs" / "task1_6_sources.json"
DEFAULT_PROTOCOL_OUTPUT = ROOT / "configs" / "task1_6_eval_protocol.json"
DEFAULT_HOLDOUT_DIR = ROOT / "results" / "task1_6" / "manifest"

SCHEMA_VERSION = "task1-6-sources-v1"
PROTOCOL_SCHEMA_VERSION = "task1-6-eval-protocol-v1"

FIT3D_EXPECTED = {
    "clips": 70,
    "gt_train_clips": 64,
    "subject_action_clusters": 16,
    "unique_rep_events": 96,
    "dev_clips": 32,
    "holdout_clips": 32,
    "smoke_clips": 6,
}
QEVD_EXPECTED_BASE_CLIPS = 41
QEVD_EXPECTED_SEALED_SEGMENTS = 6

SEALED_TARGET_CLIP_IDS = (
    "0152_pushup_set01",
    "0159_pushup_set01",
    "0127_squat_set01",
    "0177_squat_set01",
)
SEALED_NON_TARGET_REFS = (
    ("0207", "air_jump_rope"),
    ("0207", "good_morning"),
)

T1_5D_BASELINE = {
    "manifest_clip_count": 64,
    "action_instances": 16,
    "unique_gt_reps": 96,
    "consensus_count_mae": 0.625,
    "consensus_exact_count_correct": 6,
    "primary_event_median_abs_error_frames": 12.5,
    "primary_event_p90_abs_error_frames": 25.8,
    "primary_high_confidence_fp_count": 24,
    "all_camera_high_confidence_fp_count": 80,
    "top_to_top_interval_median_abs_error_frames": 3.0,
}

ANNOTATION_TEMPLATE_FIELDS = [
    "segment_id",
    "video_id",
    "action",
    "rep_id",
    "window_start_frame",
    "window_end_frame",
    "draft_start_frame",
    "draft_end_frame",
    "gt_start_frame",
    "gt_end_frame",
    "frame_unit",
    "phase_landmark",
    "rep_unit",
    "review_status",
    "boundary_confidence",
    "annotator",
    "reviewed_at",
    "notes",
]

CONTACT_FRACTIONS = (0.00, 0.12, 0.25, 0.38, 0.50, 0.62, 0.75, 0.88)

ACTION_PATTERNS = (
    re.compile(r"first up (?:are|is) (.+?)[!.]", re.IGNORECASE),
    re.compile(r"moving on to (.+?)[!.]", re.IGNORECASE),
)

ROTATION_CODES = {
    "none": None,
    "90cw": cv2.ROTATE_90_CLOCKWISE,
    "180": cv2.ROTATE_180,
    "90ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit3d-manifest", type=Path, default=DEFAULT_FIT3D_MANIFEST)
    parser.add_argument("--qevd-manifest", type=Path, default=DEFAULT_QEVD_MANIFEST)
    parser.add_argument(
        "--first-round-annotations",
        type=Path,
        default=DEFAULT_FIRST_ROUND_ANNOTATIONS,
    )
    parser.add_argument(
        "--blind-annotations", type=Path, default=DEFAULT_BLIND_ANNOTATIONS
    )
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument(
        "--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY
    )
    parser.add_argument("--sources-output", type=Path, default=DEFAULT_SOURCES_OUTPUT)
    parser.add_argument("--protocol-output", type=Path, default=DEFAULT_PROTOCOL_OUTPUT)
    parser.add_argument("--holdout-dir", type=Path, default=DEFAULT_HOLDOUT_DIR)
    parser.add_argument("--skip-file-checks", action="store_true")
    parser.add_argument(
        "--skip-holdout-artifacts",
        action="store_true",
        help="Do not write the sealed-holdout CSV/contact sheet artifacts",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(payload: Any) -> str:
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def video_metadata(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    try:
        return {
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


def checked_frames(frame: int, trim: dict[str, Any], label: str) -> dict[str, Any]:
    start = int(trim["start_frame"])
    end = int(trim["end_frame"])
    frame = int(frame)
    if not start <= frame <= end:
        raise ValueError(f"{label}: frame {frame} outside trim [{start}, {end}]")
    return {
        "frame": frame,
        "frame_unit": "source_video_absolute",
        "clip_frame": frame - start,
    }


def normalize_action_text(text: str) -> tuple[str, str]:
    display = re.sub(r"\s+", " ", text.strip().strip(" .!")).lower()
    aliases = (
        ("jumping jack", "jumping_jack"),
        ("high knee", "high_knees"),
        ("butt kicker", "butt_kickers"),
        ("lunge jump", "lunge_jumps"),
        ("walking lunge", "walking_lunges"),
        ("mountain climber", "mountain_climbers"),
        ("good morning", "good_morning"),
        ("air jump rope", "air_jump_rope"),
        ("standing kick", "standing_kicks"),
        ("boxing squat punch", "boxing_squat_punches"),
        ("toe toucher", "toe_touchers"),
        ("shoulder gator", "shoulder_gators"),
        ("squat jump", "squat_jumps"),
        ("squat kick", "squat_kicks"),
        ("squat", "squat"),
        ("pushup", "pushup"),
        ("push up", "pushup"),
    )
    compact = display.replace("-", " ").replace("_", " ")
    for pattern, action in aliases:
        if pattern in compact:
            return action, display
    return re.sub(r"[^a-z0-9]+", "_", compact).strip("_"), display


def load_feedback_segments(
    feedback_path: Path, video_ids: Iterable[str]
) -> list[dict[str, Any]]:
    payload = load_json(feedback_path)
    by_video = {
        Path(item["long_range_video_file"]).stem: item for item in payload
    }
    selected: list[dict[str, Any]] = []
    for video_id in video_ids:
        if video_id not in by_video:
            raise ValueError(f"feedback JSON has no record for video {video_id}")
        record = by_video[video_id]
        timestamps_path = feedback_path.parent / record["video_timestamps"]
        timestamps = np.load(timestamps_path).astype(np.float64) / 1e9
        feedbacks = record.get("feedbacks", [])
        transition_frames: list[int] = []
        for timestamp, is_transition in zip(
            record.get("feedback_timestamps", []),
            record.get("is_transition", []),
        ):
            if not is_transition:
                continue
            shifted = float(timestamp) - 28800.0
            transition_frames.append(int(np.argmin(np.abs(timestamps - shifted))))
        transition_frames = sorted(set(transition_frames))
        for index, start in enumerate(transition_frames):
            end = (
                transition_frames[index + 1] - 1
                if index + 1 < len(transition_frames)
                else len(feedbacks) - 1
            )
            action_text = None
            for feedback in feedbacks[start : end + 1]:
                if not feedback:
                    continue
                for pattern in ACTION_PATTERNS:
                    match = pattern.search(feedback)
                    if match:
                        action_text = match.group(1).strip()
                        break
                if action_text:
                    break
            if action_text is None or end < start:
                continue
            action, display = normalize_action_text(action_text)
            selected.append(
                {
                    "video_id": video_id,
                    "action": action,
                    "action_display": display,
                    "start_frame": int(start),
                    "end_frame": int(end),
                }
            )
    return selected


def read_annotation_set(path: Path, annotation_set_id: str) -> dict[str, Any]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    accepted: list[dict[str, Any]] = []
    rejected_count = 0
    seen: set[tuple[str, str]] = set()
    for row in rows:
        status = str(row.get("review_status", "")).strip().lower()
        if status not in {"accept", "adjust"}:
            rejected_count += 1
            continue
        start_text = str(row.get("gt_start_frame", "")).strip()
        end_text = str(row.get("gt_end_frame", "")).strip()
        if not start_text or not end_text:
            rejected_count += 1
            continue
        key = (str(row["segment_id"]), str(row["rep_id"]))
        if key in seen:
            raise ValueError(f"duplicate annotation row in {path}: {key}")
        seen.add(key)
        start = int(start_text)
        end = int(end_text)
        if start > end:
            raise ValueError(f"invalid annotation interval in {path}: {key}")
        accepted.append(
            {
                "segment_id": str(row["segment_id"]),
                "video_id": str(row["video_id"]).zfill(4),
                "action": str(row["action"]),
                "rep_id": int(row["rep_id"]),
                "start_frame": start,
                "end_frame": end,
                "frame_unit": "source_video_absolute",
                "rep_unit": str(row.get("rep_unit", "")).strip() or "cycle",
                "phase_landmark": str(row.get("phase_landmark", "")).strip(),
                "boundary_confidence": str(
                    row.get("boundary_confidence", "")
                ).strip(),
                "review_status": status,
                "annotator": str(row.get("annotator", "")).strip(),
                "reviewed_at": str(row.get("reviewed_at", "")).strip(),
                "notes": str(row.get("notes", "")).strip(),
            }
        )
    accepted.sort(
        key=lambda item: (item["segment_id"], item["rep_id"], item["start_frame"])
    )
    return {
        "annotation_set_id": annotation_set_id,
        "source_path": str(path),
        "source_sha256": sha256_path(path),
        "gt_semantics": "rep_interval",
        "row_count": len(rows),
        "accepted_count": len(accepted),
        "excluded_or_incomplete_count": rejected_count,
        "segments": sorted({item["segment_id"] for item in accepted}),
        "intervals": accepted,
    }


def fit3d_clip_record(
    clip: dict[str, Any], manifest_sha256: str
) -> dict[str, Any]:
    trim = dict(clip["trim"])
    gt = clip.get("gt") or {}
    events = []
    for index, frame in enumerate(gt.get("rep_frames") or [], start=1):
        event = checked_frames(
            int(frame), trim, f"Fit3D {clip['clip_id']} rep event {index}"
        )
        events.append({"event_index": index, "event": "top", **event})
    return {
        "source_id": "fit3d_train",
        "dataset": "Fit3D",
        "clip_id": clip["clip_id"],
        "video_id": clip["video_id"],
        "subject_id": clip.get("subject_id"),
        "camera_id": clip.get("camera_id"),
        "cluster_id": f"subject_action:{clip.get('subject_id')}:{clip['exercise']}",
        "evaluation_unit": "subject_action",
        "action": clip["exercise"],
        "action_display": clip.get("action_display", clip["exercise"]),
        "action_variant": clip.get("action_variant"),
        "video_path": clip["video_path"],
        "trim": trim,
        "fps": float(clip["fps"]),
        "frame_count": int(clip["frame_count"]),
        "view": clip.get("view"),
        "rotation": clip.get("rotation"),
        "split_role": clip.get("split"),
        "source_split_role": clip.get("split"),
        "phase_role": clip.get("phase_role"),
        "smoke_only": bool(clip.get("smoke_only")),
        "selected_for_phase": bool(clip.get("selected_for_phase")),
        "has_gt": bool(events),
        "gt": {
            "semantics": "top_event" if events else "none",
            "status": "available" if events else "not_applicable",
            "events": events,
            "rep_intervals": [],
            "bottom_status": "not_computable_no_bottom_gt",
        },
        "source_manifest_sha256": manifest_sha256,
    }


def sealed_segment_id(video_id: str, action: str) -> str:
    return f"qeud_{video_id}_{action}_sealed01"


def qevd_clip_record(
    clip: dict[str, Any],
    manifest_sha256: str,
    annotation_sets: list[dict[str, Any]],
) -> dict[str, Any]:
    trim = dict(clip["trim"])
    matching_sets = []
    for annotation_set in annotation_sets:
        intervals = [
            item
            for item in annotation_set["intervals"]
            if item["video_id"] == str(clip["video_id"]).zfill(4)
            and item["action"] == clip["exercise"]
        ]
        if intervals:
            matching_sets.append(
                {
                    "annotation_set_id": annotation_set["annotation_set_id"],
                    "intervals": intervals,
                }
            )
    source_split = clip.get("split")
    split_role = (
        "sealed_holdout"
        if clip["clip_id"] in SEALED_TARGET_CLIP_IDS and source_split == "reserve"
        else source_split
    )
    if split_role == "sealed_holdout":
        matching_sets = []
    gt_available = bool(matching_sets)
    return {
        "source_id": "qeud_fit_coach",
        "dataset": "QEVD-FIT-COACH",
        "clip_id": clip["clip_id"],
        "video_id": clip["video_id"],
        "subject_id": None,
        "camera_id": None,
        "cluster_id": f"video:{clip['video_id']}",
        "evaluation_unit": "video_or_segment",
        "action": clip["exercise"],
        "action_display": clip.get("action_display", clip["exercise"]),
        "action_variant": None,
        "video_path": clip["video_path"],
        "trim": trim,
        "fps": float(clip["fps"]),
        "frame_count": int(clip["frame_count"]),
        "view": clip.get("view"),
        "rotation": clip.get("rotation"),
        "split_role": split_role,
        "source_split_role": source_split,
        "phase_role": clip.get("phase_role"),
        "smoke_only": False,
        "selected_for_phase": bool(clip.get("selected_for_phase")),
        "has_gt": gt_available,
        "gt": {
            "semantics": "rep_interval" if gt_available else "none",
            "status": (
                "sealed_unannotated"
                if split_role == "sealed_holdout"
                else "legacy_sets_available"
                if gt_available
                else "unannotated"
            ),
            "events": [],
            "rep_intervals": [],
            "annotation_sets": matching_sets,
            "bottom_status": "legacy_interval_has_no_phase_state_labels",
        },
        "source_manifest_sha256": manifest_sha256,
    }


def sealed_non_target_record(
    segment: dict[str, Any],
    feedback_sha256: str,
    check_files: bool,
) -> dict[str, Any]:
    video_id = segment["video_id"]
    video_path = DEFAULT_FEEDBACK.parent / "long_range_videos" / f"{video_id}.mp4"
    metadata = (
        video_metadata(video_path)
        if check_files
        else {
            "frame_count": int(segment["end_frame"]) + 1,
            "fps": 30.0,
            "width": 640,
            "height": 360,
        }
    )
    trim = {
        "start_frame": int(segment["start_frame"]),
        "end_frame": int(segment["end_frame"]),
    }
    return {
        "source_id": "qeud_fit_coach",
        "dataset": "QEVD-FIT-COACH",
        "clip_id": sealed_segment_id(video_id, segment["action"]),
        "video_id": video_id,
        "subject_id": None,
        "camera_id": None,
        "cluster_id": f"video:{video_id}",
        "evaluation_unit": "video_or_segment",
        "action": segment["action"],
        "action_display": segment["action_display"],
        "action_variant": None,
        "video_path": str(video_path),
        "trim": trim,
        "fps": float(metadata["fps"]),
        "frame_count": int(metadata["frame_count"]),
        "view": "unknown",
        "rotation": "none",
        "split_role": "sealed_holdout",
        "source_split_role": "sealed_holdout",
        "phase_role": "sealed_holdout",
        "smoke_only": False,
        "selected_for_phase": False,
        "has_gt": False,
        "gt": {
            "semantics": "none",
            "status": "sealed_unannotated",
            "events": [],
            "rep_intervals": [],
            "annotation_sets": [],
            "bottom_status": "not_annotated",
        },
        "source_manifest_sha256": feedback_sha256,
    }


def validate_sources_payload(
    payload: dict[str, Any], check_files: bool
) -> dict[str, Any]:
    checks: list[str] = []
    errors: list[str] = []
    clips = payload["clips"]
    by_source: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for clip in clips:
        by_source[clip["source_id"]].append(clip)

    clip_ids = [clip["clip_id"] for clip in clips]
    if len(clip_ids) != len(set(clip_ids)):
        errors.append("clip_id values are not unique")
    else:
        checks.append(f"unique_clip_ids={len(clip_ids)}")

    metadata_cache: dict[str, dict[str, Any]] = {}
    for clip in clips:
        trim = clip["trim"]
        start = int(trim["start_frame"])
        end = int(trim["end_frame"])
        if not 0 <= start <= end < int(clip["frame_count"]):
            errors.append(f"{clip['clip_id']}: invalid trim/frame_count")
            continue
        if clip["fps"] <= 0:
            errors.append(f"{clip['clip_id']}: fps must be positive")
        frames = [item["frame"] for item in clip["gt"]["events"]]
        if frames != sorted(frames) or len(frames) != len(set(frames)):
            errors.append(f"{clip['clip_id']}: GT top events are not strictly ordered")
        for event in clip["gt"]["events"]:
            if not start <= int(event["frame"]) <= end:
                errors.append(f"{clip['clip_id']}: GT event outside trim")
            if int(event["clip_frame"]) != int(event["frame"]) - start:
                errors.append(f"{clip['clip_id']}: GT clip-frame conversion mismatch")
        path = Path(clip["video_path"])
        if check_files:
            if str(path) not in metadata_cache:
                metadata_cache[str(path)] = video_metadata(path)
            metadata = metadata_cache[str(path)]
            if int(metadata["frame_count"]) != int(clip["frame_count"]):
                errors.append(f"{clip['clip_id']}: declared frame_count mismatch")
            if abs(float(metadata["fps"]) - float(clip["fps"])) > 0.01:
                errors.append(f"{clip['clip_id']}: declared fps mismatch")
    checks.append(f"trim_and_frame_contract_checked={len(clips)}")

    fit3d = by_source["fit3d_train"]
    fit3d_train = [
        clip for clip in fit3d if clip["has_gt"] and not clip["smoke_only"]
    ]
    fit3d_smoke = [clip for clip in fit3d if clip["smoke_only"]]
    clusters: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for clip in fit3d_train:
        clusters[clip["cluster_id"]].append(clip)
    unique_rep_events = sum(
        len(cluster[0]["gt"]["events"]) for cluster in clusters.values()
    )
    fit3d_counts = {
        "clips": len(fit3d),
        "gt_train_clips": len(fit3d_train),
        "subject_action_clusters": len(clusters),
        "unique_rep_events": unique_rep_events,
        "dev_clips": sum(clip["split_role"] == "dev" for clip in fit3d),
        "holdout_clips": sum(clip["split_role"] == "holdout" for clip in fit3d),
        "smoke_clips": len(fit3d_smoke),
    }
    for key, expected in FIT3D_EXPECTED.items():
        if fit3d_counts[key] != expected:
            errors.append(f"Fit3D {key}: expected {expected}, found {fit3d_counts[key]}")
    if any(clip["has_gt"] for clip in fit3d_smoke):
        errors.append("Fit3D smoke clip unexpectedly has GT")
    if any(clip["split_role"] not in {"dev", "holdout", "smoke"} for clip in fit3d):
        errors.append("Fit3D contains an unexpected split role")
    for cluster_id, cluster in clusters.items():
        if len(cluster) != 4:
            errors.append(f"Fit3D cluster {cluster_id}: expected 4 cameras")
        event_lists = {
            tuple(event["frame"] for event in clip["gt"]["events"])
            for clip in cluster
        }
        if len(event_lists) != 1:
            errors.append(f"Fit3D cluster {cluster_id}: camera GT events disagree")
    checks.append("fit3d_cluster_gt_consistent=16x4")

    qevd = by_source["qeud_fit_coach"]
    qevd_roles = {clip["split_role"] for clip in qevd}
    if not qevd_roles <= {"dev", "holdout", "reserve", "sealed_holdout"}:
        errors.append(f"QEVD contains unexpected split roles: {sorted(qevd_roles)}")
    video_roles: defaultdict[str, set[str]] = defaultdict(set)
    for clip in qevd:
        video_roles[clip["video_id"]].add(clip["split_role"])
    mixed_videos = {
        video_id: sorted(roles)
        for video_id, roles in video_roles.items()
        if len(roles) > 1
    }
    if mixed_videos:
        errors.append(f"QEVD video crosses split roles: {mixed_videos}")
    sealed = [clip for clip in qevd if clip["split_role"] == "sealed_holdout"]
    sealed_actions = Counter(clip["action"] for clip in sealed)
    sealed_nontarget_actions = {
        clip["action"] for clip in sealed if clip["action"] not in {"pushup", "squat"}
    }
    if len(sealed) != QEVD_EXPECTED_SEALED_SEGMENTS:
        errors.append(
            f"QEVD sealed segments: expected {QEVD_EXPECTED_SEALED_SEGMENTS}, "
            f"found {len(sealed)}"
        )
    if sealed_actions["pushup"] < 2 or sealed_actions["squat"] < 2:
        errors.append("QEVD sealed set must contain at least 2 pushup and 2 squat segments")
    if len(sealed_nontarget_actions) < 2:
        errors.append("QEVD sealed set must contain at least 2 non-target cycle actions")
    if any(video_roles[clip["video_id"]] & {"dev", "holdout"} for clip in sealed):
        errors.append("QEVD sealed video overlaps an observed dev/holdout video")
    qevd_counts = {
        "base_manifest_clips": sum(
            clip["source_split_role"] in {"dev", "holdout", "reserve"}
            for clip in qevd
        ),
        "clips_with_sealed_additions": len(qevd),
        "dev": sum(clip["split_role"] == "dev" for clip in qevd),
        "holdout": sum(clip["split_role"] == "holdout" for clip in qevd),
        "reserve": sum(clip["split_role"] == "reserve" for clip in qevd),
        "sealed_holdout": len(sealed),
        "sealed_action_counts": dict(sorted(sealed_actions.items())),
    }
    if qevd_counts["base_manifest_clips"] != QEVD_EXPECTED_BASE_CLIPS:
        errors.append(
            "QEVD base manifest clip count changed: "
            f"expected {QEVD_EXPECTED_BASE_CLIPS}, "
            f"found {qevd_counts['base_manifest_clips']}"
        )
    checks.append("qeud_video_split_isolation=passed")

    annotation_counts = {}
    for annotation_set in payload["legacy_annotation_sets"]:
        intervals = annotation_set["intervals"]
        if any(item["start_frame"] > item["end_frame"] for item in intervals):
            errors.append(f"{annotation_set['annotation_set_id']}: invalid interval")
        keys = [(item["segment_id"], item["rep_id"]) for item in intervals]
        if len(keys) != len(set(keys)):
            errors.append(f"{annotation_set['annotation_set_id']}: duplicate rep key")
        if not annotation_set["segments"]:
            errors.append(f"{annotation_set['annotation_set_id']}: no accepted segments")
        annotation_counts[annotation_set["annotation_set_id"]] = {
            "accepted_intervals": len(intervals),
            "segments": len(annotation_set["segments"]),
        }
    checks.append("legacy_annotation_interval_semantics=rep_interval")

    passed = not errors
    return {
        "passed": passed,
        "checks": checks,
        "errors": errors,
        "counts": {
            "total_clips": len(clips),
            "by_source": {key: len(value) for key, value in sorted(by_source.items())},
            "fit3d": fit3d_counts,
            "qeud": qevd_counts,
            "legacy_annotations": annotation_counts,
        },
    }


def nested_value(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        value = value[part]
    return value


def validate_t1_5d_baseline(
    summary_path: Path,
    fit3d_manifest_sha256: str,
) -> dict[str, Any]:
    summary = load_json(summary_path)
    checks: list[str] = []
    errors: list[str] = []
    fields = {
        "manifest_clip_count": "dataset.train_clips",
        "action_instances": "dataset.action_instances",
        "unique_gt_reps": "dataset.unique_gt_reps",
        "consensus_count_mae": "results.consensus_count_primary.count_mae",
        "consensus_exact_count_correct": (
            "results.consensus_count_primary.exact_count_correct"
        ),
        "primary_event_median_abs_error_frames": (
            "results.events_primary_camera.event_abs_error_median_frames"
        ),
        "primary_event_p90_abs_error_frames": (
            "results.events_primary_camera.event_abs_error_p90_frames"
        ),
        "primary_high_confidence_fp_count": (
            "results.events_primary_camera.high_confidence_fp_count"
        ),
        "all_camera_high_confidence_fp_count": (
            "results.events_all_cameras_diagnostic.high_confidence_fp_count"
        ),
        "top_to_top_interval_median_abs_error_frames": (
            "results.interval_auxiliary.interval_abs_error_median_frames"
        ),
    }
    observed: dict[str, Any] = {}
    for name, json_path in fields.items():
        try:
            value = nested_value(summary, json_path)
        except (KeyError, TypeError):
            errors.append(f"baseline summary missing {json_path}")
            continue
        observed[name] = value
        expected = T1_5D_BASELINE[name]
        if not math.isclose(float(value), float(expected), rel_tol=0.0, abs_tol=1e-9):
            errors.append(f"baseline {name}: expected {expected}, found {value}")
    source_manifest_sha = (summary.get("source") or {}).get("manifest_sha256")
    if source_manifest_sha != fit3d_manifest_sha256:
        errors.append("baseline summary was produced from a different Fit3D manifest")
    if not errors:
        checks.append("t1_5d_baseline_values_match_frozen_contract")
    return {
        "passed": not errors,
        "checks": checks,
        "errors": errors,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_path(summary_path),
        "expected": T1_5D_BASELINE,
        "observed": observed,
        "source_manifest_sha256": source_manifest_sha,
    }


def build_protocol_payload(
    sources: dict[str, Any],
    baseline: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    sealed = [
        {
            "clip_id": clip["clip_id"],
            "video_id": clip["video_id"],
            "action": clip["action"],
            "trim": clip["trim"],
            "annotation_status": "pending_human_labels",
        }
        for clip in sources["clips"]
        if clip["split_role"] == "sealed_holdout"
    ]
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "task": "T1-6",
        "stage": "T1-6A",
        "generated_at": generated_at,
        "sources_sha256": payload_sha256(sources),
        "source_manifest_sha256": sources["source_manifests"],
        "algorithm_core": {
            "contract_version": "repchain-v2",
            "rep_unit": "cycle",
            "dataset_independent": True,
            "prohibited_features": [
                "dataset_name",
                "video_id",
                "subject_id",
                "camera_id",
                "fixed_rep_count",
                "fixed_frame_offset",
                "fixed_plateau_width",
                "camera_order_or_consensus",
                "sealed_holdout_gt_tuning",
            ],
            "selection_inputs": [
                "phase_signal",
                "state_intervals",
                "chain_shape_cost",
                "chain_timing_cost",
                "chain_confidence",
            ],
            "forbidden_core_constants": [
                "Fit3D",
                "QEVD",
                "s03",
                "s11",
                "50591643",
                "58860488",
                "60457274",
                "65906101",
                "+13",
                "6_reps",
                "camera_median",
            ],
        },
        "boundary_semantics": {
            "anchors": [
                "plateau_entry",
                "plateau_midpoint",
                "plateau_exit",
                "phase_zero_crossing",
            ],
            "default_anchor": "plateau_midpoint",
            "selection_stage": "T1-6B",
            "selection_scope": "single_global_policy",
            "holdout_rule": "anchor_policy_cannot_change_after_holdout_start",
            "fit3d_gt": "top_or_extension_event_only",
            "fit3d_bottom_gt": "not_available",
            "qeud_legacy_gt": "rep_interval_not_phase_state",
        },
        "evaluation": {
            "fit3d_primary_unit": "subject_action",
            "fit3d_camera_observations": (
                "repeated_observations_not_independent_samples"
            ),
            "qeud_primary_unit": "segment_or_video",
            "smoke_rule": "never_enter_error_statistics",
            "event_matching": {
                "method": "one_to_one_hungarian",
                "prediction": "accepted_complete_rep_anchor_frame",
                "gate": (
                    "0.25_x_median_gt_top_to_top_interval_round_half_up_min_1"
                ),
            },
            "confidence_layers": ["complete", "in_set", "combined"],
            "high_confidence_threshold": 0.5,
            "rejection_reasons": [
                "insufficient_visibility",
                "phase_ambiguous",
                "incomplete_rep",
                "outside_set",
                "shape_mismatch",
                "timing_outlier",
                "no_confident_set",
            ],
            "split_policy": {
                "fit3d": "subject",
                "qeud": "video",
                "question_action_segments_share_video_split": True,
                "sealed_holdout_annotation_required_before_metrics": True,
            },
        },
        "acceptance": {
            "fit3d": {
                "count_mae_lte": 0.5,
                "exact_count_at_least": "14/16",
                "first_event_recall_gte": 0.75,
                "last_event_recall_gte": 0.75,
                "primary_event_median_abs_error_frames_lte": 5,
                "primary_event_p90_abs_error_frames_lte": 12,
                "primary_high_confidence_fp": 0,
                "all_camera_high_confidence_fp": 0,
                "top_to_top_interval_median_abs_error_must_not_degrade": True,
            },
            "qeud_sealed": {
                "count_exact_at_least": "5/6 segments",
                "count_mae_lte": 0.25,
                "boundary_abs_error_median_frames_lte": 5,
                "boundary_abs_error_p95_frames_lte": 12,
                "f1_at_iou_0_30_gte": 0.95,
                "high_confidence_fp": 0,
                "non_rep_output_as_complete_rep": 0,
            },
        },
        "t1_5d_baseline": baseline,
        "sealed_holdout": {
            "dataset": "QEVD-FIT-COACH",
            "selection_scope": "video_ids_outside_t1_4_dev_and_holdout",
            "target_segments": 4,
            "non_target_segments": 2,
            "segments": sealed,
            "annotation_status": "pending_human_labels",
            "cross_domain_acceptance_complete": False,
            "reason_not_complete": "sealed holdout GT has not been human-annotated",
        },
        "generalization_claims": {
            "baseline_reproduced": baseline["passed"],
            "cross_domain_acceptance_complete": False,
            "allowed_claim": "protocol_and_adapters_frozen_baseline_reproduced",
            "disallowed_claim": (
                "general_segmenter_has_passed_cross_dataset_acceptance"
            ),
        },
    }


def rotate_frame(frame: np.ndarray, rotation: str) -> np.ndarray:
    if rotation not in ROTATION_CODES:
        raise ValueError(f"unsupported rotation: {rotation}")
    code = ROTATION_CODES[rotation]
    return frame if code is None else cv2.rotate(frame, code)


def letterbox(image: np.ndarray, width: int = 240, height: int = 180) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def write_contact_sheet(
    sealed_clips: list[dict[str, Any]], output_path: Path
) -> None:
    rows: list[np.ndarray] = []
    label_height = 30
    thumb_width = 240
    thumb_height = 180
    for clip in sealed_clips:
        cap = cv2.VideoCapture(clip["video_path"])
        if not cap.isOpened():
            raise ValueError(f"cannot open sealed video: {clip['video_path']}")
        start = int(clip["trim"]["start_frame"])
        end = int(clip["trim"]["end_frame"])
        frames = []
        for fraction in CONTACT_FRACTIONS:
            frame_index = int(round(start + (end - start) * fraction))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(
                    f"cannot read {clip['clip_id']} frame {frame_index}"
                )
            frames.append(letterbox(rotate_frame(frame, clip["rotation"])))
        cap.release()
        row = np.zeros(
            (label_height + thumb_height, thumb_width * len(frames), 3),
            dtype=np.uint8,
        )
        row[:label_height, :] = (30, 30, 30)
        label = (
            f"{clip['clip_id']} | {start}-{end} | "
            f"{clip['rotation']} | {clip['fps']:.0f}fps"
        )
        cv2.putText(
            row,
            label,
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        for index, frame in enumerate(frames):
            row[label_height:, index * thumb_width : (index + 1) * thumb_width] = frame
        rows.append(row)
    sheet = np.vstack(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise ValueError(f"cannot write contact sheet: {output_path}")


def write_holdout_artifacts(
    sources: dict[str, Any],
    holdout_dir: Path,
) -> dict[str, Any]:
    sealed = [
        clip
        for clip in sources["clips"]
        if clip["split_role"] == "sealed_holdout"
    ]
    segment_rows = []
    template_rows = []
    for clip in sealed:
        segment_rows.append(
            {
                "segment_id": clip["clip_id"],
                "video_id": clip["video_id"],
                "action": clip["action"],
                "window_start_frame": clip["trim"]["start_frame"],
                "window_end_frame": clip["trim"]["end_frame"],
                "fps": clip["fps"],
                "selection_role": (
                    "target_action"
                    if clip["action"] in {"pushup", "squat"}
                    else "non_target_cycle_control"
                ),
                "source_split_role": clip["source_split_role"],
                "review_status": "pending",
                "notes": "Annotation window only; not GT boundaries.",
            }
        )
        template_rows.append(
            {
                "segment_id": clip["clip_id"],
                "video_id": clip["video_id"],
                "action": clip["action"],
                "rep_id": "",
                "window_start_frame": clip["trim"]["start_frame"],
                "window_end_frame": clip["trim"]["end_frame"],
                "draft_start_frame": "",
                "draft_end_frame": "",
                "gt_start_frame": "",
                "gt_end_frame": "",
                "frame_unit": "source_video_absolute",
                "phase_landmark": "",
                "rep_unit": "cycle",
                "review_status": "pending",
                "boundary_confidence": "",
                "annotator": "",
                "reviewed_at": "",
                "notes": (
                    "Duplicate this row for each complete rep; do not import "
                    "T1-3 labels from the same or related videos."
                ),
            }
        )
    segments_path = holdout_dir / "qevd_sealed_holdout_segments.csv"
    template_path = holdout_dir / "qevd_sealed_holdout_template.csv"
    contact_sheet_path = holdout_dir / "qevd_sealed_holdout_contact_sheet.jpg"
    write_csv(
        segments_path,
        segment_rows,
        [
            "segment_id",
            "video_id",
            "action",
            "window_start_frame",
            "window_end_frame",
            "fps",
            "selection_role",
            "source_split_role",
            "review_status",
            "notes",
        ],
    )
    write_csv(template_path, template_rows, ANNOTATION_TEMPLATE_FIELDS)
    write_contact_sheet(sealed, contact_sheet_path)
    manifest = {
        "schema_version": "task1-6-qevd-sealed-holdout-v1",
        "generated_at": sources["generated_at"],
        "sources_sha256": payload_sha256(sources),
        "status": "pending_human_annotation",
        "segments": len(sealed),
        "actions": dict(sorted(Counter(clip["action"] for clip in sealed).items())),
        "video_ids": sorted({clip["video_id"] for clip in sealed}),
        "files": {
            "segments_csv": {
                "path": str(segments_path),
                "sha256": sha256_path(segments_path),
            },
            "annotation_template_csv": {
                "path": str(template_path),
                "sha256": sha256_path(template_path),
            },
            "contact_sheet_jpg": {
                "path": str(contact_sheet_path),
                "sha256": sha256_path(contact_sheet_path),
            },
        },
        "annotation_complete": False,
        "cross_domain_acceptance_complete": False,
    }
    manifest_path = holdout_dir / "qevd_sealed_holdout_manifest.json"
    write_json(manifest_path, manifest)
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_path(manifest_path),
        "segments_path": str(segments_path),
        "template_path": str(template_path),
        "contact_sheet_path": str(contact_sheet_path),
        "status": manifest["status"],
        "segments": manifest["segments"],
        "actions": manifest["actions"],
    }


def build_sources(
    fit3d_manifest_path: Path,
    qevd_manifest_path: Path,
    first_round_annotations_path: Path,
    blind_annotations_path: Path,
    feedback_path: Path,
    baseline_summary_path: Path,
    check_files: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fit3d_manifest = load_json(fit3d_manifest_path)
    qevd_manifest = load_json(qevd_manifest_path)
    if fit3d_manifest.get("schema_version") != "task1-4-v1":
        raise ValueError("unexpected Fit3D source manifest schema")
    if qevd_manifest.get("schema_version") != "task1-4-v1":
        raise ValueError("unexpected QEVD source manifest schema")

    fit3d_sha = sha256_path(fit3d_manifest_path)
    qevd_sha = sha256_path(qevd_manifest_path)
    feedback_sha = sha256_path(feedback_path)
    annotation_sets = [
        read_annotation_set(first_round_annotations_path, "t1_3_first_round"),
        read_annotation_set(blind_annotations_path, "t1_3_blind"),
    ]

    fit3d_clips = [
        fit3d_clip_record(clip, fit3d_sha) for clip in fit3d_manifest["clips"]
    ]
    qevd_clips = [
        qevd_clip_record(clip, qevd_sha, annotation_sets)
        for clip in qevd_manifest["clips"]
    ]
    selected_segments = load_feedback_segments(
        feedback_path,
        sorted({video_id for video_id, _ in SEALED_NON_TARGET_REFS}),
    )
    by_key = {
        (segment["video_id"], segment["action"]): segment
        for segment in selected_segments
    }
    for key in SEALED_NON_TARGET_REFS:
        if key not in by_key:
            raise ValueError(f"sealed non-target feedback segment not found: {key}")
        qevd_clips.append(
            sealed_non_target_record(by_key[key], feedback_sha, check_files)
        )

    generated_at = utc_now()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task": "T1-6",
        "stage": "T1-6A",
        "generated_at": generated_at,
        "adapter_contract": {
            "scope": "paths_frame_numbers_gt_semantics_splits_clusters_only",
            "changes_algorithm_output": False,
            "invokes_segmenter": False,
        },
        "source_manifests": {
            "fit3d": {
                "path": str(fit3d_manifest_path),
                "sha256": fit3d_sha,
                "clip_count": len(fit3d_manifest["clips"]),
            },
            "qevd_fit_coach": {
                "path": str(qevd_manifest_path),
                "sha256": qevd_sha,
                "clip_count": len(qevd_manifest["clips"]),
            },
            "qevd_feedback": {
                "path": str(feedback_path),
                "sha256": feedback_sha,
            },
        },
        "split_policy": {
            "fit3d": {
                "unit": "subject",
                "dev_subjects": fit3d_manifest.get("split_policy", {}).get(
                    "dev", []
                ),
                "holdout_subjects": fit3d_manifest.get("split_policy", {}).get(
                    "holdout", []
                ),
                "smoke_subjects": fit3d_manifest.get("split_policy", {}).get(
                    "smoke", []
                ),
            },
            "qevd_fit_coach": {
                "unit": "video_id",
                "sealed_rule": "video_ids_disjoint_from_t1_4_dev_and_holdout",
            },
        },
        "legacy_annotation_sets": annotation_sets,
        "clips": fit3d_clips + qevd_clips,
    }
    validation = validate_sources_payload(payload, check_files=check_files)
    baseline = validate_t1_5d_baseline(baseline_summary_path, fit3d_sha)
    payload["baseline_reproduction"] = baseline
    validation["baseline_reproduction"] = {
        "passed": baseline["passed"],
        "errors": baseline["errors"],
    }
    validation["passed"] = validation["passed"] and baseline["passed"]
    payload["validation"] = validation
    if not validation["passed"]:
        raise ValueError(
            "T1-6A source validation failed: "
            + json.dumps(validation["errors"], ensure_ascii=False)
        )
    protocol = build_protocol_payload(payload, baseline, generated_at)
    return payload, protocol


def main() -> int:
    args = parse_args()
    sources, protocol = build_sources(
        fit3d_manifest_path=args.fit3d_manifest,
        qevd_manifest_path=args.qevd_manifest,
        first_round_annotations_path=args.first_round_annotations,
        blind_annotations_path=args.blind_annotations,
        feedback_path=args.feedback,
        baseline_summary_path=args.baseline_summary,
        check_files=not args.skip_file_checks,
    )
    holdout_result = None
    if not args.skip_holdout_artifacts:
        holdout_result = write_holdout_artifacts(sources, args.holdout_dir)
        protocol["sealed_holdout"]["artifacts"] = holdout_result
    write_json(args.sources_output, sources)
    write_json(args.protocol_output, protocol)
    print(
        json.dumps(
            {
                "sources": str(args.sources_output),
                "protocol": str(args.protocol_output),
                "total_clips": len(sources["clips"]),
                "fit3d": sources["validation"]["counts"]["fit3d"],
                "qeud": sources["validation"]["counts"]["qeud"],
                "baseline_passed": sources["baseline_reproduction"]["passed"],
                "sealed_holdout": holdout_result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
