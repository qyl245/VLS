#!/usr/bin/env python3
"""Segment repeated exercise cycles from task1-2 angle tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks, savgol_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import DATA_ROOT


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANGLES_DIR = ROOT / "outputs" / "angles"
DEFAULT_LANDMARKS_DIR = ROOT / "outputs" / "pose_landmarker_full"
DEFAULT_FEEDBACK_JSON = DATA_ROOT / "feedbacks_long_range.json"
DEFAULT_SELECTION = ROOT / "configs" / "task1_1_selection.json"
DEFAULT_PHASE_CONFIG = ROOT / "configs" / "phase_guidance.json"
DEFAULT_REP_METADATA = ROOT / "configs" / "rep_metadata.json"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "segmentation"

VALID_REP_UNITS = {"side", "stride", "cycle"}
VALID_PHASE_POLARITIES = {"peak", "trough", "alternating"}

SMOOTH_WINDOW = 7
SMOOTH_POLYORDER = 2
MIN_ACTIVE_RANGE_DEG = 5.0
ACTIVITY_WINDOW_S = 0.5
ACTIVITY_THRESHOLD = 0.10
MIN_NONREP_S = 0.5
ACTIVITY_CLOSE_GAP_S = 0.2
INCOMPLETE_TAIL_RATIO = 0.75
PHASE_TOLERANCE_FRACTION = 0.20
PHASE_PROMINENCE_FRACTION = 0.10
PHASE_MIN_EXTREMA = 3
PHASE_MIN_SUPPORT_FRACTION = 0.65
PHASE_AMBIGUITY_MARGIN_FRACTION = 0.10
PHASE_MIN_EXTREMA_COVERAGE = 0.75
BOUNDARY_PLATEAU_FRACTION = 0.80
BOUNDARY_MIN_PLATEAU_FRACTION = 0.08
BOUNDARY_SNAP_RADIUS_FRACTION = 0.75
BOUNDARY_MIN_WIDTH_FRACTION = 0.25
SHAPE_TEMPLATE_METHODS = ("mean", "medoid", "trimmed", "irls")
DEFAULT_SHAPE_TEMPLATE = "irls"
REVIEW_HIGH_SCORE = 0.35
REVIEW_MEDIUM_SCORE = 0.50
REVIEW_HIGH_COMPONENT = 0.15
REVIEW_MEDIUM_COMPONENT = 0.30
REVIEW_COMPONENT_FLOOR = 1e-6
MOTION_LOG_SCALE_FLOOR = 0.15

LANDMARK_INDICES = {
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

LOWER_BODY_ANGLES = (
    "left_knee_flexion",
    "right_knee_flexion",
    "left_hip_flexion",
    "right_hip_flexion",
    "left_hip_flexion_sagittal",
    "right_hip_flexion_sagittal",
    "torso_flexion",
)
UPPER_BODY_ANGLES = (
    "left_elbow_flexion",
    "right_elbow_flexion",
    "left_shoulder_flexion",
    "right_shoulder_flexion",
    "left_shoulder_abduction_coronal",
    "right_shoulder_abduction_coronal",
    "left_shoulder_horizontal_abduction_transverse",
    "right_shoulder_horizontal_abduction_transverse",
)
MIXED_ANGLES = tuple(dict.fromkeys((*LOWER_BODY_ANGLES, *UPPER_BODY_ANGLES)))

ACTION_CANDIDATE_ANGLES = {
    "jumping_jack": (
        "left_shoulder_abduction_coronal",
        "right_shoulder_abduction_coronal",
        "left_knee_flexion",
        "right_knee_flexion",
    ),
    "high_knees": LOWER_BODY_ANGLES,
    "squat": LOWER_BODY_ANGLES,
    "pushup": (
        "left_elbow_flexion",
        "right_elbow_flexion",
        "left_shoulder_flexion",
        "right_shoulder_flexion",
        "torso_flexion",
    ),
    "butt_kickers": LOWER_BODY_ANGLES,
    "lunge_jumps": LOWER_BODY_ANGLES,
    "walking_lunges": LOWER_BODY_ANGLES,
    "mountain_climbers": LOWER_BODY_ANGLES,
    "good_morning": LOWER_BODY_ANGLES,
    "air_jump_rope": (
        "left_elbow_flexion",
        "right_elbow_flexion",
        "left_shoulder_flexion",
        "right_shoulder_flexion",
        "left_shoulder_abduction_coronal",
        "right_shoulder_abduction_coronal",
        "left_knee_flexion",
        "right_knee_flexion",
    ),
    "standing_kicks": LOWER_BODY_ANGLES,
    "boxing_squat_punches": (
        "left_elbow_flexion",
        "right_elbow_flexion",
        "left_shoulder_flexion",
        "right_shoulder_flexion",
        "left_knee_flexion",
        "right_knee_flexion",
        "left_hip_flexion_sagittal",
        "right_hip_flexion_sagittal",
    ),
    "toe_touchers": (
        "torso_flexion",
        "left_hip_flexion",
        "right_hip_flexion",
        "left_hip_flexion_sagittal",
        "right_hip_flexion_sagittal",
    ),
    "shoulder_gators": UPPER_BODY_ANGLES,
}

FIXED_ANGLES = {
    "jumping_jack": "left_shoulder_abduction_coronal",
    "high_knees": "left_hip_flexion_sagittal",
    "squat": "left_knee_flexion",
    "pushup": "left_elbow_flexion",
    "butt_kickers": "left_knee_flexion",
    "lunge_jumps": "left_knee_flexion",
    "walking_lunges": "left_knee_flexion",
    "mountain_climbers": "left_hip_flexion_sagittal",
    "good_morning": "left_hip_flexion_sagittal",
    "air_jump_rope": "left_elbow_flexion",
    "standing_kicks": "left_hip_flexion_sagittal",
    "boxing_squat_punches": "left_elbow_flexion",
    "toe_touchers": "torso_flexion",
    "shoulder_gators": "left_shoulder_abduction_coronal",
}

ACTION_PERIOD_RANGES_S = {
    "jumping_jack": (0.45, 2.00),
    "high_knees": (0.35, 1.50),
    "squat": (0.70, 5.00),
    "pushup": (0.70, 5.00),
    "butt_kickers": (0.35, 1.60),
    "lunge_jumps": (0.60, 3.00),
    "walking_lunges": (1.20, 5.00),
    "mountain_climbers": (0.30, 1.50),
    "good_morning": (0.80, 5.00),
    "air_jump_rope": (0.35, 1.50),
    "standing_kicks": (0.50, 3.00),
    "boxing_squat_punches": (0.45, 2.50),
    "toe_touchers": (0.80, 5.00),
    "shoulder_gators": (0.80, 5.00),
}

ANGLE_SCORING_METHODS = ("legacy", "evidence")
TAIL_FILTER_MODES = ("off", "guarded")
ANGLE_EVIDENCE_WEIGHTS = {
    "periodicity": 0.30,
    "peak_prominence": 0.20,
    "phase_consistency": 0.25,
    "boundary_stability": 0.25,
}
ANGLE_CLUSTER_RELATIVE_PERIOD_TOLERANCE = 0.15
ANGLE_AMBIGUITY_SCORE_MARGIN = 0.03
AIR_JUMP_TAIL_END_TOLERANCE_FRACTION = 0.35
AIR_JUMP_TAIL_SHAPE_RATIO = 0.65
AIR_JUMP_TAIL_SHAPE_FLOOR = 0.25
AIR_JUMP_TAIL_MOTION_RATIO = 0.15
AIR_JUMP_TAIL_MOTION_FLOOR = 0.05

ACTION_ALIASES = (
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
    ("armcrosschest", "armcrosschest"),
    ("quad stretch", "quad_stretch"),
    ("squat", "squat"),
    ("pushup", "pushup"),
    ("push up", "pushup"),
    ("plank", "plank"),
)

ACTION_PATTERNS = (
    re.compile(r"first up (?:are|is) (.+?)[!.]", re.IGNORECASE),
    re.compile(r"moving on to (.+?)[!.]", re.IGNORECASE),
)


@dataclass(frozen=True)
class ActionSegment:
    video_id: str
    action: str
    action_display: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float


@dataclass(frozen=True)
class Rep:
    rep_id: int
    start_frame: int
    end_frame: int
    peak_frame: int | None
    confidence: float
    visibility: float = 0.0
    shape_similarity: float = 0.0
    phase_consistency: float = 0.5
    motion_evidence: float = 0.5
    review_score: float = 0.5
    review_priority: str = "unscored"
    reject_reason: str = ""


@dataclass(frozen=True)
class RejectedRep:
    start_frame: int
    end_frame: int
    reason: str
    evidence: dict[str, Any]


@dataclass
class MethodResult:
    method: str
    angle: str
    reps: list[Rep]
    period_frames: float
    similarity: float
    valid_fraction: float
    objective: float
    rejected_reps: list[RejectedRep] = field(default_factory=list)
    phase_guidance: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedSignal:
    smooth: np.ndarray
    detrended: np.ndarray
    normalized: np.ndarray
    valid: np.ndarray
    valid_fraction: float
    dynamic_range: float
    source_start: int
    source_end: int


@dataclass
class AngleCandidate:
    angle: str
    prepared: PreparedSignal
    period_frames: int | None
    periodicity: float
    peak_prominence: float
    phase_consistency: float
    boundary_stability: float
    score: float
    cluster_id: int = -1


@dataclass
class AngleSelection:
    angle: str
    prepared: PreparedSignal
    periodicity: float
    evidence: dict[str, Any]


@dataclass(frozen=True)
class PhaseSignalSpec:
    source: str
    name: str
    axis: str | None = None
    sign: float = 1.0


@dataclass(frozen=True)
class BoundarySnapGuidance:
    mode: str = "disabled"
    plateau_fraction: float = BOUNDARY_PLATEAU_FRACTION
    min_plateau_fraction: float = BOUNDARY_MIN_PLATEAU_FRACTION
    snap_radius_fraction: float = BOUNDARY_SNAP_RADIUS_FRACTION
    min_width_fraction: float = BOUNDARY_MIN_WIDTH_FRACTION


@dataclass(frozen=True)
class PhaseGuidance:
    phase_landmark: str
    annotation_polarity: str
    signal: PhaseSignalSpec
    polarity: str
    tolerance_fraction: float = PHASE_TOLERANCE_FRACTION
    prominence_fraction: float = PHASE_PROMINENCE_FRACTION
    min_extrema: int = PHASE_MIN_EXTREMA
    min_support_fraction: float = PHASE_MIN_SUPPORT_FRACTION
    ambiguity_margin_fraction: float = PHASE_AMBIGUITY_MARGIN_FRACTION
    min_extrema_coverage: float = PHASE_MIN_EXTREMA_COVERAGE
    boundary_snap: BoundarySnapGuidance = field(default_factory=BoundarySnapGuidance)


@dataclass(frozen=True)
class ActiveBout:
    bout_id: int
    start_frame: int
    end_frame: int
    fit_start_frame: int
    fit_end_frame: int
    status: str
    evidence: dict[str, Any]


@dataclass(frozen=True)
class NonRepInterval:
    start_frame: int
    end_frame: int
    reason: str
    evidence: dict[str, Any]


@dataclass
class ActivityTrace:
    activity: np.ndarray
    visibility: np.ndarray
    main_speed: np.ndarray
    low_intervals: list[tuple[int, int]]
    evidence: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angles-dir", type=Path, default=DEFAULT_ANGLES_DIR)
    parser.add_argument("--landmarks-dir", type=Path, default=DEFAULT_LANDMARKS_DIR)
    parser.add_argument("--feedback-json", type=Path, default=DEFAULT_FEEDBACK_JSON)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--phase-config", type=Path, default=DEFAULT_PHASE_CONFIG)
    parser.add_argument("--no-phase-guidance", action="store_true")
    parser.add_argument("--no-phase-snap", action="store_true")
    parser.add_argument("--rep-metadata", type=Path, default=DEFAULT_REP_METADATA)
    parser.add_argument("--no-rep-metadata", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--actions", nargs="*", default=None)
    parser.add_argument("--min-period-s", type=float, default=0.40)
    parser.add_argument("--max-period-s", type=float, default=5.00)
    parser.add_argument("--prominence-frac", type=float, default=0.20)
    parser.add_argument("--no-stage-b", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-active-bouts", action="store_true")
    parser.add_argument("--activity-window-s", type=float, default=ACTIVITY_WINDOW_S)
    parser.add_argument("--activity-threshold", type=float, default=ACTIVITY_THRESHOLD)
    parser.add_argument("--min-nonrep-s", type=float, default=MIN_NONREP_S)
    parser.add_argument("--activity-close-gap-s", type=float, default=ACTIVITY_CLOSE_GAP_S)
    parser.add_argument(
        "--confidence-template",
        choices=SHAPE_TEMPLATE_METHODS,
        default=DEFAULT_SHAPE_TEMPLATE,
        help="Robust template used for shape_similarity and review_score.",
    )
    parser.add_argument(
        "--angle-scoring",
        choices=ANGLE_SCORING_METHODS,
        default="legacy",
        help="Angle selection mode; evidence enables P1-A late fusion.",
    )
    parser.add_argument(
        "--p2a-tail-filter",
        choices=TAIL_FILTER_MODES,
        default="guarded",
        help="Guarded air-jump-rope transition-tail filter.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=SMOOTH_WINDOW,
        help="Savitzky-Golay window in frames; 0 disables smoothing.",
    )
    parser.add_argument(
        "--smooth-polyorder",
        type=int,
        default=SMOOTH_POLYORDER,
        help="Savitzky-Golay polynomial order.",
    )
    args = parser.parse_args()
    if args.min_period_s <= 0 or args.max_period_s <= args.min_period_s:
        parser.error("--max-period-s must be greater than --min-period-s")
    if not 0 < args.prominence_frac < 1:
        parser.error("--prominence-frac must be in (0, 1)")
    if args.activity_window_s <= 0:
        parser.error("--activity-window-s must be positive")
    if not 0 < args.activity_threshold < 1:
        parser.error("--activity-threshold must be in (0, 1)")
    if args.min_nonrep_s <= 0:
        parser.error("--min-nonrep-s must be positive")
    if args.activity_close_gap_s < 0:
        parser.error("--activity-close-gap-s must be non-negative")
    if args.smooth_window != 0 and (
        args.smooth_window < 5 or args.smooth_window % 2 == 0
    ):
        parser.error("--smooth-window must be 0 or an odd integer >= 5")
    if args.smooth_window != 0 and not 1 <= args.smooth_polyorder < args.smooth_window:
        parser.error("--smooth-polyorder must be in [1, smooth-window)")
    if not args.no_phase_guidance and not args.phase_config.exists():
        parser.error(f"phase config does not exist: {args.phase_config}")
    if not args.no_rep_metadata and not args.rep_metadata.exists():
        parser.error(f"rep metadata config does not exist: {args.rep_metadata}")
    return args


def load_video_ids(selection_path: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return [Path(name).stem for name in requested]
    payload = json.loads(selection_path.read_text())
    return [entry["video_id"] for entry in payload["videos"]]


def load_wide(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        values = {name: [] for name in columns}
        for row in reader:
            for name in columns:
                values[name].append(float(row[name]))
    timestamps = np.asarray(values.pop("timestamp_ms"), dtype=np.float64)
    return timestamps, {name: np.asarray(value, dtype=np.float64) for name, value in values.items()}


def load_phase_config(path: Path) -> dict[str, PhaseGuidance]:
    payload = json.loads(path.read_text())
    defaults = payload.get("defaults", {})
    guidance_by_action: dict[str, PhaseGuidance] = {}
    for action, entry in payload.get("actions", {}).items():
        signal_payload = entry.get("signal", {})
        source = str(signal_payload.get("source", "")).strip()
        name = str(signal_payload.get("name", "")).strip()
        polarity = str(entry.get("polarity", "")).strip().lower()
        annotation_polarity = str(entry.get("annotation_polarity", polarity)).strip().lower()
        tolerance_fraction = float(
            entry.get("tolerance_fraction", defaults.get("tolerance_fraction", PHASE_TOLERANCE_FRACTION))
        )
        prominence_fraction = float(
            entry.get(
                "prominence_fraction",
                defaults.get("prominence_fraction", PHASE_PROMINENCE_FRACTION),
            )
        )
        min_support_fraction = float(
            entry.get(
                "min_support_fraction",
                defaults.get("min_support_fraction", PHASE_MIN_SUPPORT_FRACTION),
            )
        )
        ambiguity_margin_fraction = float(
            entry.get(
                "ambiguity_margin_fraction",
                defaults.get("ambiguity_margin_fraction", PHASE_AMBIGUITY_MARGIN_FRACTION),
            )
        )
        min_extrema_coverage = float(
            entry.get(
                "min_extrema_coverage",
                defaults.get("min_extrema_coverage", PHASE_MIN_EXTREMA_COVERAGE),
            )
        )
        if source not in {"angle", "landmark_world"}:
            raise ValueError(f"unsupported phase signal source for {action}: {source}")
        if not name:
            raise ValueError(f"phase signal name is empty for {action}")
        if polarity not in {"peak", "trough"}:
            raise ValueError(f"unsupported phase polarity for {action}: {polarity}")
        if annotation_polarity not in {"peak", "trough", "alternating"}:
            raise ValueError(
                f"unsupported annotation polarity for {action}: {annotation_polarity}"
            )
        if not 0 < tolerance_fraction <= 0.45:
            raise ValueError(f"phase tolerance must be in (0, 0.45] for {action}")
        if not 0 <= prominence_fraction < 1:
            raise ValueError(f"phase prominence must be in [0, 1) for {action}")
        if not 0 < min_support_fraction <= 1:
            raise ValueError(f"phase support fraction must be in (0, 1] for {action}")
        if not 0 <= ambiguity_margin_fraction <= 1:
            raise ValueError(f"phase ambiguity margin must be in [0, 1] for {action}")
        if not 0 < min_extrema_coverage <= 1:
            raise ValueError(f"phase extrema coverage must be in (0, 1] for {action}")
        min_extrema = int(
            entry.get("min_extrema", defaults.get("min_extrema", PHASE_MIN_EXTREMA))
        )
        if min_extrema < 1:
            raise ValueError(f"phase min_extrema must be positive for {action}")
        snap_payload = entry.get("boundary_snap", {})
        boundary_mode = str(snap_payload.get("mode", "disabled")).strip().lower()
        plateau_fraction = float(
            snap_payload.get(
                "plateau_fraction",
                defaults.get("plateau_fraction", BOUNDARY_PLATEAU_FRACTION),
            )
        )
        min_plateau_fraction = float(
            snap_payload.get(
                "min_plateau_fraction",
                defaults.get("min_plateau_fraction", BOUNDARY_MIN_PLATEAU_FRACTION),
            )
        )
        snap_radius_fraction = float(
            snap_payload.get(
                "snap_radius_fraction",
                defaults.get("snap_radius_fraction", BOUNDARY_SNAP_RADIUS_FRACTION),
            )
        )
        min_width_fraction = float(
            snap_payload.get(
                "min_width_fraction",
                defaults.get("min_width_fraction", BOUNDARY_MIN_WIDTH_FRACTION),
            )
        )
        if boundary_mode not in {"disabled", "plateau_entry"}:
            raise ValueError(f"unsupported boundary snap mode for {action}: {boundary_mode}")
        if not 0 < plateau_fraction < 1:
            raise ValueError(f"boundary plateau_fraction must be in (0, 1) for {action}")
        if not 0 < min_plateau_fraction <= 1:
            raise ValueError(
                f"boundary min_plateau_fraction must be in (0, 1] for {action}"
            )
        if not 0 < snap_radius_fraction <= 1:
            raise ValueError(f"boundary snap_radius_fraction must be in (0, 1] for {action}")
        if not 0 < min_width_fraction <= 1:
            raise ValueError(f"boundary min_width_fraction must be in (0, 1] for {action}")
        axis = signal_payload.get("axis")
        if source == "landmark_world":
            if name not in LANDMARK_INDICES:
                raise ValueError(f"unknown landmark for {action}: {name}")
            if axis not in {"x", "y", "z"}:
                raise ValueError(f"landmark phase signal needs an x/y/z axis for {action}")
        guidance_by_action[action] = PhaseGuidance(
            phase_landmark=str(entry.get("phase_landmark", "")).strip(),
            annotation_polarity=annotation_polarity,
            signal=PhaseSignalSpec(
                source=source,
                name=name,
                axis=str(axis) if axis is not None else None,
                sign=float(signal_payload.get("sign", 1.0)),
            ),
            polarity=polarity,
            tolerance_fraction=tolerance_fraction,
            prominence_fraction=prominence_fraction,
            min_extrema=min_extrema,
            min_support_fraction=min_support_fraction,
            ambiguity_margin_fraction=ambiguity_margin_fraction,
            min_extrema_coverage=min_extrema_coverage,
            boundary_snap=BoundarySnapGuidance(
                mode=boundary_mode,
                plateau_fraction=plateau_fraction,
                min_plateau_fraction=min_plateau_fraction,
                snap_radius_fraction=snap_radius_fraction,
                min_width_fraction=min_width_fraction,
            ),
        )
    return guidance_by_action


def load_rep_metadata(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(path.read_text())
    entries = payload.get("segments")
    if not isinstance(entries, dict):
        raise ValueError(f"rep metadata needs a segments object: {path}")
    metadata_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for key, raw in entries.items():
        if not isinstance(raw, dict) or "/" not in key:
            raise ValueError(f"invalid rep metadata entry: {key}")
        video_id, action = key.split("/", 1)
        video_id = video_id.strip().zfill(4)
        action = action.strip()
        if not video_id or not action:
            raise ValueError(f"invalid rep metadata key: {key}")
        rep_unit = str(raw.get("rep_unit", "")).strip().lower()
        phase_landmark = str(raw.get("phase_landmark", "")).strip()
        phase_polarity = str(raw.get("phase_polarity", "")).strip().lower()
        if rep_unit not in VALID_REP_UNITS:
            raise ValueError(f"invalid rep_unit for {key}: {rep_unit}")
        if not phase_landmark:
            raise ValueError(f"missing phase_landmark for {key}")
        if phase_polarity not in VALID_PHASE_POLARITIES:
            raise ValueError(f"invalid phase_polarity for {key}: {phase_polarity}")
        metadata_by_key[(video_id, action)] = {
            **raw,
            "rep_unit": rep_unit,
            "phase_landmark": phase_landmark,
            "phase_polarity": phase_polarity,
        }
    return metadata_by_key


def rep_metadata_fields(metadata: dict[str, Any] | None) -> dict[str, str]:
    if metadata is None:
        return {"rep_unit": "", "phase_landmark": "", "phase_polarity": ""}
    return {
        "rep_unit": str(metadata.get("rep_unit", "")),
        "phase_landmark": str(metadata.get("phase_landmark", "")),
        "phase_polarity": str(metadata.get("phase_polarity", "")),
    }


def load_landmark_phase_signal(
    path: Path,
    index: int,
    axis: str,
    sign: float,
) -> np.ndarray:
    records = json.loads(path.read_text())
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    values = np.full(len(records), np.nan, dtype=np.float64)
    for frame_index, record in enumerate(records):
        landmarks = record.get("landmarks_world")
        if not record.get("detected") or not landmarks or index >= len(landmarks):
            continue
        point = landmarks[index]
        if not isinstance(point, list) or axis_index >= len(point):
            continue
        values[frame_index] = sign * float(point[axis_index])
    return values


def resolve_phase_signal(
    guidance: PhaseGuidance,
    wide: dict[str, np.ndarray],
    landmarks_dir: Path,
    video_id: str,
) -> np.ndarray | None:
    if guidance.signal.source == "angle":
        return wide.get(guidance.signal.name)
    landmarks_path = landmarks_dir / f"{video_id}_landmarks.json"
    if not landmarks_path.exists():
        return None
    return load_landmark_phase_signal(
        landmarks_path,
        LANDMARK_INDICES[guidance.signal.name],
        guidance.signal.axis or "y",
        guidance.signal.sign,
    )


def normalize_action_text(text: str) -> tuple[str, str]:
    display = re.sub(r"\s+", " ", text.strip().strip(" .!")).lower()
    compact = display.replace("-", " ").replace("_", " ")
    for pattern, action in ACTION_ALIASES:
        if pattern in compact:
            return action, display
    return re.sub(r"[^a-z0-9]+", "_", compact).strip("_"), display


def find_action_text(texts: list[str], start: int, end: int) -> str | None:
    for index in range(start, end + 1):
        text = texts[index]
        if not text:
            continue
        for pattern in ACTION_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
    return None


def load_feedback_segments(path: Path, video_ids: list[str]) -> list[ActionSegment]:
    payload = json.loads(path.read_text())
    by_video = {
        Path(item["long_range_video_file"]).stem: item
        for item in payload
    }
    segments: list[ActionSegment] = []
    for video_id in video_ids:
        if video_id not in by_video:
            raise ValueError(f"feedback JSON has no record for {video_id}")
        record = by_video[video_id]
        timestamps_path = path.parent / record["video_timestamps"]
        timestamps = np.load(timestamps_path).astype(np.float64) / 1e9
        feedbacks = record.get("feedbacks", [])
        transition_frames = []
        for timestamp, is_transition in zip(
            record.get("feedback_timestamps", []), record.get("is_transition", [])
        ):
            if not is_transition:
                continue
            shifted = float(timestamp) - 28800.0
            transition_frames.append(int(np.argmin(np.abs(timestamps - shifted))))
        transition_frames = sorted(set(transition_frames))
        for segment_index, start in enumerate(transition_frames):
            end = (
                transition_frames[segment_index + 1] - 1
                if segment_index + 1 < len(transition_frames)
                else len(feedbacks) - 1
            )
            if end < start:
                continue
            action_text = find_action_text(feedbacks, start, end)
            if action_text is None:
                continue
            action, display = normalize_action_text(action_text)
            segments.append(
                ActionSegment(
                    video_id=video_id,
                    action=action,
                    action_display=display,
                    start_frame=start,
                    end_frame=end,
                    start_time_s=float(timestamps[start]),
                    end_time_s=float(timestamps[end]),
                )
            )
    return segments


def prepare_core(values: np.ndarray) -> PreparedSignal | None:
    count = len(values)
    valid = np.isfinite(values)
    valid_fraction = float(valid.mean())
    if valid.sum() < 20 or valid_fraction < 0.30:
        return None
    indexes = np.arange(count)
    filled = np.interp(indexes, indexes[valid], values[valid])
    window = min(SMOOTH_WINDOW, count if count % 2 == 1 else count - 1)
    if window >= 5:
        smooth = savgol_filter(filled, window, SMOOTH_POLYORDER, mode="interp")
    else:
        smooth = filled
    coefficients = np.polyfit(indexes, smooth, 1)
    detrended = smooth - np.polyval(coefficients, indexes)
    low, high = np.percentile(detrended, [5, 95])
    dynamic_range = float(high - low)
    scale = float(np.std(detrended))
    if dynamic_range < MIN_ACTIVE_RANGE_DEG or scale < 1e-8:
        return None
    normalized = detrended / scale
    return PreparedSignal(
        smooth=smooth,
        detrended=detrended,
        normalized=normalized,
        valid=valid,
        valid_fraction=valid_fraction,
        dynamic_range=dynamic_range,
        source_start=0,
        source_end=count - 1,
    )


def active_region(signal: np.ndarray, dynamic_range: float) -> tuple[int, int]:
    count = len(signal)
    oriented = signal if peak_prominence_score(signal) >= peak_prominence_score(-signal) else -signal
    distance = 10
    prominence = adaptive_prominence(oriented, dynamic_range, distance, 0.20)
    peaks, _ = find_peaks(oriented, distance=distance, prominence=prominence)
    if len(peaks) >= 2:
        spacing = float(np.median(np.diff(peaks)))
        start = max(0, int(peaks[0] - spacing))
        end = min(count - 1, int(peaks[-1] + spacing))
        if end - start + 1 >= max(20, int(0.15 * count)):
            return start, end
    window = min(90, count if count % 2 == 1 else count - 1)
    if window < 15:
        return 0, count - 1
    padding = window // 2
    padded = np.pad(signal, (padding, padding), mode="edge")
    windows = sliding_window_view(padded, window)
    low, high = np.percentile(windows, [10, 90], axis=1)
    rolling_range = high - low
    threshold = max(MIN_ACTIVE_RANGE_DEG, 0.45 * dynamic_range)
    active = np.flatnonzero(rolling_range >= threshold)
    if len(active) == 0:
        return 0, count - 1
    start = max(0, int(active[0]) - 6)
    end = min(count - 1, int(active[-1]) + 6)
    if end - start + 1 < max(30, int(0.15 * count)):
        return 0, count - 1
    return start, end


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [(int(start), int(end - 1)) for start, end in zip(edges[::2], edges[1::2])]


def merge_close_runs(
    intervals: list[tuple[int, int]], max_gap: int
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if merged and start - merged[-1][1] - 1 <= max_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def rolling_normalized_range(values: np.ndarray, window: int, scale: float) -> np.ndarray:
    padding = window // 2
    padded = np.pad(values, (padding, padding), mode="edge")
    windows = sliding_window_view(padded, window)
    low, high = np.percentile(windows, [10, 90], axis=1)
    return (high - low) / max(1e-8, scale)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    padding = window // 2
    padded = np.pad(values, (padding, padding), mode="edge")
    return sliding_window_view(padded, window).mean(axis=1)


def build_activity_trace(
    wide: dict[str, np.ndarray],
    segment: ActionSegment,
    chosen_angle: str,
    fps: float,
    window_s: float,
    threshold: float,
    close_gap_s: float,
) -> ActivityTrace | None:
    """Build a multi-angle, cycle-scale activity trace in action-window coordinates."""
    count = segment.end_frame - segment.start_frame + 1
    requested_window = max(3, int(round(window_s * fps)))
    window = min(requested_window, count if count % 2 == 1 else count - 1)
    if window < 3:
        return None

    angle_activity = []
    main_activity = None
    main_speed = None
    main_visibility = None
    used_angles = []
    for angle in ACTION_CANDIDATE_ANGLES.get(segment.action, MIXED_ANGLES):
        if angle not in wide:
            continue
        values = wide[angle][segment.start_frame : segment.end_frame + 1]
        prepared = prepare_core(values)
        if prepared is None:
            continue
        activity = rolling_normalized_range(prepared.smooth, window, prepared.dynamic_range)
        speed = rolling_mean(
            np.abs(np.diff(prepared.smooth, prepend=prepared.smooth[0])),
            window,
        ) / max(1e-8, prepared.dynamic_range)
        visibility = rolling_mean(prepared.valid.astype(np.float64), window)
        angle_activity.append(activity)
        used_angles.append(angle)
        if angle == chosen_angle:
            main_activity = activity
            main_speed = speed
            main_visibility = visibility

    if not angle_activity or main_activity is None:
        return None
    stacked = np.stack(angle_activity, axis=0)
    with np.errstate(all="ignore"):
        activity = np.nanmedian(stacked, axis=0)
    activity = np.nan_to_num(activity, nan=1.0)
    activity = rolling_mean(activity, min(5, len(activity) if len(activity) % 2 == 1 else len(activity) - 1))
    low_mask = activity < threshold
    max_gap = max(0, int(round(close_gap_s * fps)))
    low_intervals = merge_close_runs(contiguous_runs(low_mask), max_gap)
    evidence = {
        "window_frames": window,
        "window_s": window / fps,
        "threshold": threshold,
        "close_gap_frames": max_gap,
        "candidate_angles": used_angles,
        "activity_percentiles": {
            "p10": float(np.percentile(activity, 10)),
            "p25": float(np.percentile(activity, 25)),
            "p50": float(np.percentile(activity, 50)),
            "p75": float(np.percentile(activity, 75)),
            "p90": float(np.percentile(activity, 90)),
        },
    }
    return ActivityTrace(
        activity=activity,
        visibility=main_visibility,
        main_speed=main_speed,
        low_intervals=low_intervals,
        evidence=evidence,
    )


def significant_low_intervals(
    trace: ActivityTrace,
    period_frames: float,
    fps: float,
    min_nonrep_s: float,
) -> list[tuple[int, int]]:
    minimum = max(
        int(round(min_nonrep_s * fps)),
        int(round(0.5 * period_frames)) if np.isfinite(period_frames) else 0,
    )
    return [(start, end) for start, end in trace.low_intervals if end - start + 1 >= minimum]


def prepare_signal(values: np.ndarray) -> PreparedSignal | None:
    initial = prepare_core(values)
    if initial is None:
        return None
    start, end = active_region(initial.detrended, initial.dynamic_range)
    if start == 0 and end == len(values) - 1:
        return initial
    prepared = prepare_core(values[start : end + 1])
    if prepared is None:
        return None
    prepared.source_start = start
    prepared.source_end = end
    return prepared


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 4 or len(first) != len(second):
        return np.nan
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = float(np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    if denominator < 1e-8:
        return np.nan
    return float(np.dot(first_centered, second_centered) / denominator)


def robust_template(curves: np.ndarray, method: str = DEFAULT_SHAPE_TEMPLATE) -> np.ndarray:
    """Build a rep template without letting a single outlier curve dominate."""
    if method not in SHAPE_TEMPLATE_METHODS:
        raise ValueError(f"unsupported shape template method: {method}")
    if curves.ndim != 2 or len(curves) == 0:
        raise ValueError("curves must be a non-empty 2-D array")
    if len(curves) == 1:
        return curves[0].copy()

    if method == "mean":
        return curves.mean(axis=0)
    if method == "medoid":
        scores = []
        for index, curve in enumerate(curves):
            values = [
                correlation(curve, other)
                for other_index, other in enumerate(curves)
                if other_index != index
            ]
            values = [value for value in values if np.isfinite(value)]
            scores.append(float(np.mean(values)) if values else -np.inf)
        return curves[int(np.argmax(scores))].copy()
    if method == "trimmed":
        trim = max(1, int(np.floor(0.20 * len(curves)))) if len(curves) >= 5 else 0
        if trim == 0:
            return curves.mean(axis=0)
        ordered = np.sort(curves, axis=0)
        return ordered[trim : len(curves) - trim].mean(axis=0)

    template = np.median(curves, axis=0)
    for _ in range(20):
        values = np.asarray([correlation(curve, template) for curve in curves])
        distances = np.where(np.isfinite(values), np.clip(1.0 - values, 0.0, 2.0), 2.0)
        center = float(np.median(distances))
        mad = float(np.median(np.abs(distances - center)))
        scale = max(0.05, 1.4826 * mad)
        weights = np.maximum(1e-6, 1.0 / (1.0 + (distances / scale) ** 2))
        candidate = np.average(curves, axis=0, weights=weights)
        if correlation(candidate, template) >= 0.9999:
            template = candidate
            break
        template = candidate
    if float(np.std(template)) < 1e-8:
        return curves.mean(axis=0)
    return template


def rep_curve_matrix(
    signal: np.ndarray,
    offset: int,
    reps: list[Rep],
    samples: int = 48,
) -> np.ndarray | None:
    """Resample every predicted rep onto a common normalized curve grid."""
    curves = []
    indexes = np.arange(len(signal), dtype=np.float64)
    for rep in reps:
        start = rep.start_frame - offset
        end = rep.end_frame + 1 - offset
        if start < 0 or end > len(signal) or end - start < 2:
            return None
        points = np.linspace(float(start), float(end), samples, endpoint=False)
        curve = np.interp(points, indexes, signal)
        if not np.isfinite(curve).all():
            return None
        scale = float(np.std(curve))
        if scale < 1e-8:
            return None
        curves.append((curve - curve.mean()) / scale)
    return np.stack(curves, axis=0) if curves else None


def rep_visibility_scores(reps: list[Rep], valid: np.ndarray, offset: int) -> list[float]:
    scores = []
    for rep in reps:
        start = max(0, rep.start_frame - offset)
        end = min(len(valid), rep.end_frame + 1 - offset)
        scores.append(float(valid[start:end].mean()) if end > start else 0.0)
    return scores


def rep_shape_similarities(
    reps: list[Rep],
    signal: np.ndarray,
    offset: int,
    method: str = DEFAULT_SHAPE_TEMPLATE,
) -> list[float] | None:
    curves = rep_curve_matrix(signal, offset, reps)
    if curves is None:
        return None
    template = robust_template(curves, method)
    scores = []
    for curve in curves:
        value = correlation(curve, template)
        scores.append(min(1.0, max(0.0, value)) if np.isfinite(value) else 0.0)
    return scores


def rep_phase_consistency_scores(
    reps: list[Rep],
    offset: int,
    phase_evidence: dict[str, Any] | None,
    period_frames: float,
    phase_signal_available: bool,
) -> list[float]:
    """Score boundary phase offsets and cycle-width stability without interval IoU."""
    neutral = [0.5] * len(reps)
    if not phase_signal_available or not phase_evidence or period_frames <= 0:
        return neutral
    extrema = phase_evidence.get("extrema_frames") or []
    if len(extrema) < 3:
        return neutral
    extrema_array = np.asarray(extrema, dtype=np.float64)
    tolerance = float(
        phase_evidence.get("tolerance_frames")
        or max(2.0, PHASE_TOLERANCE_FRACTION * period_frames)
    )
    tolerance = max(2.0, tolerance)
    cycle_ratios = np.asarray(
        [(rep.end_frame - rep.start_frame + 1) / period_frames for rep in reps],
        dtype=np.float64,
    )
    target_ratio = float(np.median(cycle_ratios))
    ratio_center = float(np.median(cycle_ratios))
    ratio_mad = float(np.median(np.abs(cycle_ratios - ratio_center)))
    ratio_scale = max(0.10, 1.4826 * ratio_mad)
    scores = []
    for rep, cycle_ratio in zip(reps, cycle_ratios):
        start = rep.start_frame - offset
        end = rep.end_frame + 1 - offset
        start_distance = float(np.min(np.abs(extrema_array - start)))
        end_distance = float(np.min(np.abs(extrema_array - end)))
        boundary_score = float(
            np.exp(-0.5 * ((0.5 * (start_distance + end_distance) / tolerance) ** 2))
        )
        width_score = float(
            np.exp(-0.5 * (((cycle_ratio - target_ratio) / ratio_scale) ** 2))
        )
        scores.append(float(np.clip(np.sqrt(boundary_score * width_score), 0.0, 1.0)))
    return scores


def rep_motion_evidence_scores(
    reps: list[Rep],
    trace: ActivityTrace | None,
    segment_start_frame: int,
) -> list[float] | None:
    """Measure whether a rep's activity resembles the segment's typical reps."""
    if trace is None or len(reps) < 2:
        return None
    activities = []
    for rep in reps:
        start = max(0, rep.start_frame - segment_start_frame)
        end = min(len(trace.activity), rep.end_frame + 1 - segment_start_frame)
        if end <= start:
            return None
        value = float(np.nanmedian(trace.activity[start:end]))
        activities.append(max(1e-3, value))
    logs = np.log(np.asarray(activities, dtype=np.float64))
    center = float(np.median(logs))
    mad = float(np.median(np.abs(logs - center)))
    scale = max(MOTION_LOG_SCALE_FLOOR, 1.4826 * mad)
    deviations = np.abs(logs - center) / scale
    return [float(np.clip(np.exp(-0.5 * value**2), 0.0, 1.0)) for value in deviations]


def review_priority_for(
    review_score: float,
    visibility: float,
    shape_similarity: float,
    phase_consistency: float,
    motion_evidence: float,
) -> str:
    components = (visibility, shape_similarity, phase_consistency, motion_evidence)
    if review_score < REVIEW_HIGH_SCORE or min(components) < REVIEW_HIGH_COMPONENT:
        return "high"
    if review_score < REVIEW_MEDIUM_SCORE or min(components) < REVIEW_MEDIUM_COMPONENT:
        return "medium"
    return "low"


def finalize_rep_confidences(reps: list[Rep]) -> list[Rep]:
    """Recompute the review score while preserving legacy confidence and boundaries."""
    finalized = []
    component_names = (
        "visibility",
        "shape_similarity",
        "phase_consistency",
        "motion_evidence",
    )
    for rep in reps:
        components = (
            rep.visibility,
            rep.shape_similarity,
            rep.phase_consistency,
            rep.motion_evidence,
        )
        clipped = np.clip(np.asarray(components, dtype=np.float64), REVIEW_COMPONENT_FLOOR, 1.0)
        review_score = float(np.exp(np.mean(np.log(clipped))))
        priority = review_priority_for(review_score, *components)
        low_components = [
            name
            for name, value in zip(component_names, components)
            if value < REVIEW_MEDIUM_COMPONENT
        ]
        reject_reason = ";".join(f"low_{name}" for name in low_components)
        if not reject_reason and priority != "low":
            reject_reason = "composite_low_score"
        finalized.append(
            replace(
                rep,
                review_score=review_score,
                review_priority=priority,
                reject_reason=reject_reason,
            )
        )
    return finalized


def enrich_rep_confidences(
    reps: list[Rep],
    *,
    signal: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    offset: int = 0,
    template_method: str = DEFAULT_SHAPE_TEMPLATE,
    phase_evidence: dict[str, Any] | None = None,
    period_frames: float | None = None,
    phase_signal_available: bool = False,
    trace: ActivityTrace | None = None,
    segment_start_frame: int | None = None,
) -> list[Rep]:
    """Attach interpretable evidence components without deleting any rep."""
    if not reps:
        return []
    visibility = (
        rep_visibility_scores(reps, valid, offset)
        if valid is not None
        else None
    )
    shapes = (
        rep_shape_similarities(reps, signal, offset, template_method)
        if signal is not None
        else None
    )
    phases = (
        rep_phase_consistency_scores(
            reps,
            offset,
            phase_evidence,
            float(period_frames or 0.0),
            phase_signal_available,
        )
        if phase_evidence is not None
        else None
    )
    motions = (
        rep_motion_evidence_scores(reps, trace, segment_start_frame)
        if trace is not None and segment_start_frame is not None
        else None
    )
    enriched = []
    for index, rep in enumerate(reps):
        enriched.append(
            replace(
                rep,
                visibility=visibility[index] if visibility is not None else rep.visibility,
                shape_similarity=shapes[index] if shapes is not None else rep.shape_similarity,
                phase_consistency=phases[index] if phases is not None else rep.phase_consistency,
                motion_evidence=motions[index] if motions is not None else rep.motion_evidence,
            )
        )
    return finalize_rep_confidences(enriched)


def renumber_reps(reps: list[Rep]) -> list[Rep]:
    return [replace(rep, rep_id=index) for index, rep in enumerate(reps, start=1)]


def autocorrelation_curve(
    signal: np.ndarray, min_lag: int, max_lag: int
) -> tuple[np.ndarray, np.ndarray]:
    count = len(signal)
    max_lag = min(max_lag, max(min_lag + 1, count // 2))
    lags = np.arange(min_lag, max_lag + 1)
    trims = np.unique(
        np.linspace(0, min(int(round(0.15 * count)), max_lag), 4).astype(int)
    )
    scores = []
    for lag in lags:
        values = []
        for trim in trims:
            first = signal[trim : count - trim - lag]
            second = signal[trim + lag : count - trim]
            if len(first) < max(8, lag // 2):
                continue
            value = correlation(first, second)
            if np.isfinite(value):
                values.append(value)
        scores.append(float(np.mean(values)) if values else np.nan)
    return lags, np.asarray(scores, dtype=np.float64)


def estimate_period(
    signal: np.ndarray, fps: float, min_period_s: float, max_period_s: float
) -> tuple[int, float] | None:
    min_lag = max(4, int(round(min_period_s * fps)))
    max_lag = max(min_lag + 2, int(round(max_period_s * fps)))
    lags, scores = autocorrelation_curve(signal, min_lag, max_lag)
    if len(lags) < 3 or not np.isfinite(scores).any():
        return None
    filled = np.nan_to_num(scores, nan=-1.0)
    maximum = float(np.max(filled))
    if maximum < 0.10:
        return None
    peaks, _ = find_peaks(
        filled,
        distance=max(2, min_lag // 4),
        prominence=0.03,
    )
    threshold = max(0.15, 0.55 * maximum)
    candidates = peaks[filled[peaks] >= threshold]
    lag = int(lags[candidates[0]]) if len(candidates) else int(lags[int(np.argmax(filled))])
    return lag, maximum


def phase_scores(signal: np.ndarray, period: int) -> np.ndarray:
    scores = np.full(period, np.nan, dtype=np.float64)
    count = len(signal)
    for start in range(period):
        window_count = (count - 1 - start) // period
        if window_count < 2:
            continue
        windows = [
            signal[start + index * period : start + (index + 1) * period]
            for index in range(window_count)
        ]
        values = [
            correlation(windows[index], windows[index + 1])
            for index in range(window_count - 1)
        ]
        values = [value for value in values if np.isfinite(value)]
        if values:
            scores[start] = float(np.mean(values))
    return scores


def prepare_phase_signal(values: np.ndarray) -> np.ndarray | None:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    if valid.sum() < max(8, int(0.30 * len(values))):
        return None
    indexes = np.arange(len(values), dtype=np.float64)
    filled = np.interp(indexes, indexes[valid], values[valid])
    window = min(SMOOTH_WINDOW, len(filled) if len(filled) % 2 == 1 else len(filled) - 1)
    if window >= 5:
        filled = savgol_filter(filled, window, SMOOTH_POLYORDER, mode="interp")
    coefficients = np.polyfit(indexes, filled, 1)
    detrended = filled - np.polyval(coefficients, indexes)
    scale = float(np.std(detrended))
    if scale < 1e-8:
        return None
    return detrended / scale


def circular_distance(first: np.ndarray | float, second: float, period: int) -> np.ndarray:
    return np.abs((np.asarray(first, dtype=np.float64) - second + 0.5 * period) % period - 0.5 * period)


def select_phase_start(
    correlation_scores: np.ndarray,
    phase_signal: np.ndarray,
    period: int,
    guidance: PhaseGuidance,
    boundary_floor: int,
    expected_interval_count: int,
) -> tuple[int | None, dict[str, Any]]:
    prepared = prepare_phase_signal(phase_signal)
    evidence: dict[str, Any] = {
        "status": "fallback",
        "phase_landmark": guidance.phase_landmark,
        "annotation_polarity": guidance.annotation_polarity,
        "signal": {
            "source": guidance.signal.source,
            "name": guidance.signal.name,
            "axis": guidance.signal.axis,
            "sign": guidance.signal.sign,
        },
        "polarity": guidance.polarity,
        "tolerance_frames": max(1, int(round(guidance.tolerance_fraction * period))),
    }
    if prepared is None:
        evidence["reason"] = "phase_signal_unavailable"
        return None, evidence

    oriented = prepared if guidance.polarity == "peak" else -prepared
    low, high = np.percentile(oriented, [5, 95])
    dynamic_range = float(high - low)
    evidence["dynamic_range"] = dynamic_range
    if dynamic_range < 1e-8:
        evidence["reason"] = "flat_phase_signal"
        return None, evidence
    extrema, properties = find_peaks(
        oriented,
        distance=max(4, int(round(0.60 * period))),
        prominence=max(1e-8, guidance.prominence_fraction * dynamic_range),
    )
    evidence["extrema_count"] = len(extrema)
    evidence["extrema_frames"] = [int(index) for index in extrema]
    if len(extrema) < guidance.min_extrema:
        evidence["reason"] = "too_few_extrema"
        return None, evidence
    coverage = len(extrema) / max(1, expected_interval_count)
    evidence["extrema_coverage"] = coverage
    if coverage < guidance.min_extrema_coverage:
        evidence["reason"] = "insufficient_extrema_coverage"
        return None, evidence

    tolerance = max(1, int(round(guidance.tolerance_fraction * period)))
    residuals = np.mod(extrema.astype(np.float64), float(period))
    candidates = []
    for start in range(period):
        distances = circular_distance(residuals, float(start), period)
        support = int(np.sum(distances <= tolerance))
        median_distance = float(np.median(distances))
        correlation_score = (
            float(correlation_scores[start])
            if np.isfinite(correlation_scores[start])
            else -1.0
        )
        candidates.append((support, -median_distance, correlation_score, start))
    candidates.sort(reverse=True)
    best = candidates[0]
    best_support = int(best[0])
    support_fraction = best_support / len(extrema)
    evidence.update(
        {
            "support": best_support,
            "support_fraction": support_fraction,
            "candidate_phase": int(best[3]),
            "candidate_median_distance_frames": float(-best[1]),
        }
    )
    if support_fraction < guidance.min_support_fraction:
        evidence["reason"] = "insufficient_phase_support"
        return None, evidence

    distinct = [
        candidate
        for candidate in candidates[1:]
        if float(circular_distance(float(candidate[3]), float(best[3]), period)) > tolerance
    ]
    second_support = int(distinct[0][0]) if distinct else 0
    ambiguity_margin = guidance.ambiguity_margin_fraction * len(extrema)
    evidence["second_support"] = second_support
    evidence["ambiguity_margin_extrema"] = ambiguity_margin
    if distinct and second_support >= best_support - ambiguity_margin:
        evidence["reason"] = "phase_ambiguity"
        evidence["second_phase"] = int(distinct[0][3])
        return None, evidence

    supporting = residuals[
        circular_distance(residuals, float(best[3]), period) <= tolerance
    ]
    target_phase = int(round(float(np.median(supporting)))) % period
    target_start = target_phase
    if target_start < boundary_floor:
        target_start += int(np.ceil((boundary_floor - target_start) / period)) * period
    evidence.update(
        {
            "status": "applied",
            "target_phase": target_phase,
            "selected_start": target_start,
            "boundary_floor": boundary_floor,
            "prominence": [
                float(value)
                for value in properties.get("prominences", np.asarray([], dtype=np.float64))
            ],
        }
    )
    return target_start, evidence


def snap_boundaries_to_phase(
    boundaries: np.ndarray,
    period: int,
    phase_signal: np.ndarray,
    guidance: PhaseGuidance,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Snap residual Stage-B boundaries to repeated plateau-entry phase events."""
    original = np.asarray(boundaries, dtype=np.int64).copy()
    snap = guidance.boundary_snap
    evidence: dict[str, Any] = {
        "status": "disabled",
        "mode": snap.mode,
    }
    if snap.mode != "plateau_entry":
        return original, evidence

    prepared = prepare_phase_signal(phase_signal)
    if prepared is None:
        evidence.update(
            {
                "status": "fallback",
                "reason": "phase_signal_unavailable",
                "low_phase_confidence": True,
            }
        )
        return original, evidence

    oriented = prepared if guidance.polarity == "peak" else -prepared
    low, high = np.percentile(oriented, [5, 95])
    dynamic_range = float(high - low)
    threshold = low + snap.plateau_fraction * dynamic_range
    minimum_plateau_frames = max(3, int(round(snap.min_plateau_fraction * period)))
    runs = [
        (start, end)
        for start, end in contiguous_runs(oriented >= threshold)
        if end - start + 1 >= minimum_plateau_frames
    ]
    candidates = np.asarray([start for start, _ in runs], dtype=np.int64)
    radius = max(2, int(round(snap.snap_radius_fraction * period)))
    evidence.update(
        {
            "status": "fallback",
            "phase_landmark": guidance.phase_landmark,
            "polarity": guidance.polarity,
            "dynamic_range": dynamic_range,
            "threshold": float(threshold),
            "minimum_plateau_frames": minimum_plateau_frames,
            "plateau_runs": [
                {"start": int(start), "end": int(end), "frames": int(end - start + 1)}
                for start, end in runs
            ],
            "candidates": [int(value) for value in candidates],
            "snap_radius_frames": radius,
        }
    )
    if len(candidates) < 2:
        evidence["reason"] = "too_few_plateau_entries"
        evidence["low_phase_confidence"] = True
        return original, evidence

    snapped = original.copy()
    snap_limit = len(snapped)
    trailing_incomplete_protected = bool(
        len(snapped) >= 3
        and snapped[-1] - snapped[-2] < INCOMPLETE_TAIL_RATIO * period
    )
    if trailing_incomplete_protected:
        snap_limit = max(0, snap_limit - 2)
    decisions: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates[:snap_limit]):
        original_frame = int(original[index])
        candidate_frame = int(candidate)
        shift = candidate_frame - original_frame
        accepted = abs(shift) <= radius
        reason = "accepted" if accepted else "outside_radius"
        if accepted and index > 0 and candidate_frame <= int(snapped[index - 1]):
            accepted = False
            reason = "non_monotonic"
        if accepted:
            snapped[index] = candidate_frame
        decisions.append(
            {
                "boundary_index": index,
                "original": original_frame,
                "candidate": candidate_frame,
                "shift": shift,
                "accepted": accepted,
                "reason": reason,
            }
        )

    minimum_width = max(4, int(round(snap.min_width_fraction * period)))
    widths = np.diff(snapped)
    if len(snapped) < 2 or np.any(widths < minimum_width):
        evidence.update(
            {
                "reason": "invalid_width",
                "minimum_width_frames": minimum_width,
                "decisions": decisions,
                "low_phase_confidence": True,
            }
        )
        return original, evidence

    changed = int(np.count_nonzero(snapped != original))
    leading_plateau_truncated = bool(
        len(candidates)
        and int(candidates[0]) == 0
        and int(snapped[0]) == 0
    )
    evidence.update(
        {
            "status": "applied" if changed else "fallback",
            "reason": "accepted" if changed else "no_accepted_candidates",
            "changed_boundaries": changed,
            "minimum_width_frames": minimum_width,
            "trailing_incomplete_protected": trailing_incomplete_protected,
            "boundaries_before": [int(value) for value in original],
            "boundaries_after": [int(value) for value in snapped],
            "decisions": decisions,
            "leading_plateau_truncated": leading_plateau_truncated,
            "low_phase_confidence": bool(
                not changed
                or any(not decision["accepted"] for decision in decisions)
                or leading_plateau_truncated
            ),
        }
    )
    return snapped, evidence


def stage_a_grid(length: int, start: int, period: int) -> np.ndarray | None:
    if start < 0 or start >= length:
        return None
    count = (length - 1 - start) // period
    if count < 2:
        return None
    boundaries = start + np.arange(count + 1, dtype=np.float64) * period
    tail = (length - 1) - boundaries[-1]
    if tail >= 0.65 * period:
        boundaries = np.append(
            boundaries,
            min(float(length - 1), boundaries[-1] + period),
        )
    return boundaries


def resample_intervals(
    signal: np.ndarray, boundaries: np.ndarray, samples: int = 48
) -> np.ndarray | None:
    curves = []
    indexes = np.arange(len(signal), dtype=np.float64)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < 2:
            return None
        points = np.linspace(start, end, samples, endpoint=False)
        curve = np.interp(points, indexes, signal)
        scale = float(np.std(curve))
        if scale < 1e-8:
            return None
        curves.append((curve - curve.mean()) / scale)
    return np.stack(curves, axis=0)


def boundary_similarity(signal: np.ndarray, boundaries: np.ndarray) -> float:
    curves = resample_intervals(signal, boundaries)
    if curves is None:
        return 0.0
    template = curves.mean(axis=0)
    values = [correlation(curve, template) for curve in curves]
    values = [min(1.0, max(0.0, value)) for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else 0.0


def stage_a(
    signal: np.ndarray,
    period: int,
    phase_signal: np.ndarray | None = None,
    phase_guidance: PhaseGuidance | None = None,
    boundary_floor: int = 0,
) -> tuple[np.ndarray, int, float, dict[str, Any]] | None:
    scores = phase_scores(signal, period)
    if not np.isfinite(scores).any():
        return None
    baseline_start = int(np.nanargmax(scores))
    start = baseline_start
    evidence: dict[str, Any] = {
        "status": "disabled",
        "baseline_start": baseline_start,
    }
    if phase_guidance is not None:
        if phase_signal is None or len(phase_signal) != len(signal):
            evidence.update({"status": "fallback", "reason": "phase_signal_length_mismatch"})
        else:
            baseline_preview = stage_a_grid(len(signal), baseline_start, period)
            selected, evidence = select_phase_start(
                scores,
                phase_signal,
                period,
                phase_guidance,
                boundary_floor,
                max(1, len(baseline_preview) - 1) if baseline_preview is not None else 1,
            )
            if selected is not None:
                start = selected
            evidence["baseline_start"] = baseline_start
            evidence["baseline_correlation"] = float(scores[baseline_start])
            if start != baseline_start:
                evidence["shift_frames"] = int(start - baseline_start)
    boundaries = stage_a_grid(len(signal), start, period)
    if boundaries is None:
        return None
    if phase_guidance is not None and start != baseline_start:
        baseline_boundaries = stage_a_grid(len(signal), baseline_start, period)
        if baseline_boundaries is None or len(boundaries) != len(baseline_boundaries):
            evidence.update(
                {
                    "status": "fallback",
                    "reason": "phase_changes_boundary_count",
                    "candidate_boundary_count": len(boundaries),
                    "baseline_boundary_count": (
                        len(baseline_boundaries) if baseline_boundaries is not None else None
                    ),
                }
            )
            boundaries = baseline_boundaries
            start = baseline_start
            if boundaries is None:
                return None
    return boundaries, start, boundary_similarity(signal, boundaries), evidence


def refine_boundaries(
    signal: np.ndarray, initial: np.ndarray, period: int
) -> tuple[np.ndarray, float, float]:
    count = len(signal)
    minimum_width = max(6.0, 0.65 * period)
    offset = 0.20 * period
    boundaries = initial.astype(np.float64).copy()
    lower = np.maximum(0.0, initial - offset)
    upper = np.minimum(count - 1.0, initial + offset)

    def objective(candidate: np.ndarray) -> float:
        widths = np.diff(candidate)
        if np.any(widths < minimum_width):
            return 1e9 + float(np.sum((minimum_width - widths[widths < minimum_width]) ** 2))
        curves = resample_intervals(signal, candidate)
        if curves is None:
            return 1e9
        template = curves.mean(axis=0)
        return float(np.mean((curves - template) ** 2))

    best = objective(boundaries)
    step = max(2, int(round(period * 0.08)))
    for _ in range(5):
        changed = False
        for index in range(len(boundaries)):
            for delta in (-2 * step, -step, step, 2 * step):
                candidate = boundaries.copy()
                candidate[index] += delta
                if candidate[index] < lower[index] or candidate[index] > upper[index]:
                    continue
                if index > 0 and candidate[index] - candidate[index - 1] < minimum_width:
                    continue
                if index + 1 < len(candidate) and candidate[index + 1] - candidate[index] < minimum_width:
                    continue
                value = objective(candidate)
                if value < best - 1e-7:
                    boundaries = candidate
                    best = value
                    changed = True
        if not changed:
            step = max(1, step // 2)
            if step == 1:
                break
    return boundaries, best, minimum_width


def finalize_boundaries(
    boundaries: np.ndarray, minimum_width: float, count: int
) -> np.ndarray | None:
    rounded = np.rint(boundaries).astype(int)
    rounded[0] = max(0, rounded[0])
    for index in range(1, len(rounded)):
        rounded[index] = max(rounded[index], rounded[index - 1] + int(np.ceil(minimum_width)))
    rounded[-1] = min(count - 1, rounded[-1])
    if len(rounded) < 3 or np.any(np.diff(rounded) < minimum_width):
        return None
    return rounded


def reps_from_boundaries(
    boundaries: np.ndarray,
    valid: np.ndarray,
    signal: np.ndarray,
    offset: int,
    force_final_low_amplitude: bool = False,
) -> tuple[list[Rep], float, list[RejectedRep]]:
    curves = resample_intervals(signal, boundaries.astype(np.float64))
    if curves is None:
        return [], 0.0, []
    low, high = np.percentile(signal, [5, 95])
    global_range = max(1e-8, float(high - low))
    keep = []
    rejected = []
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        segment = signal[int(start) : int(end)]
        if len(segment) < 4:
            rejected.append(
                RejectedRep(
                    start_frame=offset + int(start),
                    end_frame=offset + int(end) - 1,
                    reason="short_rep",
                    evidence={"segment_frames": len(segment)},
                )
            )
            continue
        local_low, local_high = np.percentile(segment, [5, 95])
        local_range = float(local_high - local_low)
        force_low_amplitude = force_final_low_amplitude and index == len(boundaries) - 2
        if local_range < 0.25 * global_range and not force_low_amplitude:
            rejected.append(
                RejectedRep(
                    start_frame=offset + int(start),
                    end_frame=offset + int(end) - 1,
                    reason="low_local_amplitude",
                    evidence={
                        "local_range": local_range,
                        "global_range": global_range,
                        "local_range_fraction": local_range / global_range,
                    },
                )
            )
            continue
        keep.append(index)
    if len(keep) < 2:
        return [], 0.0, rejected
    kept_curves = curves[keep]
    template = kept_curves.mean(axis=0)
    reps = []
    similarities = []
    for rep_id, index in enumerate(keep, start=1):
        start, end = boundaries[index], boundaries[index + 1]
        similarity = correlation(curves[index], template)
        similarity = min(1.0, max(0.0, similarity)) if np.isfinite(similarity) else 0.0
        valid_fraction = float(valid[start:end].mean()) if end > start else 0.0
        confidence = valid_fraction * similarity
        similarities.append(similarity)
        reps.append(
            Rep(
                rep_id=rep_id,
                start_frame=offset + int(start),
                end_frame=offset + int(end) - 1,
                peak_frame=None,
                confidence=confidence,
            )
        )
    return reps, float(np.mean(similarities)) if similarities else 0.0, rejected


def final_rep_aligns_phase(
    boundaries: np.ndarray,
    phase_evidence: dict[str, Any],
) -> bool:
    if phase_evidence.get("status") != "applied" or len(boundaries) < 3:
        return False
    extrema = phase_evidence.get("extrema_frames", [])
    if not extrema:
        return False
    tolerance = int(phase_evidence.get("tolerance_frames", 0))
    return min(abs(float(boundaries[-2]) - float(frame)) for frame in extrema) <= tolerance


def run_aifit(
    signal: np.ndarray,
    valid: np.ndarray,
    angle: str,
    offset: int,
    fps: float,
    min_period_s: float,
    max_period_s: float,
    use_stage_b: bool,
    phase_signal: np.ndarray | None = None,
    phase_guidance: PhaseGuidance | None = None,
    boundary_floor: int = 0,
    use_phase_snap: bool = True,
    shape_template_method: str = DEFAULT_SHAPE_TEMPLATE,
) -> MethodResult | None:
    period_result = estimate_period(signal, fps, min_period_s, max_period_s)
    if period_result is None:
        return None
    period, periodicity = period_result
    initial = stage_a(
        signal,
        period,
        phase_signal=phase_signal,
        phase_guidance=phase_guidance,
        boundary_floor=boundary_floor,
    )
    if initial is None:
        return None
    boundaries, start_a, similarity_a, phase_evidence = initial
    boundaries_final = boundaries
    similarity = similarity_a
    objective = 1.0 - similarity_a
    stage_b_applied = False
    if use_stage_b:
        refined, refined_objective, minimum_width = refine_boundaries(signal, boundaries, period)
        rounded = finalize_boundaries(refined, minimum_width, len(signal))
        if rounded is not None and len(rounded) == len(boundaries):
            similarity_b = boundary_similarity(signal, rounded.astype(np.float64))
            if similarity_b + 1e-6 >= similarity_a:
                boundaries_final = rounded.astype(np.float64)
                similarity = similarity_b
                objective = refined_objective
                stage_b_applied = True
    rounded = finalize_boundaries(
        boundaries_final,
        max(6.0, 0.65 * period),
        len(signal),
    )
    if rounded is None or len(rounded) < 3:
        return None
    force_final_low_amplitude = final_rep_aligns_phase(rounded, phase_evidence)
    reps, rep_similarity, rejected = reps_from_boundaries(
        rounded,
        valid,
        signal,
        offset,
        force_final_low_amplitude=force_final_low_amplitude,
    )
    if stage_b_applied and phase_evidence.get("status") == "applied":
        stage_a_rounded = finalize_boundaries(
            boundaries,
            max(6.0, 0.65 * period),
            len(signal),
        )
        if stage_a_rounded is not None and len(stage_a_rounded) >= 3:
            stage_a_reps, stage_a_similarity, stage_a_rejected = reps_from_boundaries(
                stage_a_rounded,
                valid,
                signal,
                offset,
                force_final_low_amplitude=final_rep_aligns_phase(
                    stage_a_rounded,
                    phase_evidence,
                ),
            )
            if len(stage_a_reps) > len(reps):
                rounded = stage_a_rounded
                reps = stage_a_reps
                rep_similarity = stage_a_similarity
                rejected = stage_a_rejected
                similarity = similarity_a
                objective = 1.0 - similarity_a
                phase_evidence["stage_b_fallback"] = "phase_guided_rep_count_decreased"
    if use_phase_snap and phase_guidance is not None:
        snapped, snap_evidence = snap_boundaries_to_phase(
            rounded,
            period,
            phase_signal if phase_signal is not None else np.asarray([]),
            phase_guidance,
        )
        snap_evidence["boundary_floor"] = int(boundary_floor)
        phase_evidence["boundary_snap"] = snap_evidence
        if snap_evidence.get("status") == "applied":
            snapped_reps, snapped_similarity, snapped_rejected = reps_from_boundaries(
                snapped,
                valid,
                signal,
                offset,
                force_final_low_amplitude=force_final_low_amplitude,
            )
            if len(snapped_reps) == len(reps):
                rounded = snapped
                reps = snapped_reps
                rep_similarity = snapped_similarity
                rejected = snapped_rejected
                similarity = boundary_similarity(signal, snapped.astype(np.float64))
                objective = 1.0 - similarity
            else:
                snap_evidence["status"] = "fallback"
                snap_evidence["reason"] = "snap_changes_rep_count"
                snap_evidence["low_phase_confidence"] = True
                snap_evidence["candidate_rep_count"] = len(snapped_reps)
                snap_evidence["baseline_rep_count"] = len(reps)
    if force_final_low_amplitude:
        phase_evidence["final_low_amplitude_accepted"] = "phase_aligned_edge_rep"
    if len(reps) < 2:
        return None
    widths = np.asarray([rep.end_frame - rep.start_frame + 1 for rep in reps])
    reps = enrich_rep_confidences(
        reps,
        signal=signal,
        valid=valid,
        offset=offset,
        template_method=shape_template_method,
        phase_evidence=phase_evidence,
        period_frames=float(np.median(widths)),
        phase_signal_available=phase_signal is not None,
    )
    rep_similarity = float(
        np.mean([rep.shape_similarity for rep in reps])
    ) if reps else 0.0
    return MethodResult(
        method="aifit",
        angle=angle,
        reps=reps,
        period_frames=float(np.median(widths)),
        similarity=max(similarity, rep_similarity),
        valid_fraction=float(valid.mean()),
        objective=float(objective),
        rejected_reps=rejected,
        phase_guidance={
            **phase_evidence,
            "period_frames": int(period),
            "start_local_frame": int(start_a),
            "start_frame": int(offset + start_a),
        },
    )


def peak_prominence_score(signal: np.ndarray) -> float:
    peaks, properties = find_peaks(signal, distance=3, prominence=0)
    if len(peaks) == 0:
        return -np.inf
    prominences = np.sort(properties.get("prominences", np.asarray([])))
    if len(prominences) == 0:
        return -np.inf
    return float(prominences[-min(5, len(prominences)) :].sum())


def adaptive_prominence(
    signal: np.ndarray,
    dynamic_range: float,
    distance: int,
    default_fraction: float,
) -> float:
    peaks, properties = find_peaks(
        signal,
        distance=distance,
        prominence=max(1.0, 0.05 * dynamic_range),
    )
    prominences = np.sort(properties.get("prominences", np.asarray([])))
    threshold = default_fraction * dynamic_range
    if len(peaks) >= 6 and len(prominences) >= 6:
        gaps = np.diff(prominences)
        gap_index = int(np.argmax(gaps))
        gap = float(gaps[gap_index])
        if gap >= 0.15 * dynamic_range:
            threshold = float(prominences[gap_index] + 0.25 * gap)
    return float(np.clip(threshold, 0.10 * dynamic_range, 0.60 * dynamic_range))


def run_peak_baseline(
    signal: np.ndarray,
    valid: np.ndarray,
    angle: str,
    offset: int,
    fps: float,
    min_period_s: float,
    max_period_s: float,
    prominence_fraction: float,
) -> MethodResult | None:
    raw_score = peak_prominence_score(signal)
    inverted_score = peak_prominence_score(-signal)
    oriented = signal if raw_score >= inverted_score else -signal
    low, high = np.percentile(oriented, [5, 95])
    dynamic_range = float(high - low)
    if dynamic_range < MIN_ACTIVE_RANGE_DEG:
        return None
    distance = max(4, int(round(min_period_s * fps)))
    prominence = max(
        4.0,
        adaptive_prominence(oriented, dynamic_range, distance, prominence_fraction),
    )
    peaks, properties = find_peaks(
        oriented,
        distance=distance,
        prominence=prominence,
    )
    if len(peaks) == 0:
        return None
    minima, _ = find_peaks(
        -oriented,
        distance=max(2, int(round(0.20 * min_period_s * fps))),
    )
    boundaries = []
    before = minima[minima < peaks[0]]
    boundaries.append(int(before[-1]) if len(before) else 0)
    for first_peak, second_peak in zip(peaks[:-1], peaks[1:]):
        window = np.arange(first_peak, second_peak + 1)
        boundaries.append(int(window[int(np.argmin(oriented[window]))]))
    after = minima[minima > peaks[-1]]
    boundaries.append(int(after[0]) if len(after) else len(signal) - 1)
    if len(boundaries) != len(peaks) + 1:
        return None
    minimum_width = max(4, int(round(0.20 * min_period_s * fps)))
    maximum_width = max(minimum_width + 1, int(round(1.50 * max_period_s * fps)))
    reps = []
    prominences = properties.get("prominences", np.zeros(len(peaks)))
    global_low, global_high = np.percentile(oriented, [5, 95])
    global_range = max(1e-8, float(global_high - global_low))
    for index, (start, end, peak, peak_prominence) in enumerate(
        zip(boundaries[:-1], boundaries[1:], peaks, prominences)
    ):
        width = end - start
        if width < minimum_width or width > maximum_width:
            continue
        local_low, local_high = np.percentile(oriented[start:end], [5, 95])
        if float(local_high - local_low) < 0.25 * global_range:
            continue
        valid_fraction = float(valid[start:end].mean()) if end > start else 0.0
        prominence_score = min(1.0, float(peak_prominence) / max(1.0, 0.25 * dynamic_range))
        reps.append(
            Rep(
                rep_id=len(reps) + 1,
                start_frame=offset + int(start),
                end_frame=offset + int(end) - 1,
                peak_frame=offset + int(peak),
                confidence=valid_fraction * prominence_score,
            )
        )
    if len(reps) < 2:
        return None
    widths = np.asarray([rep.end_frame - rep.start_frame + 1 for rep in reps])
    return MethodResult(
        method="peak",
        angle=angle,
        reps=reps,
        period_frames=float(np.median(widths)),
        similarity=float(np.mean([rep.confidence for rep in reps])),
        valid_fraction=float(valid.mean()),
        objective=0.0,
    )


def interval_evidence(
    trace: ActivityTrace,
    start: int,
    end: int,
) -> dict[str, Any]:
    activity = trace.activity[start : end + 1]
    visibility = trace.visibility[start : end + 1]
    speed = trace.main_speed[start : end + 1]
    return {
        "activity_mean": float(np.mean(activity)),
        "activity_median": float(np.median(activity)),
        "activity_min": float(np.min(activity)),
        "low_activity_fraction": float(np.mean(activity < trace.evidence["threshold"])),
        "visibility_mean": float(np.mean(visibility)),
        "main_speed_median": float(np.median(speed)),
        "main_speed_max": float(np.max(speed)),
    }


def active_bout_windows(
    prepared: PreparedSignal,
    low_intervals: list[tuple[int, int]],
    count: int,
) -> list[dict[str, int]]:
    """Split the original effective window into active cores and phase-preserving fit windows."""
    core_regions = []
    cursor = 0
    for start, end in low_intervals:
        if start > cursor:
            core_regions.append((cursor, start - 1))
        cursor = max(cursor, end + 1)
    if cursor < count:
        core_regions.append((cursor, count - 1))

    windows = []
    for start, end in core_regions:
        previous_low = [item for item in low_intervals if item[1] < start]
        next_low = [item for item in low_intervals if item[0] > end]
        fit_start = (
            (previous_low[-1][0] + previous_low[-1][1]) // 2 + 1
            if previous_low
            else prepared.source_start
        )
        fit_end = (
            (next_low[0][0] + next_low[0][1]) // 2
            if next_low
            else prepared.source_end
        )
        start = max(start, prepared.source_start)
        end = min(end, prepared.source_end)
        fit_start = max(prepared.source_start, min(fit_start, end))
        fit_end = min(prepared.source_end, max(fit_end, start))
        if end < start or fit_end < fit_start:
            continue
        windows.append(
            {
                "start": int(start),
                "end": int(end),
                "fit_start": int(fit_start),
                "fit_end": int(fit_end),
            }
        )
    return windows


def reject_incomplete_tail(
    result: MethodResult,
    fit_end_frame: int,
    fps: float,
) -> None:
    if len(result.reps) < 3:
        return
    widths = np.asarray(
        [rep.end_frame - rep.start_frame + 1 for rep in result.reps],
        dtype=np.float64,
    )
    reference_width = float(np.median(widths))
    if reference_width < fps:
        return
    last = result.reps[-1]
    last_width = last.end_frame - last.start_frame + 1
    end_tolerance = max(3, int(round(0.25 * reference_width)))
    if (
        last_width >= INCOMPLETE_TAIL_RATIO * reference_width
        or last.end_frame < fit_end_frame - end_tolerance
    ):
        return
    result.rejected_reps.append(
        RejectedRep(
            start_frame=last.start_frame,
            end_frame=fit_end_frame,
            reason="incomplete_rep",
            evidence={
                "predicted_end_frame": last.end_frame,
                "rep_frames": last_width,
                "reference_width_frames": reference_width,
                "width_ratio": last_width / reference_width,
                "fit_end_frame": fit_end_frame,
            },
        )
    )
    result.reps.pop()
    result.reps = renumber_reps(result.reps)
    remaining_widths = np.asarray(
        [rep.end_frame - rep.start_frame + 1 for rep in result.reps],
        dtype=np.float64,
    )
    result.period_frames = float(np.median(remaining_widths))


def phase_extrema_frames(phase_guidance: dict[str, Any] | None) -> list[int]:
    if not phase_guidance:
        return []
    entries = phase_guidance.get("bouts") or [phase_guidance]
    frames = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for value in entry.get("extrema_frames", []):
            try:
                frames.append(int(value))
            except (TypeError, ValueError):
                continue
    return sorted(set(frames))


def reject_air_jump_rope_transition_tail(
    result: MethodResult,
    segment: ActionSegment,
    fit_end_frame: int,
) -> bool:
    """Remove only a guarded, evidence-backed final transition for air jump rope."""
    if segment.action != "air_jump_rope" or len(result.reps) < 4:
        return False
    last = result.reps[-1]
    reference_width = float(result.period_frames or np.median(
        [rep.end_frame - rep.start_frame + 1 for rep in result.reps[:-1]]
    ))
    if reference_width <= 0:
        return False
    end_tolerance = max(3, int(round(AIR_JUMP_TAIL_END_TOLERANCE_FRACTION * reference_width)))
    tail_gap = int(fit_end_frame - last.end_frame)
    if tail_gap < -3 or tail_gap > end_tolerance:
        return False

    extrema = [
        segment.start_frame + frame
        for frame in phase_extrema_frames(result.phase_guidance)
    ]
    phase_available = len(extrema) >= 3
    has_phase_apex = any(last.start_frame <= frame <= last.end_frame for frame in extrema)
    missing_phase_apex = phase_available and not has_phase_apex

    previous = result.reps[:-1]
    shape_reference = float(np.median([rep.shape_similarity for rep in previous]))
    motion_reference = float(np.median([rep.motion_evidence for rep in previous]))
    shape_limit = max(AIR_JUMP_TAIL_SHAPE_FLOOR, AIR_JUMP_TAIL_SHAPE_RATIO * shape_reference)
    motion_limit = min(AIR_JUMP_TAIL_MOTION_FLOOR, AIR_JUMP_TAIL_MOTION_RATIO * motion_reference)
    low_shape = last.shape_similarity < shape_limit
    low_motion = last.motion_evidence < motion_limit
    support_count = int(missing_phase_apex) + int(low_shape) + int(low_motion)
    if support_count < 2:
        return False

    result.rejected_reps.append(
        RejectedRep(
            start_frame=last.start_frame,
            end_frame=last.end_frame,
            reason="air_jump_rope_transition_tail",
            evidence={
                "fit_end_frame": int(fit_end_frame),
                "tail_gap_frames": tail_gap,
                "tail_end_tolerance_frames": end_tolerance,
                "period_frames": reference_width,
                "phase_extrema_available": phase_available,
                "phase_apex_frames_in_rep": [
                    frame
                    for frame in extrema
                    if last.start_frame <= frame <= last.end_frame
                ],
                "missing_phase_apex": missing_phase_apex,
                "shape_similarity": last.shape_similarity,
                "shape_limit": shape_limit,
                "shape_reference_median": shape_reference,
                "low_shape_similarity": low_shape,
                "motion_evidence": last.motion_evidence,
                "motion_limit": motion_limit,
                "motion_reference_median": motion_reference,
                "low_motion_evidence": low_motion,
                "support_count": support_count,
            },
        )
    )
    result.reps.pop()
    result.reps = renumber_reps(result.reps)
    remaining_widths = np.asarray(
        [rep.end_frame - rep.start_frame + 1 for rep in result.reps],
        dtype=np.float64,
    )
    result.period_frames = float(np.median(remaining_widths))
    return True


def combine_method_results(
    results: list[MethodResult],
    angle: str,
) -> MethodResult | None:
    if not results:
        return None
    combined_reps = []
    rejected = []
    for result in results:
        for rep in result.reps:
            combined_reps.append(replace(rep, rep_id=len(combined_reps) + 1))
        rejected.extend(result.rejected_reps)
    if not combined_reps:
        return None
    weights = np.asarray([len(result.reps) for result in results], dtype=np.float64)
    total_weight = float(weights.sum())
    phase_bouts = [
        result.phase_guidance
        for result in results
        if result.phase_guidance
    ]
    phase_statuses = {item.get("status") for item in phase_bouts}
    if not phase_bouts:
        phase_status = "disabled"
    elif phase_statuses == {"applied"}:
        phase_status = "applied"
    elif phase_statuses == {"fallback"}:
        phase_status = "fallback"
    else:
        phase_status = "mixed"
    return MethodResult(
        method="aifit",
        angle=angle,
        reps=combined_reps,
        period_frames=float(np.median([result.period_frames for result in results])),
        similarity=float(
            sum(result.similarity * weight for result, weight in zip(results, weights))
            / total_weight
        ),
        valid_fraction=float(
            sum(result.valid_fraction * weight for result, weight in zip(results, weights))
            / total_weight
        ),
        objective=float(
            sum(result.objective * weight for result, weight in zip(results, weights))
            / total_weight
        ),
        rejected_reps=sorted(rejected, key=lambda item: item.start_frame),
        phase_guidance={
            "status": phase_status,
            "bouts": phase_bouts,
        },
    )


def non_rep_intervals_from_bouts(
    bouts: list[ActiveBout],
    rejected_reps: list[RejectedRep],
    trace: ActivityTrace,
    low_intervals: list[tuple[int, int]],
    segment: ActionSegment,
    count: int,
) -> list[NonRepInterval]:
    active = [bout for bout in bouts if bout.status == "active"]
    intervals: list[NonRepInterval] = []
    if not active:
        intervals.append(
            NonRepInterval(
                start_frame=segment.start_frame,
                end_frame=segment.end_frame,
                reason="low_activity",
                evidence={"reason": "no active bout produced at least one rep"},
            )
        )
    else:
        cursor = 0
        for index, bout in enumerate(active):
            start = bout.start_frame - segment.start_frame
            end = bout.end_frame - segment.start_frame
            if start > cursor:
                gap_start, gap_end = cursor, start - 1
                if index == 0:
                    reason = "transition"
                else:
                    overlapping = [
                        (low_start, low_end)
                        for low_start, low_end in low_intervals
                        if min(low_end, gap_end) >= max(low_start, gap_start)
                    ]
                    reason = "rest" if overlapping else "low_activity"
                intervals.append(
                    NonRepInterval(
                        start_frame=segment.start_frame + gap_start,
                        end_frame=segment.start_frame + gap_end,
                        reason=reason,
                        evidence=interval_evidence(trace, gap_start, gap_end),
                    )
                )
            cursor = max(cursor, end + 1)
        if cursor < count:
            intervals.append(
                NonRepInterval(
                    start_frame=segment.start_frame + cursor,
                    end_frame=segment.end_frame,
                    reason="transition",
                    evidence=interval_evidence(trace, cursor, count - 1),
                )
            )

    for rejected in rejected_reps:
        start = max(0, rejected.start_frame - segment.start_frame)
        end = min(count - 1, rejected.end_frame - segment.start_frame)
        intervals.append(
            NonRepInterval(
                start_frame=rejected.start_frame,
                end_frame=rejected.end_frame,
                reason=rejected.reason,
                evidence=rejected.evidence,
            )
        )
    return sorted(intervals, key=lambda item: (item.start_frame, item.end_frame))


def filter_reps_to_active_core(
    result: MethodResult,
    core_start_frame: int,
    core_end_frame: int,
) -> None:
    """Drop reps that lie entirely in a low-activity fit-extension region."""
    kept: list[Rep] = []
    removed: list[RejectedRep] = []
    for rep in result.reps:
        overlap = max(
            0,
            min(rep.end_frame, core_end_frame) - max(rep.start_frame, core_start_frame) + 1,
        )
        if overlap:
            kept.append(rep)
            continue
        removed.append(
            RejectedRep(
                start_frame=rep.start_frame,
                end_frame=rep.end_frame,
                reason="outside_active_bout",
                evidence={
                    "active_start_frame": core_start_frame,
                    "active_end_frame": core_end_frame,
                    "overlap_frames": 0,
                },
            )
        )
    if not removed:
        return
    result.rejected_reps.extend(removed)
    result.reps = renumber_reps(kept)
    if result.reps:
        widths = np.asarray(
            [rep.end_frame - rep.start_frame + 1 for rep in result.reps],
            dtype=np.float64,
        )
        result.period_frames = float(np.median(widths))
        result.similarity = float(np.mean([rep.confidence for rep in result.reps]))


def run_aifit_with_active_bouts(
    values: np.ndarray,
    prepared: PreparedSignal,
    trace: ActivityTrace,
    low_intervals: list[tuple[int, int]],
    segment: ActionSegment,
    angle: str,
    fps: float,
    min_period_s: float,
    max_period_s: float,
    use_stage_b: bool,
    phase_values: np.ndarray | None = None,
    phase_guidance: PhaseGuidance | None = None,
    use_phase_snap: bool = True,
    shape_template_method: str = DEFAULT_SHAPE_TEMPLATE,
) -> tuple[MethodResult | None, list[ActiveBout], list[NonRepInterval]]:
    count = len(values)
    windows = active_bout_windows(prepared, low_intervals, count)
    results = []
    bouts = []
    all_rejected = []
    for bout_id, window in enumerate(windows, start=1):
        fit_values = values[window["fit_start"] : window["fit_end"] + 1]
        bout_prepared = prepare_core(fit_values)
        result = None
        if bout_prepared is not None:
            fit_offset = segment.start_frame + window["fit_start"]
            phase_fit = (
                phase_values[window["fit_start"] : window["fit_end"] + 1]
                if phase_values is not None
                else None
            )
            result = run_aifit(
                bout_prepared.normalized,
                bout_prepared.valid,
                angle,
                fit_offset,
                fps,
                min_period_s,
                max_period_s,
                use_stage_b,
                phase_signal=phase_fit,
                phase_guidance=phase_guidance,
                boundary_floor=window["start"] - window["fit_start"],
                use_phase_snap=use_phase_snap,
                shape_template_method=shape_template_method,
            )
        if result is not None:
            filter_reps_to_active_core(
                result,
                segment.start_frame + window["start"],
                segment.start_frame + window["end"],
            )
            reject_incomplete_tail(
                result,
                segment.start_frame + window["fit_end"],
                fps,
            )
            result.reps = enrich_rep_confidences(
                result.reps,
                trace=trace,
                segment_start_frame=segment.start_frame,
            )
            all_rejected.extend(result.rejected_reps)
        status = "active" if result is not None and result.reps else "no_reps"
        if status == "active":
            results.append(result)
        evidence = interval_evidence(trace, window["start"], window["end"])
        evidence.update(
            {
                "fit_start_offset_frames": window["fit_start"] - window["start"],
                "fit_end_offset_frames": window["fit_end"] - window["end"],
                "num_reps": len(result.reps) if result is not None else 0,
                "local_period_frames": result.period_frames if result is not None else 0.0,
                "phase_guidance": result.phase_guidance if result is not None else None,
            }
        )
        bouts.append(
            ActiveBout(
                bout_id=bout_id,
                start_frame=segment.start_frame + window["start"],
                end_frame=segment.start_frame + window["end"],
                fit_start_frame=segment.start_frame + window["fit_start"],
                fit_end_frame=segment.start_frame + window["fit_end"],
                status=status,
                evidence=evidence,
            )
        )
    combined = combine_method_results(results, angle)
    rejected = combined.rejected_reps if combined is not None else all_rejected
    non_rep = non_rep_intervals_from_bouts(
        bouts,
        rejected,
        trace,
        low_intervals,
        segment,
        count,
    )
    return combined, bouts, non_rep


def action_period_range_s(
    action: str,
    fallback_min_s: float,
    fallback_max_s: float,
) -> tuple[float, float]:
    action_min_s, action_max_s = ACTION_PERIOD_RANGES_S.get(
        action,
        (fallback_min_s, fallback_max_s),
    )
    minimum = max(fallback_min_s, action_min_s)
    maximum = min(fallback_max_s, action_max_s)
    if maximum <= minimum:
        return fallback_min_s, fallback_max_s
    return minimum, maximum


def peak_prominence_evidence(signal: np.ndarray, period: int) -> float:
    raw_score = peak_prominence_score(signal)
    inverted_score = peak_prominence_score(-signal)
    oriented = signal if raw_score >= inverted_score else -signal
    low, high = np.percentile(oriented, [5, 95])
    dynamic_range = float(high - low)
    if dynamic_range < 1e-8:
        return 0.0
    peaks, properties = find_peaks(
        oriented,
        distance=max(3, int(round(0.50 * period))),
        prominence=max(1e-8, 0.05 * dynamic_range),
    )
    if len(peaks) < 2:
        return 0.0
    prominences = np.sort(
        np.asarray(properties.get("prominences", []), dtype=np.float64)
    )
    if len(prominences) == 0:
        return 0.0
    return float(np.clip(np.median(prominences[len(prominences) // 2 :]) / dynamic_range, 0.0, 1.0))


def build_angle_candidate(
    angle: str,
    prepared: PreparedSignal,
    fps: float,
    min_period_s: float,
    max_period_s: float,
) -> AngleCandidate | None:
    period_result = estimate_period(
        prepared.normalized,
        fps,
        min_period_s,
        max_period_s,
    )
    if period_result is None:
        return None
    period_frames, periodicity = period_result
    period = max(4, int(period_frames))
    scores = phase_scores(prepared.normalized, period)
    finite_scores = scores[np.isfinite(scores)]
    phase_consistency = (
        float(np.clip(np.max(finite_scores), 0.0, 1.0))
        if len(finite_scores)
        else 0.0
    )
    stage = stage_a(prepared.normalized, period)
    boundary_stability = (
        float(np.clip(stage[2], 0.0, 1.0))
        if stage is not None
        else 0.0
    )
    prominence = peak_prominence_evidence(prepared.normalized, period)
    components = {
        "periodicity": float(np.clip(periodicity, 0.0, 1.0)),
        "peak_prominence": prominence,
        "phase_consistency": phase_consistency,
        "boundary_stability": boundary_stability,
    }
    score = float(
        sum(ANGLE_EVIDENCE_WEIGHTS[name] * value for name, value in components.items())
    )
    return AngleCandidate(
        angle=angle,
        prepared=prepared,
        period_frames=period,
        periodicity=components["periodicity"],
        peak_prominence=prominence,
        phase_consistency=phase_consistency,
        boundary_stability=boundary_stability,
        score=score,
    )


def cluster_angle_candidates(candidates: list[AngleCandidate]) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda item: (item.period_frames or 0, item.angle))
    groups: list[list[AngleCandidate]] = []
    for candidate in ordered:
        period = float(candidate.period_frames or 0)
        if not groups:
            groups.append([candidate])
            continue
        group_period = float(np.median([item.period_frames or 0 for item in groups[-1]]))
        if group_period > 0 and abs(period - group_period) / group_period <= ANGLE_CLUSTER_RELATIVE_PERIOD_TOLERANCE:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])

    clusters = []
    for cluster_id, group in enumerate(groups, start=1):
        periods = np.asarray([item.period_frames or 0 for item in group], dtype=np.float64)
        center = float(np.median(periods))
        dispersion = float(np.median(np.abs(periods - center)) / max(1.0, center))
        period_consistency = float(
            np.clip(
                1.0 - dispersion / ANGLE_CLUSTER_RELATIVE_PERIOD_TOLERANCE,
                0.0,
                1.0,
            )
        )
        best = max(group, key=lambda item: (item.score, item.periodicity))
        size_support = min(1.0, max(0, len(group) - 1) / 2.0)
        cluster_score = float(
            0.85 * best.score + 0.10 * period_consistency + 0.05 * size_support
        )
        clusters.append(
            {
                "cluster_id": cluster_id,
                "angles": [item.angle for item in group],
                "num_angles": len(group),
                "period_frames": center,
                "period_consistency": period_consistency,
                "period_dispersion": dispersion,
                "best_angle": best.angle,
                "best_angle_score": best.score,
                "cluster_score": cluster_score,
                "selected": False,
            }
        )
        for candidate in group:
            candidate.cluster_id = cluster_id
    return clusters


def legacy_angle_choice(
    wide: dict[str, np.ndarray],
    segment: ActionSegment,
    candidate_angles: tuple[str, ...],
    fps: float,
    min_period_s: float,
    max_period_s: float,
) -> tuple[str, PreparedSignal, float] | None:
    scored = []
    for angle in candidate_angles:
        if angle not in wide:
            continue
        values = wide[angle][segment.start_frame : segment.end_frame + 1]
        prepared = prepare_signal(values)
        if prepared is None:
            continue
        score = prepared.dynamic_range * np.sqrt(prepared.valid_fraction)
        scored.append((score, prepared.dynamic_range, angle, prepared))
    if scored:
        reliable = [item for item in scored if item[3].valid_fraction >= 0.50]
        if reliable:
            scored = reliable
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, angle, prepared = scored[0]
    period_result = estimate_period(
        prepared.normalized,
        fps,
        min_period_s,
        max_period_s,
    )
    periodicity = float(period_result[1]) if period_result is not None else 0.0
    return angle, prepared, periodicity


def choose_auto_angle(
    wide: dict[str, np.ndarray],
    segment: ActionSegment,
    fps: float,
    min_period_s: float,
    max_period_s: float,
    scoring: str = "legacy",
) -> AngleSelection | None:
    candidate_angles = ACTION_CANDIDATE_ANGLES.get(segment.action, MIXED_ANGLES)
    legacy = legacy_angle_choice(
        wide,
        segment,
        candidate_angles,
        fps,
        min_period_s,
        max_period_s,
    )
    if scoring == "legacy":
        if legacy is None:
            return None
        angle, prepared, periodicity = legacy
        return AngleSelection(
            angle=angle,
            prepared=prepared,
            periodicity=periodicity,
            evidence={"method": "legacy"},
        )

    period_min_s, period_max_s = action_period_range_s(
        segment.action,
        min_period_s,
        max_period_s,
    )
    candidates = []
    for angle in candidate_angles:
        if angle not in wide:
            continue
        values = wide[angle][segment.start_frame : segment.end_frame + 1]
        prepared = prepare_signal(values)
        if prepared is None:
            continue
        if prepared.valid_fraction < 0.50 or prepared.dynamic_range < MIN_ACTIVE_RANGE_DEG:
            continue
        candidate = build_angle_candidate(
            angle,
            prepared,
            fps,
            period_min_s,
            period_max_s,
        )
        if candidate is not None:
            candidates.append(candidate)

    evidence: dict[str, Any] = {
        "method": "evidence",
        "status": "fallback",
        "selected_by": "legacy",
        "angle_ambiguous": False,
        "period_range_s": [period_min_s, period_max_s],
        "weights": dict(ANGLE_EVIDENCE_WEIGHTS),
        "candidates": [],
        "clusters": [],
    }
    if legacy is None and not candidates:
        return None
    if not candidates:
        angle, prepared, periodicity = legacy  # type: ignore[misc]
        evidence["reason"] = "no_periodic_candidate"
        evidence.update(
            {
                "selected_angle": angle,
                "legacy_angle": angle,
                "period_frames": None,
            }
        )
        return AngleSelection(angle, prepared, periodicity, evidence)

    clusters = cluster_angle_candidates(candidates)
    clusters.sort(key=lambda item: item["cluster_score"], reverse=True)
    selected_cluster = clusters[0]
    selected_cluster["selected"] = True
    selected = next(
        item for item in candidates if item.angle == selected_cluster["best_angle"]
    )
    selected_by = "cluster"
    cluster_score_margin = (
        float(selected_cluster["cluster_score"] - clusters[1]["cluster_score"])
        if len(clusters) > 1
        else float("inf")
    )
    ranked_candidates = sorted(candidates, key=lambda value: value.score, reverse=True)
    candidate_score_margin = (
        float(ranked_candidates[0].score - ranked_candidates[1].score)
        if len(ranked_candidates) > 1
        else float("inf")
    )
    score_margin = min(cluster_score_margin, candidate_score_margin)
    ambiguous = np.isfinite(score_margin) and score_margin < ANGLE_AMBIGUITY_SCORE_MARGIN
    if ambiguous and legacy is not None:
        legacy_match = next(
            (item for item in candidates if item.angle == legacy[0]),
            None,
        )
        if legacy_match is not None:
            selected = legacy_match
            selected_by = "legacy_tie_break"
    evidence.update(
        {
            "status": "selected",
            "selected_angle": selected.angle,
            "selected_by": selected_by,
            "angle_ambiguous": bool(ambiguous),
            "score_margin": float(score_margin) if np.isfinite(score_margin) else None,
            "cluster_score_margin": (
                float(cluster_score_margin)
                if np.isfinite(cluster_score_margin)
                else None
            ),
            "candidate_score_margin": (
                float(candidate_score_margin)
                if np.isfinite(candidate_score_margin)
                else None
            ),
            "legacy_angle": legacy[0] if legacy is not None else None,
            "selected_cluster_id": selected.cluster_id,
            "clusters": clusters,
            "candidates": [
                {
                    "angle": item.angle,
                    "period_frames": item.period_frames,
                    "periodicity": item.periodicity,
                    "peak_prominence": item.peak_prominence,
                    "phase_consistency": item.phase_consistency,
                    "boundary_stability": item.boundary_stability,
                    "dynamic_range": item.prepared.dynamic_range,
                    "valid_fraction": item.prepared.valid_fraction,
                    "score": item.score,
                    "cluster_id": item.cluster_id,
                    "selected": item.angle == selected.angle,
                }
                for item in sorted(candidates, key=lambda value: value.score, reverse=True)
            ],
        }
    )
    return AngleSelection(
        selected.angle,
        selected.prepared,
        selected.periodicity,
        evidence,
    )


def interval_iou(first: Rep, second: Rep) -> float:
    intersection = min(first.end_frame, second.end_frame) - max(first.start_frame, second.start_frame) + 1
    if intersection <= 0:
        return 0.0
    union = (
        max(first.end_frame, second.end_frame)
        - min(first.start_frame, second.start_frame)
        + 1
    )
    return float(intersection / union)


def agreement(first: list[Rep], second: list[Rep]) -> float:
    if not first or not second:
        return 0.0
    matrix = np.asarray(
        [[-interval_iou(left, right) for right in second] for left in first],
        dtype=np.float64,
    )
    rows, columns = linear_sum_assignment(matrix)
    values = [interval_iou(first[row], second[column]) for row, column in zip(rows, columns)]
    return float(np.mean(values)) if values else 0.0


def result_summary(
    segment: ActionSegment,
    result: MethodResult,
    agreement_iou: float = 0.0,
    bout_count: int = 0,
    non_rep_count: int = 0,
) -> dict[str, Any]:
    widths = np.asarray([rep.end_frame - rep.start_frame + 1 for rep in result.reps])
    return {
        "video_id": segment.video_id,
        "action": segment.action,
        "action_display": segment.action_display,
        "window_start_frame": segment.start_frame,
        "window_end_frame": segment.end_frame,
        "method": result.method,
        "angle": result.angle,
        "num_reps": len(result.reps),
        "period_frames": result.period_frames,
        "period_s": result.period_frames / 30.0,
        "mean_rep_frames": float(np.mean(widths)) if len(widths) else 0.0,
        "std_rep_frames": float(np.std(widths)) if len(widths) else 0.0,
        "rep_duration_cv": float(np.std(widths) / np.mean(widths)) if len(widths) and np.mean(widths) else 0.0,
        "similarity": result.similarity,
        "valid_fraction": result.valid_fraction,
        "objective": result.objective,
        "agreement_iou": agreement_iou,
        "active_bouts": bout_count,
        "non_rep_intervals": non_rep_count,
        "rejected_reps": len(result.rejected_reps),
        "rejected_rep_reasons": ";".join(
            sorted({rejected.reason for rejected in result.rejected_reps})
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_segment(
    segment: ActionSegment,
    timestamps_ms: np.ndarray,
    prepared: PreparedSignal,
    aifit: MethodResult | None,
    peak: MethodResult | None,
    path: Path,
    active_bouts: list[ActiveBout] | None = None,
    non_rep_intervals: list[NonRepInterval] | None = None,
) -> None:
    results = [
        (result, color, label)
        for result, color, label in (
            (aifit, "#c43c4b", "AIFit"),
            (peak, "#1f6fb2", "peak prominence"),
        )
        if result is not None
    ]
    if not results:
        return
    start = segment.start_frame + prepared.source_start
    end = segment.start_frame + prepared.source_end + 1
    frames = np.arange(start, end)
    seconds = timestamps_ms[start:end] / 1000.0
    figure, axes = plt.subplots(len(results), 1, figsize=(13, 3.4 * len(results)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, (result, color, label) in zip(axes, results):
        axis.plot(frames, prepared.smooth, color="#333333", linewidth=1.1)
        for interval in non_rep_intervals or []:
            axis.axvspan(
                interval.start_frame,
                interval.end_frame,
                color="#9e9e9e",
                alpha=0.16,
                linewidth=0,
            )
        for bout in active_bouts or []:
            axis.axvline(bout.start_frame, color="#2e7d32", linewidth=1.0, linestyle=":")
            axis.axvline(bout.end_frame, color="#2e7d32", linewidth=1.0, linestyle=":")
        for rep in result.reps:
            axis.axvspan(rep.start_frame, rep.end_frame, color=color, alpha=0.10)
            axis.axvline(rep.start_frame, color=color, linewidth=0.9, linestyle="--")
        for rejected in result.rejected_reps:
            axis.axvspan(
                rejected.start_frame,
                rejected.end_frame,
                facecolor="#f4a261",
                edgecolor="#b5481c",
                hatch="//",
                alpha=0.16,
                linewidth=0.8,
            )
        axis.set_title(
            f"{label} | {segment.video_id} | {segment.action} | {result.angle} | "
            f"reps={len(result.reps)} | rejected={len(result.rejected_reps)}"
        )
        axis.set_ylabel("angle (deg)")
        axis.grid(alpha=0.20)
    axes[-1].set_xlabel(f"decoded frame index (playback seconds: {seconds[0]:.1f}-{seconds[-1]:.1f})")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> int:
    global SMOOTH_WINDOW, SMOOTH_POLYORDER
    args = parse_args()
    SMOOTH_WINDOW = args.smooth_window
    SMOOTH_POLYORDER = args.smooth_polyorder
    args.angles_dir = args.angles_dir.resolve()
    args.landmarks_dir = args.landmarks_dir.resolve()
    args.feedback_json = args.feedback_json.resolve()
    args.phase_config = args.phase_config.resolve()
    args.rep_metadata = args.rep_metadata.resolve()
    args.output_dir = args.output_dir.resolve()
    phase_guidance_by_action = (
        {}
        if args.no_phase_guidance
        else load_phase_config(args.phase_config)
    )
    rep_metadata_by_key = (
        {}
        if args.no_rep_metadata
        else load_rep_metadata(args.rep_metadata)
    )
    video_ids = load_video_ids(args.selection, args.videos)
    segments = load_feedback_segments(args.feedback_json, video_ids)
    allowed_actions = set(args.actions) if args.actions else set(ACTION_CANDIDATE_ANGLES)
    segments = [
        segment
        for segment in segments
        if segment.action in allowed_actions and segment.end_frame - segment.start_frame >= 20
    ]
    missing_rep_metadata = [
        f"{segment.video_id}/{segment.action}"
        for segment in segments
        if not args.no_rep_metadata
        and (segment.video_id.zfill(4), segment.action) not in rep_metadata_by_key
    ]
    if missing_rep_metadata:
        raise ValueError(
            "missing rep metadata for selected segments: "
            + ", ".join(missing_rep_metadata)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    if not args.no_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    rep_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    non_rep_rows: list[dict[str, Any]] = []
    rejected_rep_rows: list[dict[str, Any]] = []
    angle_evidence_rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "task": "task1-3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "angles_dir": str(args.angles_dir),
        "landmarks_dir": str(args.landmarks_dir),
        "feedback_json": str(args.feedback_json),
        "phase_config": None if args.no_phase_guidance else str(args.phase_config),
        "rep_metadata_config": None if args.no_rep_metadata else str(args.rep_metadata),
        "parameters": {
            "min_period_s": args.min_period_s,
            "max_period_s": args.max_period_s,
            "prominence_frac": args.prominence_frac,
            "stage_b": not args.no_stage_b,
            "ablation": args.ablation,
            "active_bouts": not args.no_active_bouts,
            "activity_window_s": args.activity_window_s,
            "activity_threshold": args.activity_threshold,
            "min_nonrep_s": args.min_nonrep_s,
            "activity_close_gap_s": args.activity_close_gap_s,
            "incomplete_tail_ratio": INCOMPLETE_TAIL_RATIO,
            "phase_guidance": not args.no_phase_guidance,
            "phase_snap": not args.no_phase_snap,
            "rep_metadata": not args.no_rep_metadata,
            "confidence_template": args.confidence_template,
            "confidence_auto_delete": False,
            "angle_scoring": args.angle_scoring,
            "p2a_tail_filter": args.p2a_tail_filter,
            "smooth_window": args.smooth_window,
            "smooth_polyorder": args.smooth_polyorder,
        },
        "segments": [],
        "failures": [],
    }
    segment_rows = []
    for segment in segments:
        metadata = rep_metadata_by_key.get((segment.video_id.zfill(4), segment.action))
        segment_rows.append(
            {
                **asdict(segment),
                **rep_metadata_fields(metadata),
            }
        )

    for segment in segments:
        wide_path = args.angles_dir / segment.video_id / "angles_wide.csv"
        if not wide_path.exists():
            payload["failures"].append(
                {"video_id": segment.video_id, "action": segment.action, "reason": "missing angles_wide.csv"}
            )
            continue
        timestamps_ms, wide = load_wide(wide_path)
        fps = 30.0
        rep_metadata = rep_metadata_by_key.get((segment.video_id.zfill(4), segment.action))
        metadata_fields = rep_metadata_fields(rep_metadata)
        phase_guidance = phase_guidance_by_action.get(segment.action)
        phase_signal_full = None
        phase_signal_error = None
        if phase_guidance is not None:
            phase_signal_full = resolve_phase_signal(
                phase_guidance,
                wide,
                args.landmarks_dir,
                segment.video_id,
            )
            if phase_signal_full is None:
                phase_signal_error = "phase_signal_unavailable"
        selection = choose_auto_angle(
            wide,
            segment,
            fps,
            args.min_period_s,
            args.max_period_s,
            scoring=args.angle_scoring,
        )
        if selection is None:
            payload["failures"].append(
                {"video_id": segment.video_id, "action": segment.action, "reason": "no usable angle signal"}
            )
            continue
        angle = selection.angle
        prepared = selection.prepared
        effective_offset = segment.start_frame + prepared.source_start
        period_result = estimate_period(
            prepared.normalized,
            fps,
            args.min_period_s,
            args.max_period_s,
        )
        global_period_frames = float(period_result[0]) if period_result is not None else 0.0
        active_bouts: list[ActiveBout] = []
        non_rep_intervals: list[NonRepInterval] = []
        activity_evidence: dict[str, Any] | None = None
        trace: ActivityTrace | None = None
        low_intervals: list[tuple[int, int]] = []
        segment_values = wide[angle][segment.start_frame : segment.end_frame + 1]
        if args.no_active_bouts:
            phase_values = (
                phase_signal_full[prepared.source_start : prepared.source_end + 1]
                if phase_signal_full is not None
                else None
            )
            aifit = run_aifit(
                prepared.normalized,
                prepared.valid,
                angle,
                effective_offset,
                fps,
                args.min_period_s,
                args.max_period_s,
                not args.no_stage_b,
                phase_signal=phase_values,
                phase_guidance=phase_guidance if phase_values is not None else None,
                use_phase_snap=not args.no_phase_snap,
                shape_template_method=args.confidence_template,
            )
        else:
            trace = build_activity_trace(
                wide,
                segment,
                angle,
                fps,
                args.activity_window_s,
                args.activity_threshold,
                args.activity_close_gap_s,
            )
            if trace is None:
                activity_evidence = {"status": "unavailable", "fallback": "global AIFit"}
                phase_values = (
                    phase_signal_full[prepared.source_start : prepared.source_end + 1]
                    if phase_signal_full is not None
                    else None
                )
                aifit = run_aifit(
                    prepared.normalized,
                    prepared.valid,
                    angle,
                    effective_offset,
                    fps,
                    args.min_period_s,
                    args.max_period_s,
                    not args.no_stage_b,
                    phase_signal=phase_values,
                    phase_guidance=phase_guidance if phase_values is not None else None,
                    use_phase_snap=not args.no_phase_snap,
                    shape_template_method=args.confidence_template,
                )
            else:
                low_intervals = significant_low_intervals(
                    trace,
                    global_period_frames,
                    fps,
                    args.min_nonrep_s,
                )
                aifit, active_bouts, non_rep_intervals = run_aifit_with_active_bouts(
                    segment_values,
                    prepared,
                    trace,
                    low_intervals,
                    segment,
                    angle,
                    fps,
                    args.min_period_s,
                    args.max_period_s,
                    not args.no_stage_b,
                    phase_values=(
                        phase_signal_full[segment.start_frame : segment.end_frame + 1]
                        if phase_signal_full is not None
                        else None
                    ),
                    phase_guidance=phase_guidance,
                    use_phase_snap=not args.no_phase_snap,
                    shape_template_method=args.confidence_template,
                )
                activity_evidence = dict(trace.evidence)
                activity_evidence.update(
                    {
                        "global_period_frames": global_period_frames,
                        "significant_low_intervals": [
                            {
                                "start_frame": segment.start_frame + start,
                                "end_frame": segment.start_frame + end,
                                "frames": end - start + 1,
                            }
                            for start, end in low_intervals
                        ],
                    }
                )
        if aifit is not None:
            aifit.reps = enrich_rep_confidences(
                aifit.reps,
                signal=prepared.normalized,
                valid=prepared.valid,
                offset=effective_offset,
                template_method=args.confidence_template,
                trace=trace,
                segment_start_frame=segment.start_frame,
            )
            tail_fit_end = (
                max(bout.fit_end_frame for bout in active_bouts)
                if active_bouts
                else segment.end_frame
            )
            tail_rejected = (
                args.p2a_tail_filter == "guarded"
                and reject_air_jump_rope_transition_tail(
                    aifit,
                    segment,
                    tail_fit_end,
                )
            )
            if tail_rejected and not args.no_active_bouts:
                rejected_tail = aifit.rejected_reps[-1]
                interval_start = rejected_tail.start_frame
                interval_end = max(rejected_tail.end_frame, tail_fit_end)
                covered = any(
                    interval.start_frame <= interval_start
                    and interval.end_frame >= interval_end
                    for interval in non_rep_intervals
                )
                if not covered:
                    non_rep_intervals.append(
                        NonRepInterval(
                            start_frame=interval_start,
                            end_frame=interval_end,
                            reason="transition",
                            evidence={"source": "p2a_tail_filter"},
                        )
                    )
                    non_rep_intervals.sort(key=lambda item: item.start_frame)
        peak = run_peak_baseline(
            prepared.detrended,
            prepared.valid,
            angle,
            effective_offset,
            fps,
            args.min_period_s,
            args.max_period_s,
            args.prominence_frac,
        )
        if peak is not None:
            peak.reps = enrich_rep_confidences(
                peak.reps,
                signal=prepared.detrended,
                valid=prepared.valid,
                offset=effective_offset,
                template_method=args.confidence_template,
                trace=trace,
                segment_start_frame=segment.start_frame,
            )
        if aifit is None or peak is None:
            payload["failures"].append(
                {
                    "video_id": segment.video_id,
                    "action": segment.action,
                    "reason": "AIFit or peak baseline produced fewer than two reps",
                    "aifit_reps": len(aifit.reps) if aifit else 0,
                    "peak_reps": len(peak.reps) if peak else 0,
                }
            )
            if aifit is None and peak is None and not active_bouts and not non_rep_intervals:
                continue
        if aifit is not None and peak is not None:
            aifit_peak_iou = agreement(aifit.reps, peak.reps)
            summary_rows.append(
                {
                    **result_summary(
                        segment,
                        aifit,
                        aifit_peak_iou,
                        bout_count=len(active_bouts),
                        non_rep_count=len(non_rep_intervals),
                    ),
                    **metadata_fields,
                }
            )
            summary_rows.append(
                {
                    **result_summary(segment, peak, aifit_peak_iou),
                    **metadata_fields,
                }
            )
        elif aifit is not None:
            summary_rows.append(
                {
                    **result_summary(
                        segment,
                        aifit,
                        bout_count=len(active_bouts),
                        non_rep_count=len(non_rep_intervals),
                    ),
                    **metadata_fields,
                }
            )
        elif peak is not None:
            summary_rows.append(
                {
                    **result_summary(segment, peak),
                    **metadata_fields,
                }
            )

        segment_payload = {
            "video_id": segment.video_id,
            "action": segment.action,
            "action_display": segment.action_display,
            "window": {
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
            },
            "effective_window": {
                "start_frame": effective_offset,
                "end_frame": segment.start_frame + prepared.source_end,
            },
            "active_bouts": [asdict(bout) for bout in active_bouts],
            "non_rep_intervals": [asdict(interval) for interval in non_rep_intervals],
            "activity_evidence": activity_evidence,
            "phase_signal_error": phase_signal_error,
            "rep_metadata": rep_metadata,
            "angle_selection": selection.evidence,
            "angles": {},
        }
        cluster_by_id = {
            int(cluster["cluster_id"]): cluster
            for cluster in selection.evidence.get("clusters", [])
        }
        for candidate in selection.evidence.get("candidates", []):
            cluster = cluster_by_id.get(int(candidate["cluster_id"]), {})
            angle_evidence_rows.append(
                {
                    "video_id": segment.video_id,
                    "action": segment.action,
                    "scoring": args.angle_scoring,
                    "selected_angle": selection.angle,
                    "selected_by": selection.evidence.get("selected_by"),
                    "angle_ambiguous": selection.evidence.get("angle_ambiguous"),
                    "ambiguity_margin": selection.evidence.get("score_margin"),
                    "angle": candidate["angle"],
                    "period_frames": candidate["period_frames"],
                    "periodicity": candidate["periodicity"],
                    "cluster_period_consistency": cluster.get("period_consistency"),
                    "phase_consistency": candidate["phase_consistency"],
                    "peak_prominence": candidate["peak_prominence"],
                    "boundary_stability": candidate["boundary_stability"],
                    "dynamic_range": candidate["dynamic_range"],
                    "valid_fraction": candidate["valid_fraction"],
                    "score": candidate["score"],
                    "cluster_id": candidate["cluster_id"],
                    "selected": candidate["selected"],
                }
            )
        for result in (aifit, peak):
            if result is None:
                continue
            segment_payload["angles"][result.method] = {
                "angle": result.angle,
                "num_reps": len(result.reps),
                "similarity": result.similarity,
                "period_frames": result.period_frames,
                "phase_guidance": result.phase_guidance,
                "reps": [asdict(rep) for rep in result.reps],
                "rejected_reps": [asdict(rejected) for rejected in result.rejected_reps],
            }
            for rep in result.reps:
                rep_rows.append(
                    {
                        "video_id": segment.video_id,
                        "action": segment.action,
                        "method": result.method,
                        "angle": result.angle,
                        **asdict(rep),
                        **metadata_fields,
                    }
                )
            for rejected in result.rejected_reps:
                rejected_rep_rows.append(
                    {
                        "video_id": segment.video_id,
                        "action": segment.action,
                        "method": result.method,
                        "angle": result.angle,
                        **asdict(rejected),
                        **metadata_fields,
                    }
                )

        if not args.no_active_bouts:
            for interval in non_rep_intervals:
                non_rep_rows.append(
                    {
                        "video_id": segment.video_id,
                        "action": segment.action,
                        "method": "aifit",
                        "angle": angle,
                        **asdict(interval),
                    }
                )

        if args.ablation:
            fixed_angle = FIXED_ANGLES.get(segment.action)
            if fixed_angle in wide:
                fixed_prepared = prepare_signal(
                    wide[fixed_angle][segment.start_frame : segment.end_frame + 1]
                )
                fixed_result = None
                fixed_gated = False
                if fixed_prepared is not None:
                    fixed_offset = segment.start_frame + fixed_prepared.source_start
                    if not args.no_active_bouts and activity_evidence is not None:
                        fixed_gated = True
                        fixed_values = wide[fixed_angle][
                            segment.start_frame : segment.end_frame + 1
                        ]
                        fixed_trace = build_activity_trace(
                            wide,
                            segment,
                            fixed_angle,
                            fps,
                            args.activity_window_s,
                            args.activity_threshold,
                            args.activity_close_gap_s,
                        )
                        if fixed_trace is not None:
                            fixed_low_intervals = significant_low_intervals(
                                fixed_trace,
                                global_period_frames,
                                fps,
                                args.min_nonrep_s,
                            )
                            fixed_result, _, _ = run_aifit_with_active_bouts(
                                fixed_values,
                                fixed_prepared,
                                fixed_trace,
                                fixed_low_intervals,
                                segment,
                                fixed_angle,
                                fps,
                                args.min_period_s,
                                args.max_period_s,
                                not args.no_stage_b,
                                shape_template_method=args.confidence_template,
                            )
                    if fixed_result is None and not fixed_gated:
                        fixed_result = run_aifit(
                            fixed_prepared.normalized,
                            fixed_prepared.valid,
                            fixed_angle,
                            fixed_offset,
                            fps,
                            args.min_period_s,
                            args.max_period_s,
                            not args.no_stage_b,
                            shape_template_method=args.confidence_template,
                        )
                if fixed_result is not None:
                    fixed_iou = agreement(fixed_result.reps, aifit.reps) if aifit is not None else 0.0
                    segment_payload["angles"]["aifit_fixed"] = {
                        "angle": fixed_angle,
                        "num_reps": len(fixed_result.reps),
                        "similarity": fixed_result.similarity,
                        "period_frames": fixed_result.period_frames,
                        "agreement_with_auto": fixed_iou,
                    }
                    ablation_rows.append(
                        {
                            "video_id": segment.video_id,
                            "action": segment.action,
                            "auto_angle": angle,
                            "auto_reps": len(aifit.reps) if aifit is not None else 0,
                            "auto_similarity": aifit.similarity if aifit is not None else 0.0,
                            "fixed_angle": fixed_angle,
                            "fixed_reps": len(fixed_result.reps),
                            "fixed_similarity": fixed_result.similarity,
                            "agreement_iou": fixed_iou,
                        }
                    )

        payload["segments"].append(segment_payload)
        if not args.no_plots and (aifit is not None or peak is not None):
            plot_path = plots_dir / f"{segment.video_id}_{segment.action}_segmentation.png"
            plot_segment(
                segment,
                timestamps_ms,
                prepared,
                aifit,
                peak,
                plot_path,
                active_bouts,
                non_rep_intervals,
            )
        print(
            f"[{segment.video_id}/{segment.action}] angle={angle} "
            f"aifit_reps={len(aifit.reps) if aifit else 0} "
            f"peak_reps={len(peak.reps) if peak else 0}"
        )

    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "reps.csv", rep_rows)
    write_csv(args.output_dir / "action_segments.csv", segment_rows)
    write_csv(args.output_dir / "non_rep_intervals.csv", non_rep_rows)
    write_csv(args.output_dir / "rejected_reps.csv", rejected_rep_rows)
    if angle_evidence_rows:
        write_csv(args.output_dir / "angle_evidence.csv", angle_evidence_rows)
    if args.ablation:
        write_csv(args.output_dir / "ablation.csv", ablation_rows)
    payload["counts"] = {
        "segments": len(segments),
        "processed": len(payload["segments"]),
        "failures": len(payload["failures"]),
        "summary_rows": len(summary_rows),
        "rep_rows": len(rep_rows),
        "non_rep_rows": len(non_rep_rows),
        "rejected_rep_rows": len(rejected_rep_rows),
        "angle_evidence_rows": len(angle_evidence_rows),
    }
    (args.output_dir / "run.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
