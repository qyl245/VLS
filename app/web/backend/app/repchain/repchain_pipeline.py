#!/usr/bin/env python3
"""Run RepChain v3 for uploaded clips and adapt it to the legacy report contract."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import repchain_chain_selection as chain_selection
import repchain_phase_states as phase_states
import repchain_quality as quality_gates
import analysis_builder as analysis_v3
import rep_contract


DEFAULT_SPECS = ROOT / "configs" / "exercise_specs.json"
DEFAULT_SEGMENTER = ROOT / "configs" / "repchain_segmenter.json"
DEFAULT_CHAIN = ROOT / "configs" / "repchain_chain.json"

REP_FIELDS = [
    "rep_id",
    "start_frame",
    "end_frame",
    "top_frame",
    "bottom_frame",
    "start_time_s",
    "end_time_s",
    "bottom_time_s",
    "clip_start_time_s",
    "clip_end_time_s",
    "clip_bottom_time_s",
    "duration_s",
    "eccentric_start_frame",
    "eccentric_end_frame",
    "concentric_start_frame",
    "concentric_end_frame",
    "phase_angle",
    "top_angle_deg",
    "start_angle_deg",
    "end_angle_deg",
    "bottom_angle_deg",
    "confidence",
    "visibility",
    "shape_similarity",
    "phase_consistency",
    "motion_evidence",
    "review_score",
    "review_priority",
    "reject_reason",
    "width_ratio",
    "start_phase_level",
    "end_phase_level",
    "quality_flags",
    "status",
    "reject_reasons",
    "provenance.selection",
    "provenance.top_boundary_source",
    "provenance.bottom_boundary_source",
]
PHASE_FIELDS = [
    "rep_id",
    "phase",
    "frame_index",
    "clip_frame_index",
    "time_s",
    "clip_time_s",
    "angle_name",
    "angle_deg",
    "confidence",
]
REJECTED_FIELDS = [
    "candidate_rep_id",
    "start_frame",
    "end_frame",
    "start_time_s",
    "end_time_s",
    "quality_flags",
    "reject_reasons",
    "evidence",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    for row in rows:
        flattened = dict(row)
        provenance = row.get("provenance")
        if isinstance(provenance, dict):
            flattened.update(
                {
                    f"provenance.{key}": value
                    for key, value in provenance.items()
                }
            )
        csv_rows.append(flattened)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)


def source_time_s(frame: int, fps: float) -> float:
    return round(frame / fps, 6) if fps > 0 else 0.0


def signal_value(signal: np.ndarray, frame: int) -> float | None:
    local = max(0, min(len(signal) - 1, int(frame)))
    value = float(signal[local])
    return round(value, 6) if np.isfinite(value) else None


def phase_level(value: float | None, signal: np.ndarray) -> float | None:
    if value is None:
        return None
    finite = signal[np.isfinite(signal)]
    if not finite.size:
        return None
    low, high = np.percentile(finite, [10, 90])
    if high - low < 1e-8:
        return None
    return round(float(np.clip((value - low) / (high - low), 0.0, 1.0)), 6)


def _clip_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    clips = manifest.get("clips") if isinstance(manifest, dict) else manifest
    if not isinstance(clips, list):
        raise ValueError("manifest has no clip list")
    return {str(clip["clip_id"]): clip for clip in clips}


def _events_by_set(analysis: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    events: dict[int, dict[int, dict[str, Any]]] = {}
    for event in (analysis.get("diagnostics") or {}).get("selected_top_events") or []:
        frame = event.get("frame")
        if frame is None:
            continue
        for set_id in event.get("set_ids") or []:
            events.setdefault(int(set_id), {})[int(frame)] = event
    return {
        set_id: [frames[frame] for frame in sorted(frames)]
        for set_id, frames in events.items()
    }


def _rep_by_top_anchor(analysis: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    output = {}
    for set_candidate in analysis.get("sets") or []:
        set_id = int(set_candidate.get("set_id") or 0)
        for rep in set_candidate.get("reps") or []:
            anchor = (rep.get("top") or {}).get("anchor_frame")
            if anchor is not None:
                output[(set_id, int(anchor))] = rep
    return output


def _choose_bottom_frame(
    rep: dict[str, Any],
    signal: np.ndarray,
    start_local: int,
    end_local: int,
) -> int:
    anchor = (rep.get("bottom") or {}).get("anchor_frame")
    if anchor is not None and start_local <= int(anchor) <= end_local:
        return int(anchor)
    window = signal[start_local : end_local + 1]
    finite = np.flatnonzero(np.isfinite(window))
    if finite.size:
        return start_local + int(finite[int(np.argmin(window[finite]))])
    return (start_local + end_local) // 2


def _rep_bottoms_align(
    primary_reps: list[dict[str, Any]],
    candidate_reps: list[dict[str, Any]],
    *,
    fps: float,
) -> bool:
    if len(primary_reps) != len(candidate_reps):
        return False
    tolerance = max(4.0, 0.5 * max(float(fps), 1.0))
    return all(
        abs(float(left["bottom_frame"]) - float(right["bottom_frame"]))
        <= tolerance
        for left, right in zip(primary_reps, candidate_reps)
    )


def _quality_flags(rep: dict[str, Any]) -> list[str]:
    flags = [str(value) for value in rep.get("quality_flags") or [] if str(value)]
    status = str(rep.get("status") or "unknown")
    if status != "complete" and status not in flags:
        flags.append(status)
    return flags


def adapt_cycles(
    analysis: dict[str, Any],
    signal: np.ndarray,
    *,
    fps: float,
    trim_start: int,
    phase_signal_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selected_events = _events_by_set(analysis)
    reps_by_top = _rep_by_top_anchor(analysis)
    cycles: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    cycles_with_status: list[dict[str, Any]] = []

    for set_candidate in sorted(analysis.get("sets") or [], key=lambda item: int(item.get("set_id") or 0)):
        set_id = int(set_candidate.get("set_id") or 0)
        events = selected_events.get(set_id) or []
        for left_event, right_event in zip(events[:-1], events[1:]):
            left = int(left_event["frame"])
            right = int(right_event["frame"])
            if right <= left + 1:
                continue
            rep = reps_by_top.get((set_id, int(left)))
            if rep is None:
                continue
            end_local = int(right) - 1
            if right_event.get("is_terminal"):
                rep_end = rep.get("end_frame")
                if rep_end is not None and len(signal):
                    end_local = max(
                        end_local,
                        min(int(rep_end), len(signal) - 1),
                    )
            bottom_local = _choose_bottom_frame(rep, signal, int(left), end_local)
            status = str(rep.get("status") or "unknown")
            item = {
                "set_id": set_id,
                "cycle_status": status,
                "start_local": int(left),
                "end_local": end_local,
                "bottom_local": bottom_local,
                "repchain_rep": rep,
            }
            cycles_with_status.append(item)
            if status not in rep_contract.ACCEPTED_CYCLE_STATUSES:
                continue
            # Endpoint partials are real cycles whose closing top lies beyond the
            # selected set boundary. Keep them reportable, but preserve the
            # RepChain status in quality_flags/status instead of silently
            # treating them as complete.
            cycles.append(item)

    for index, item in enumerate(analysis.get("rejected_candidates") or [], start=1):
        start_frame = trim_start + int(item.get("start_frame") or 0)
        end_frame = trim_start + int(item.get("end_frame") or 0)
        rejected.append(
            {
                "candidate_rep_id": str(item.get("candidate_id") or index),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_s": source_time_s(start_frame, fps),
                "end_time_s": source_time_s(end_frame, fps),
                "quality_flags": ";".join(str(value) for value in item.get("reasons") or []),
                "reject_reasons": ";".join(str(value) for value in item.get("reasons") or []),
                "evidence": {"source": "repchain_rejected_candidate"},
            }
        )

    if not cycles:
        return [], [], cycles_with_status, rejected

    widths = np.asarray(
        [item["end_local"] - item["start_local"] + 1 for item in cycles],
        dtype=np.float64,
    )
    reference_width = float(np.median(widths)) if widths.size else 1.0
    rows = []
    phases = []
    for rep_id, item in enumerate(cycles, start=1):
        repchain_rep = item["repchain_rep"]
        start_local = int(item["start_local"])
        end_local = int(item["end_local"])
        bottom_local = int(item["bottom_local"])
        start_frame = trim_start + start_local
        end_frame = trim_start + end_local
        bottom_frame = trim_start + bottom_local
        top_value = signal_value(signal, start_local)
        bottom_value = signal_value(signal, bottom_local)
        end_value = signal_value(signal, end_local)
        confidence_payload = repchain_rep.get("confidence") or {}
        # `combined` is the clip-level calibrated score and may be intentionally
        # low when the clip abstains. Per-rep confidence should stay rep-local.
        confidence = float(confidence_payload.get("complete") or 0.0)
        in_set_confidence = float(confidence_payload.get("in_set") or confidence)
        cycle_status = str(item.get("cycle_status") or "unknown")
        width_ratio = round(
            (end_local - start_local + 1) / reference_width,
            6,
        ) if reference_width else None
        row = {
            "rep_id": rep_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "top_frame": start_frame,
            "bottom_frame": bottom_frame,
            "start_time_s": source_time_s(start_frame, fps),
            "end_time_s": source_time_s(end_frame, fps),
            "bottom_time_s": source_time_s(bottom_frame, fps),
            "clip_start_time_s": source_time_s(start_local, fps),
            "clip_end_time_s": source_time_s(end_local, fps),
            "clip_bottom_time_s": source_time_s(bottom_local, fps),
            "duration_s": round((end_local - start_local + 1) / fps, 6) if fps > 0 else 0.0,
            "eccentric_start_frame": start_frame,
            "eccentric_end_frame": bottom_frame,
            "concentric_start_frame": bottom_frame,
            "concentric_end_frame": end_frame,
            "phase_angle": phase_signal_name,
            "top_angle_deg": top_value,
            "start_angle_deg": top_value,
            "end_angle_deg": end_value,
            "bottom_angle_deg": bottom_value,
            "confidence": round(confidence, 6),
            "visibility": 1.0,
            "shape_similarity": round(float(confidence_payload.get("complete") or 0.0), 6),
            "phase_consistency": round(in_set_confidence, 6),
            "motion_evidence": 0.0,
            "review_score": round(confidence, 6),
            "review_priority": "high" if confidence < 0.35 else "low",
            "reject_reason": "",
            "width_ratio": width_ratio,
            "start_phase_level": phase_level(top_value, signal),
            "end_phase_level": phase_level(end_value, signal),
            "quality_flags": ";".join(_quality_flags(repchain_rep)),
            "status": cycle_status,
            "reject_reasons": "",
        }
        rows.append(row)
        for phase, local_frame, value in (
            ("top_start", start_local, top_value),
            ("bottom", bottom_local, bottom_value),
            ("top_end", end_local, end_value),
        ):
            frame = trim_start + local_frame
            phases.append(
                {
                    "rep_id": rep_id,
                    "phase": phase,
                    "frame_index": frame,
                    "clip_frame_index": local_frame,
                    "time_s": source_time_s(frame, fps),
                    "clip_time_s": source_time_s(local_frame, fps),
                    "angle_name": phase_signal_name,
                    "angle_deg": value,
                    "confidence": round(confidence, 6),
                }
            )
    return rows, phases, cycles_with_status, rejected


def _smooth_signal(signal: np.ndarray, fps: float) -> np.ndarray:
    """Smooth a phase signal while preserving invalid samples."""
    valid = np.isfinite(signal)
    if not valid.any():
        return signal.copy()
    window = max(1, int(round(0.12 * fps)))
    if window <= 1:
        return signal.copy()
    indices = np.arange(len(signal), dtype=np.float64)
    values = np.interp(indices, indices[valid], signal[valid])
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.convolve(values, kernel, mode="same")
    smoothed[~valid] = np.nan
    return smoothed


def _phase_peak_between(signal: np.ndarray, start: int, end: int) -> int | None:
    start = max(0, int(start))
    end = min(len(signal) - 1, int(end))
    if end < start:
        return None
    window = signal[start : end + 1]
    finite = np.flatnonzero(np.isfinite(window))
    if not finite.size:
        return None
    return start + int(finite[int(np.argmax(window[finite]))])


def _phase_top_anchor_between(
    signal: np.ndarray,
    start: int,
    end: int,
    *,
    fps: float,
) -> int | None:
    """Anchor a top just before the high-phase standing plateau ends.

    The exact peak can occur early in a long standing plateau. Following the
    near-peak band to its exit and stepping back 150 ms tracks the onset of the
    descent while remaining robust to one-frame pose noise.
    """
    peak = _phase_peak_between(signal, start, end)
    if peak is None:
        return None
    start = max(0, int(start))
    end = min(len(signal) - 1, int(end))
    finite = signal[np.isfinite(signal)]
    if not finite.size:
        return peak
    low, high = np.percentile(finite, [10, 90])
    tolerance = max(1.0, 0.04 * float(high - low))
    threshold = float(signal[peak]) - tolerance
    plateau_exit = peak
    while (
        plateau_exit < end
        and np.isfinite(signal[plateau_exit + 1])
        and float(signal[plateau_exit + 1]) >= threshold
    ):
        plateau_exit += 1
    lead = max(1, int(round(0.15 * max(float(fps), 0.0))))
    return max(start, plateau_exit - lead)


def build_legacy_recovery_analysis(
    clip_id: str,
    legacy: dict[str, Any],
    signal: np.ndarray,
    *,
    fps: float,
    trim_start: int,
    phase_signal_name: str,
    minimum_reps: int = 3,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Recover complete legacy candidates and rebuild top anchors from phase peaks.

    Legacy bottoms are used as anchors, while standing boundaries are rebuilt
    from the shared phase signal instead of copying the legacy top boundaries.
    """
    legacy_reps = [
        rep
        for rep in ((legacy.get("segmentation") or {}).get("reps") or [])
        if str(rep.get("status") or "complete") == "complete"
    ]
    legacy_reps.sort(key=lambda rep: int(rep["bottom_frame"]))
    evidence: dict[str, Any] = {
        "status": "not_eligible",
        "legacy_complete_rep_count": len(legacy_reps),
        "minimum_required_reps": minimum_reps,
    }
    if len(legacy_reps) < minimum_reps:
        evidence["reason"] = "insufficient_legacy_complete_reps"
        return None, evidence
    if not np.isfinite(signal).any():
        evidence["reason"] = "phase_signal_unavailable"
        return None, evidence

    bottoms = [int(rep["bottom_frame"]) - trim_start for rep in legacy_reps]
    if any(frame < 0 or frame >= len(signal) for frame in bottoms):
        evidence["reason"] = "legacy_bottom_outside_signal"
        return None, evidence
    if any(right <= left for left, right in zip(bottoms, bottoms[1:])):
        evidence["reason"] = "legacy_bottoms_not_monotonic"
        return None, evidence

    smoothed = _smooth_signal(signal, fps)
    intervals = [
        (0, bottoms[0]),
        *zip(bottoms[:-1], bottoms[1:]),
        (bottoms[-1], len(signal) - 1),
    ]
    tops: list[int | None] = []
    for index, (start, end) in enumerate(intervals):
        start_pad = 1 if index == 0 else 1
        end_pad = 1 if index == len(intervals) - 1 else 1
        window_start = start + start_pad
        window_end = end - end_pad
        peak = _phase_peak_between(smoothed, window_start, window_end)
        plateau_anchor = _phase_top_anchor_between(
            smoothed,
            window_start,
            window_end,
            fps=fps,
        )
        if peak is None:
            tops.append(None)
            continue
        relative_peak = (peak - window_start) / max(1, window_end - window_start)
        # Interior peaks near the start of a wide plateau are the main failure
        # mode. Use the plateau-exit anchor only there; ordinary extrema already
        # align with descent onset. The terminal interval keeps the plateau
        # anchor because the closing top may lie beyond the last observed peak.
        use_plateau_anchor = index == len(intervals) - 1 or (
            index > 0 and relative_peak < 0.40
        )
        tops.append(plateau_anchor if use_plateau_anchor else peak)
    if any(frame is None for frame in tops):
        evidence["reason"] = "phase_peak_missing"
        return None, evidence
    top_frames = [int(frame) for frame in tops if frame is not None]
    if any(right <= left for left, right in zip(top_frames, top_frames[1:])):
        evidence["reason"] = "phase_peaks_not_monotonic"
        return None, evidence
    if any(not (top_frames[index] < bottoms[index] < top_frames[index + 1]) for index in range(len(bottoms))):
        evidence["reason"] = "bottom_not_inside_adjacent_tops"
        return None, evidence

    recovery_reps: list[dict[str, Any]] = []
    for index, (source, bottom_frame) in enumerate(zip(legacy_reps, bottoms), start=1):
        confidence = float(source.get("confidence") or 0.5)
        confidence = float(np.clip(confidence, 0.0, 1.0))
        recovery_reps.append(
            {
                "rep_id": index,
                "start_frame": top_frames[index - 1],
                "end_frame": top_frames[index],
                "top": {
                    "start_frame": top_frames[index - 1],
                    "end_frame": top_frames[index - 1],
                    "anchor_frame": top_frames[index - 1],
                    "stable_fraction": 1.0,
                    "boundary_slope": 0.0,
                    "mean_angle": float(signal[top_frames[index - 1]]),
                    "evidence": {"source": "legacy_recovery_phase_peak"},
                },
                "bottom": {
                    "start_frame": bottom_frame,
                    "end_frame": bottom_frame,
                    "anchor_frame": bottom_frame,
                    "stable_fraction": 1.0,
                    "boundary_slope": 0.0,
                    "mean_angle": float(signal[bottom_frame]),
                    "evidence": {"source": "legacy_recovery_candidate"},
                },
                "phase_signal": phase_signal_name,
                "rep_unit": "cycle",
                "status": "complete",
                "partial_side": None,
                "confidence": {
                    "complete": round(confidence, 6),
                    "in_set": round(confidence, 6),
                    "combined": round(confidence, 6),
                },
                "quality_flags": ["legacy_recovery", "selected_chain"],
            }
        )

    period_frames = float(np.median(np.diff(top_frames))) if len(top_frames) > 1 else 0.0
    selected_events = [
        {
            "frame": frame,
            "set_ids": [1],
            "is_terminal": index == len(top_frames) - 1,
        }
        for index, frame in enumerate(top_frames)
    ]
    chain = {
        "schema_version": "vls-clip-analysis-v2",
        "clip_id": clip_id,
        "algorithm": {"name": "repchain", "version": "v3"},
        "anchor_policy": "plateau_midpoint",
        "status": "ok",
        "motion_status": "ok",
        "sets": [
            {
                "set_id": 1,
                "start_frame": top_frames[0],
                "end_frame": top_frames[-1],
                "status": "complete_set",
                "chain_confidence": round(
                    float(np.mean([rep["confidence"]["combined"] for rep in recovery_reps])),
                    6,
                ),
                "reps": recovery_reps,
                "rejected_candidates": [],
                "non_rep_intervals": [],
                "consensus_event_count": len(top_frames),
                "endpoint_closure_count": 0,
            }
        ],
        "rejected_candidates": [],
        "non_rep_intervals": [],
        "diagnostics": {
            "selected_phase_signal": phase_signal_name,
            "phase_signal_candidates": [],
            "chain_source": "legacy_candidate_recovery",
            "selected_top_events": selected_events,
            "period_frames": period_frames,
            "periodicity": None,
        },
    }
    evidence.update(
        {
            "status": "eligible",
            "reason": "repchain_abstained_with_recoverable_legacy_candidates",
            "legacy_bottom_frames": bottoms,
            "reconstructed_top_frames": top_frames,
            "period_frames": period_frames,
        }
    )
    return chain, evidence


def _copy_legacy_tree(source: Path, target: Path) -> None:
    if source.resolve() == target.resolve():
        return
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def analyze_clip(
    clip: dict[str, Any],
    *,
    legacy_dir: Path,
    output_dir: Path,
    repchain_dir: Path,
    specs: dict[str, Any],
    segmenter_config: dict[str, Any],
    chain_config: dict[str, Any],
) -> dict[str, Any]:
    clip_id = str(clip["clip_id"])
    legacy_clip_dir = legacy_dir / clip_id
    legacy = load_json(legacy_clip_dir / "analysis.json")
    signals = phase_states.load_angle_signals(legacy_clip_dir / "angles" / "angles_wide.csv")
    action_name = str(clip.get("action") or clip.get("exercise") or "").lower()
    action = phase_states.action_config(action_name, specs)
    # Phase resolution evaluates a private copy that includes derived bilateral
    # signals. Materialize the same candidates here so boundary reconstruction
    # uses the exact series selected by the chain.
    phase_states.add_bilateral_signal_candidates(signals, action)
    prior_signal = (legacy.get("segmentation") or {}).get("phase_angle")
    gate = quality_gates.from_legacy_quality(legacy)

    chain_analysis, _ = chain_selection.analyze_clip_signals(
        {**clip, "action": action_name},
        signals,
        action,
        prior_signal=prior_signal,
        phase_policy=segmenter_config["core"]["phase_signal"]["policy"],
        chain_config=chain_config,
        legacy_analysis=legacy,
        allow_legacy_fallback=False,
        quality_gate=gate,
        detector=segmenter_config["core"].get("detector"),
    )
    v3 = analysis_v3.build_v3_analysis(
        chain_analysis,
        {**clip, "action": action_name},
        gate=gate,
    )
    analysis_v3.validate_v3_analysis(v3)
    primary_v3 = copy.deepcopy(v3)
    selection_mode = "none"
    selection_reason = "no_accepted_cycles"
    recovery_evidence: dict[str, Any] = {
        "status": "not_needed",
        "reason": "repchain_primary_available",
    }

    repchain_dir.mkdir(parents=True, exist_ok=True)
    write_json(repchain_dir / f"{clip_id}_chain.json", chain_analysis)
    write_json(repchain_dir / f"{clip_id}_gate.json", gate)

    phase_signal_name = str(
        (v3.get("diagnostics") or {}).get("selected_phase_signal")
        or (chain_analysis.get("diagnostics") or {}).get("selected_phase_signal")
        or action["phase_signal_pairs"][0]["name"]
    )
    if phase_signal_name in signals:
        phase_signal = signals[phase_signal_name]
    else:
        phase_signal = np.full(int(v3.get("timebase", {}).get("frame_count") or 0), np.nan)
    fps = float(legacy["fps"])
    trim_start = int(legacy["trim"]["start_frame"])
    reps, phases, _, rejected = adapt_cycles(
        v3,
        phase_signal,
        fps=fps,
        trim_start=trim_start,
        phase_signal_name=phase_signal_name,
    )
    recovery_chain, recovery_evidence = build_legacy_recovery_analysis(
        clip_id,
        legacy,
        phase_signal,
        fps=fps,
        trim_start=trim_start,
        phase_signal_name=phase_signal_name,
    )
    action_status = (v3.get("action_verification") or {}).get("status")
    if recovery_chain is not None and action_status not in {"mismatch", "unsupported"}:
        recovery_v3 = analysis_v3.build_v3_analysis(
            recovery_chain,
            {**clip, "action": action_name},
            gate=gate,
        )
        analysis_v3.validate_v3_analysis(recovery_v3)
        recovery_reps, recovery_phases, _, recovery_rejected = adapt_cycles(
            recovery_v3,
            phase_signal,
            fps=fps,
            trim_start=trim_start,
            phase_signal_name=phase_signal_name,
        )
        use_as_fallback = not reps and len(recovery_reps) >= 3
        use_as_refinement = (
            bool(reps)
            and len(recovery_reps) >= 3
            and _rep_bottoms_align(reps, recovery_reps, fps=fps)
        )
        if use_as_fallback or use_as_refinement:
            if use_as_refinement:
                for candidate, primary in zip(recovery_chain["sets"][0]["reps"], reps):
                    candidate["status"] = primary.get("status") or "complete"
                    candidate["partial_side"] = primary.get("partial_side")
                    candidate.setdefault("quality_flags", []).append(
                        "legacy_boundary_refinement"
                    )
                recovery_v3 = analysis_v3.build_v3_analysis(
                    recovery_chain,
                    {**clip, "action": action_name},
                    gate=gate,
                )
                analysis_v3.validate_v3_analysis(recovery_v3)
                recovery_reps, recovery_phases, _, recovery_rejected = adapt_cycles(
                    recovery_v3,
                    phase_signal,
                    fps=fps,
                    trim_start=trim_start,
                    phase_signal_name=phase_signal_name,
                )
            v3 = recovery_v3
            reps = recovery_reps
            phases = recovery_phases
            rejected = recovery_rejected
            selection_mode = (
                "fallback" if use_as_fallback else "boundary_refinement"
            )
            selection_reason = (
                "repchain_abstained_with_recoverable_legacy_candidates"
                if use_as_fallback
                else "legacy_bottom_anchors_refined_repchain_boundaries"
            )
            recovery_evidence["status"] = "applied"
            recovery_evidence["mode"] = selection_mode
            recovery_evidence["reason"] = selection_reason
            write_json(repchain_dir / f"{clip_id}_primary.json", primary_v3)
        else:
            recovery_evidence["status"] = "rejected"
            recovery_evidence["reason"] = (
                "recovery_analysis_not_usable"
                if len(recovery_reps) < 3
                else "recovery_cycles_do_not_match_primary"
            )
    elif recovery_chain is not None:
        recovery_evidence["status"] = "rejected"
        recovery_evidence["reason"] = "action_mismatch_or_unsupported"

    if selection_mode == "none" and reps:
        selection_mode = "primary"
        selection_reason = "repchain_chain_accepted"
    rep_contract.attach_rep_provenance(reps, selection_mode=selection_mode)
    cycle_counts = rep_contract.count_cycles(
        reps,
        rejected_candidate_count=len(rejected),
    )
    rep_selection = rep_contract.build_rep_selection(
        reps,
        mode=selection_mode,
        reason=selection_reason,
        rejected_candidate_count=len(rejected),
    )
    recovery_used = selection_mode == "fallback"
    boundary_refinement_used = selection_mode == "boundary_refinement"
    write_json(repchain_dir / f"{clip_id}.json", v3)
    complete_rep_count = cycle_counts["complete_cycle"]
    endpoint_partial_rep_count = cycle_counts["endpoint_partial_cycle"]

    adapted_clip_dir = output_dir / clip_id
    _copy_legacy_tree(legacy_clip_dir, adapted_clip_dir)
    adapted = copy.deepcopy(legacy)
    adapted["algorithm"] = {"name": "repchain", "version": "v3"}
    adapted["rep_selection"] = rep_selection
    adapted["rep_count"] = cycle_counts["accepted_cycle"]
    adapted["rejected_rep_count"] = len(rejected)
    if adapted.get("status") == "ok" and len(reps) < 3:
        adapted["status"] = "partial"
    segmentation = adapted.setdefault("segmentation", {})
    segmentation["legacy_method"] = segmentation.get("method")
    segmentation["legacy_phase_angle"] = segmentation.get("phase_angle")
    segmentation["legacy_rep_count"] = len(segmentation.get("reps") or [])
    segmentation["method"] = "repchain_v3"
    segmentation["phase_angle"] = phase_signal_name
    segmentation["angle"] = phase_signal_name
    segmentation["rep_unit"] = "cycle"
    segmentation["reps"] = reps
    segmentation["phases"] = phases
    segmentation["rejected_reps"] = rejected
    segmentation["rep_selection"] = rep_selection
    segmentation["candidate_sources"] = {
        "selected": rep_selection["selected_source"],
        "selected_algorithm": "repchain_v3",
        "selection_mode": selection_mode,
        "repchain_complete_rep_count": complete_rep_count,
        "repchain_endpoint_partial_rep_count": endpoint_partial_rep_count,
        "accepted_cycle_count": cycle_counts["accepted_cycle"],
        "repchain_set_count": len(v3.get("sets") or []),
        "legacy_recovery_used": recovery_used,
        "legacy_boundary_refinement_used": boundary_refinement_used,
    }
    confidence = v3.get("confidence") or {}
    segmentation["repchain"] = {
        "schema_version": v3.get("schema_version"),
        "frame_space": "clip_local",
        "status": v3.get("status"),
        "action_status": (v3.get("action_verification") or {}).get("status"),
        "quality_status": v3.get("quality_gate", {}).get("status"),
        "abstained": confidence.get("abstained"),
        "abstain_reason": confidence.get("abstain_reason"),
        "calibrated": confidence.get("calibrated"),
        "selected_phase_signal": phase_signal_name,
        "event_anchor_policy": v3.get("anchor_policy"),
        "sets": [
            {
                "set_id": item.get("set_id"),
                "start_frame": item.get("start_frame"),
                "end_frame": item.get("end_frame"),
                "status": item.get("status"),
                "rep_count": len(item.get("reps") or []),
            }
            for item in v3.get("sets") or []
        ],
        "non_rep_intervals": v3.get("non_rep_intervals") or [],
        "output": str((repchain_dir / f"{clip_id}.json").resolve()),
        "recovery": recovery_evidence,
    }
    adapted["outputs"] = {
        name: str((adapted_clip_dir / relative).resolve())
        for name, relative in {
            "clip": "clip.json",
            "quality": "quality.json",
            "exercise": "exercise.json",
            "reps": "reps.csv",
            "rejected_reps": "rejected_reps.csv",
            "phases": "phases.csv",
            "angles": "angles/angles.csv",
            "angles_wide": "angles/angles_wide.csv",
            "landmarks": "angles/landmarks.json",
        }.items()
    }
    angle_extraction = adapted.setdefault("angle_extraction", {})
    angle_extraction["outputs"] = {
        name: str((adapted_clip_dir / relative).resolve())
        for name, relative in {
            "landmarks": "angles/landmarks.json",
            "landmarks_body_frame": "angles/landmarks_body_frame.json",
            "angles": "angles/angles.csv",
            "angles_wide": "angles/angles_wide.csv",
        }.items()
    }
    quality_payload = adapted.setdefault("quality", {})
    quality_checks = quality_payload.setdefault("checks", {})
    quality_checks["legacy_rep_count"] = segmentation.get("legacy_rep_count")
    quality_checks["rep_count"] = cycle_counts["accepted_cycle"]
    quality_checks["accepted_cycle_count"] = cycle_counts["accepted_cycle"]
    quality_checks["complete_cycle_count"] = cycle_counts["complete_cycle"]
    quality_checks["endpoint_partial_cycle_count"] = cycle_counts[
        "endpoint_partial_cycle"
    ]
    quality_checks["candidate_rep_count"] = cycle_counts["accepted_cycle"] + len(rejected)
    quality_checks["rejected_rep_count"] = len(rejected)
    quality_checks["rep_count_basis"] = "accepted_cycle"
    warnings = [
        str(value)
        for value in quality_payload.get("warnings") or []
        if str(value)
        not in {
            "no_complete_reps",
            "fewer_than_three_complete_reps",
            "no_accepted_cycles",
            "fewer_than_three_accepted_cycles",
        }
    ]
    cycle_warning = rep_contract.cycle_count_warning(cycle_counts["accepted_cycle"])
    if cycle_warning:
        warnings.append(cycle_warning)
    quality_payload["warnings"] = warnings
    quality_payload["legacy_warning_aliases"] = {
        "no_accepted_cycles": "no_complete_reps",
        "fewer_than_three_accepted_cycles": "fewer_than_three_complete_reps",
    } if cycle_warning else {}
    quality_payload["status"] = (
        "reject"
        if quality_payload.get("failures")
        else ("warn" if warnings else "pass")
    )
    adapted["quality_status"] = quality_payload["status"]
    quality_payload["repchain"] = {
        "status": v3.get("status"),
        "abstained": confidence.get("abstained"),
        "abstain_reason": confidence.get("abstain_reason"),
        "rep_selection": rep_selection,
        "accepted_cycle_count": cycle_counts["accepted_cycle"],
        "complete_rep_count": complete_rep_count,
        "endpoint_partial_rep_count": endpoint_partial_rep_count,
    }

    write_json(adapted_clip_dir / "analysis.json", adapted)
    write_json(adapted_clip_dir / "quality.json", quality_payload)
    write_csv(adapted_clip_dir / "reps.csv", reps, REP_FIELDS)
    write_csv(adapted_clip_dir / "phases.csv", phases, PHASE_FIELDS)
    write_csv(adapted_clip_dir / "rejected_reps.csv", rejected, REJECTED_FIELDS)
    return {
        "clip_id": clip_id,
        "status": adapted.get("status"),
        "repchain_status": v3.get("status"),
        "action_status": (v3.get("action_verification") or {}).get("status"),
        "abstained": (v3.get("confidence") or {}).get("abstained"),
        "abstain_reason": (v3.get("confidence") or {}).get("abstain_reason"),
        "rep_count": cycle_counts["accepted_cycle"],
        "accepted_cycle_count": cycle_counts["accepted_cycle"],
        "complete_rep_count": complete_rep_count,
        "endpoint_partial_rep_count": endpoint_partial_rep_count,
        "rejected_rep_count": len(rejected),
        "set_count": len(v3.get("sets") or []),
        "selected_phase_signal": phase_signal_name,
        "rep_selection_mode": selection_mode,
        "rep_selection": rep_selection,
        "recovery_used": recovery_used,
        "boundary_refinement_used": boundary_refinement_used,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clip-id", nargs="+", required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repchain-dir", type=Path, required=True)
    parser.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    parser.add_argument("--segmenter-config", type=Path, default=DEFAULT_SEGMENTER)
    parser.add_argument("--chain-config", type=Path, default=DEFAULT_CHAIN)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clips = _clip_index(load_json(args.manifest))
    missing = [clip_id for clip_id in args.clip_id if clip_id not in clips]
    if missing:
        raise ValueError("unknown clip_id: " + ", ".join(missing))
    specs = load_json(args.specs)
    segmenter_config = load_json(args.segmenter_config)
    chain_config = chain_selection.load_chain_config(args.chain_config)
    rows = [
        analyze_clip(
            clips[clip_id],
            legacy_dir=args.analysis_dir,
            output_dir=args.output_dir,
            repchain_dir=args.repchain_dir,
            specs=specs,
            segmenter_config=segmenter_config,
            chain_config=chain_config,
        )
        for clip_id in args.clip_id
    ]
    summary = {
        "schema_version": "vls-upload-repchain-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": {"name": "repchain", "version": "v3"},
        "clips": rows,
        "configs": {
            "segmenter": str(args.segmenter_config.resolve()),
            "chain": str(args.chain_config.resolve()),
        },
    }
    write_json(args.repchain_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
