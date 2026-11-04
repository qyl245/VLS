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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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

    repchain_dir.mkdir(parents=True, exist_ok=True)
    write_json(repchain_dir / f"{clip_id}_chain.json", chain_analysis)
    write_json(repchain_dir / f"{clip_id}_gate.json", gate)
    write_json(repchain_dir / f"{clip_id}.json", v3)

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
    complete_rep_count = sum(1 for rep in reps if rep.get("status") == "complete")
    endpoint_partial_rep_count = sum(
        1 for rep in reps if rep.get("status") == "endpoint_partial"
    )

    adapted_clip_dir = output_dir / clip_id
    _copy_legacy_tree(legacy_clip_dir, adapted_clip_dir)
    adapted = copy.deepcopy(legacy)
    adapted["algorithm"] = {"name": "repchain", "version": "v3"}
    adapted["rep_count"] = len(reps)
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
    segmentation["candidate_sources"] = {
        "selected": "repchain_v3",
        "repchain_complete_rep_count": complete_rep_count,
        "repchain_endpoint_partial_rep_count": endpoint_partial_rep_count,
        "repchain_set_count": len(v3.get("sets") or []),
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
    quality_checks["rep_count"] = len(reps)
    quality_checks["candidate_rep_count"] = len(reps) + len(rejected)
    quality_checks["rejected_rep_count"] = len(rejected)
    warnings = [
        str(value)
        for value in quality_payload.get("warnings") or []
        if str(value)
        not in {"no_complete_reps", "fewer_than_three_complete_reps"}
    ]
    if not reps:
        warnings.append("no_complete_reps")
    elif len(reps) < 3:
        warnings.append("fewer_than_three_complete_reps")
    quality_payload["warnings"] = warnings
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
        "rep_count": len(reps),
        "complete_rep_count": complete_rep_count,
        "endpoint_partial_rep_count": endpoint_partial_rep_count,
        "rejected_rep_count": len(rejected),
        "set_count": len(v3.get("sets") or []),
        "selected_phase_signal": phase_signal_name,
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
