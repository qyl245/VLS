#!/usr/bin/env python3
"""Verify the minimal squat/pushup action contract for C/E-lite."""

from __future__ import annotations

import copy
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "task1_7_actions.json"
CONFIG_SCHEMA_VERSION = "task1-7e-actions-config-v1"
PROTECTED_STATUSES = {
    "rejected_quality",
    "not_evaluable_quality",
    "phase_ambiguous",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_json(path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"unexpected action config: {config.get('schema_version')}")
    core = config.get("core") or {}
    if not core.get("dataset_independent"):
        raise ValueError("action config must be dataset independent")
    actions = config.get("actions") or {}
    if not actions:
        raise ValueError("action config must define at least one action")
    for name, policy in actions.items():
        if not policy.get("allowed_phase_families"):
            raise ValueError(f"action {name} has no allowed phase family")
        low, high = policy.get("period_seconds_range") or [0.0, math.inf]
        if float(low) < 0 or float(high) <= float(low):
            raise ValueError(f"action {name} has invalid period range")
    return config


def signal_family(name: Any) -> str:
    family = str(name or "").strip()
    if family.endswith("_bilateral_mean"):
        family = family[: -len("_bilateral_mean")]
    for prefix in ("left_", "right_"):
        if family.startswith(prefix):
            family = family[len(prefix) :]
            break
    return family


def _default_set(analysis: dict[str, Any]) -> dict[str, Any] | None:
    sets = analysis.get("sets") or []
    return next(
        (item for item in sets if item.get("is_default")),
        sets[0] if sets else None,
    )


def _complete_reps(set_candidate: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not set_candidate:
        return []
    return [
        rep
        for rep in set_candidate.get("reps") or []
        if rep.get("status") == "complete"
    ]


def _period_seconds(
    analysis: dict[str, Any],
    set_candidate: dict[str, Any] | None,
    reps: list[dict[str, Any]],
    fps: float | None,
) -> float | None:
    if reps and fps and float(fps) > 0:
        starts = [float(rep.get("start_frame") or 0.0) for rep in reps]
        intervals = [
            right - left for left, right in zip(starts[:-1], starts[1:]) if right > left
        ]
        if intervals:
            return round(float(statistics.median(intervals)) / float(fps), 6)
    diagnostics = analysis.get("diagnostics") or {}
    period_frames = diagnostics.get("period_frames")
    if period_frames is not None and fps and float(fps) > 0:
        return round(float(period_frames) / float(fps), 6)
    return None


def verify_action(
    analysis: dict[str, Any],
    action: str,
    *,
    fps: float | None,
    config: dict[str, Any],
    signal_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    core = config["core"]
    actions = config["actions"]
    normalized_action = str(action or "").strip().lower()
    if normalized_action not in actions or normalized_action not in set(
        core.get("supported_actions") or actions
    ):
        return {
            "action": action or "unknown",
            "status": "unsupported",
            "supported": False,
            "confidence": 0.0,
            "reasons": ["unsupported_action"],
            "evidence": {
                "supported_actions": sorted(actions),
                "contract_version": config.get("revision"),
            },
        }

    policy = actions[normalized_action]
    diagnostics = analysis.get("diagnostics") or {}
    selected_signal = (
        (signal_evidence or {}).get("selected_phase_signal")
        or diagnostics.get("selected_phase_signal")
    )
    family = signal_family(selected_signal)
    allowed = {signal_family(value) for value in policy["allowed_phase_families"]}
    default_set = _default_set(analysis)
    reps = _complete_reps(default_set)
    period_seconds = _period_seconds(analysis, default_set, reps, fps)
    minimum_reps = int(policy.get("minimum_complete_reps", core["minimum_complete_reps"]))
    period_low, period_high = (
        float(value) for value in policy["period_seconds_range"]
    )
    evidence = {
        "contract_version": config.get("revision"),
        "selected_phase_signal": selected_signal,
        "phase_family": family or None,
        "allowed_phase_families": sorted(allowed),
        "complete_rep_count": len(reps),
        "minimum_complete_reps": minimum_reps,
        "period_seconds": period_seconds,
        "period_seconds_range": [period_low, period_high],
        "periodicity": diagnostics.get("periodicity"),
        "set_count": len(analysis.get("sets") or []),
        "full_rom_and_posture_checks": False,
    }
    reasons = []
    status = "supported"
    confidence = 0.5
    if not family:
        status = "ambiguous"
        reasons.append("missing_phase_signal")
    elif family not in allowed:
        status = "mismatch"
        reasons.append("phase_signal_family_mismatch")
    if status == "supported" and len(reps) < minimum_reps:
        status = "ambiguous"
        reasons.append("insufficient_complete_reps")
    if (
        status == "supported"
        and period_seconds is not None
        and not period_low <= period_seconds <= period_high
    ):
        status = "mismatch"
        reasons.append("period_out_of_contract")
    if status == "supported":
        periodicity = diagnostics.get("periodicity")
        confidence = (
            min(1.0, max(0.5, 0.5 + 0.5 * float(periodicity)))
            if periodicity is not None
            else 0.5
        )
    elif status == "ambiguous":
        confidence = 0.25
    else:
        confidence = 0.1
    return {
        "action": normalized_action,
        "status": status,
        "supported": status == "supported",
        "confidence": round(float(confidence), 6),
        "reasons": reasons,
        "evidence": evidence,
    }


def _clear_structure(
    analysis: dict[str, Any],
    status: str,
    verification: dict[str, Any],
) -> None:
    diagnostics = analysis.setdefault("diagnostics", {})
    diagnostics["pre_action_structure"] = {
        "status": analysis.get("status"),
        "set_count": len(analysis.get("sets") or []),
        "rep_count": sum(
            len(item.get("reps") or []) for item in analysis.get("sets") or []
        ),
        "selected_top_event_count": len(
            diagnostics.get("selected_top_events") or []
        ),
    }
    analysis["status"] = status
    analysis["sets"] = []
    analysis["rejected_candidates"] = []
    analysis["non_rep_intervals"] = []
    diagnostics["selected_top_events"] = []
    diagnostics["suppressed_by_action_verification"] = verification


def apply_action_verification(
    analysis: dict[str, Any],
    verification: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    output = copy.deepcopy(analysis)
    output["action_verification"] = verification
    diagnostics = output.setdefault("diagnostics", {})
    if output.get("status") in PROTECTED_STATUSES:
        diagnostics["action_verification_disposition"] = "status_precedence"
        return output
    status = verification.get("status")
    if status == "supported":
        diagnostics["action_verification_disposition"] = "reportable"
        return output
    target_status = {
        "mismatch": "action_mismatch",
        "unsupported": "unsupported_action",
        "ambiguous": "no_confident_set",
    }.get(str(status), "no_confident_set")
    _clear_structure(output, target_status, verification)
    diagnostics["action_verification_disposition"] = "fail_closed"
    return output
