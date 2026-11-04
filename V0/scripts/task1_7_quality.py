#!/usr/bin/env python3
"""Attach C-lite quality evidence and enforce existing quality gates."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import task1_6_quality as base_quality


DEFAULT_CONFIG = ROOT / "configs" / "task1_7_quality.json"
CONFIG_SCHEMA_VERSION = "task1-7c-quality-config-v1"
STATUS_RANK = {"pass": 0, "warn": 1, "reject": 2, "unknown": 3}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_json(path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"unexpected quality config: {config.get('schema_version')}")
    core = config.get("core") or {}
    if not core.get("dataset_independent"):
        raise ValueError("quality config must be dataset independent")
    return config


def _analysis_frame_count(analysis: dict[str, Any], fallback: int = 0) -> int:
    maximum = int(fallback or 0)
    for set_candidate in analysis.get("sets") or []:
        maximum = max(maximum, int(set_candidate.get("end_frame") or 0) + 1)
        for rep in set_candidate.get("reps") or []:
            maximum = max(maximum, int(rep.get("end_frame") or 0) + 1)
    for interval in analysis.get("non_rep_intervals") or []:
        maximum = max(maximum, int(interval.get("end_frame") or 0) + 1)
    for event in (analysis.get("diagnostics") or {}).get(
        "selected_top_events"
    ) or []:
        maximum = max(maximum, int(event.get("frame") or 0) + 1)
    return max(0, maximum)


def _evidence(
    evidence_id: str,
    *,
    status: str,
    scope: str,
    source: str,
    reasons: Iterable[Any] | None,
    start_frame: int | None,
    end_frame: int | None,
    target_id: Any = None,
    value: float | str | bool | None = None,
    threshold: float | str | bool | None = None,
    severity: float | None = None,
) -> dict[str, Any]:
    normalized = base_quality.normalize_status(status)
    return {
        "evidence_id": evidence_id,
        "status": normalized,
        "scope": scope,
        "source": source,
        "reasons": [str(value) for value in (reasons or []) if str(value)],
        "target_id": target_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "value": value,
        "threshold": threshold,
        "severity": (
            float(severity)
            if severity is not None
            else float(STATUS_RANK[normalized])
        ),
    }


def build_quality_evidence(
    analysis: dict[str, Any],
    gate: dict[str, Any],
    config: dict[str, Any],
    *,
    frame_count: int | None = None,
) -> list[dict[str, Any]]:
    """Build clip/set/rep evidence without inventing new blocking rules."""

    settings = config["core"]
    normalized_gate = base_quality.make_quality_gate(
        gate.get("status"),
        reasons=gate.get("reasons"),
        source=str(gate.get("source") or "pose"),
        scope=str(gate.get("scope") or "clip"),
        target_id=gate.get("target_id"),
    )
    count = _analysis_frame_count(analysis, frame_count or 0)
    evidence = [
        _evidence(
            "clip:quality_gate",
            status=normalized_gate["status"],
            scope="clip",
            source=normalized_gate["source"],
            reasons=normalized_gate["reasons"],
            start_frame=0 if count else None,
            end_frame=count - 1 if count else None,
            value=normalized_gate["status"],
            threshold="pass",
        )
    ]
    set_warn_statuses = set(settings.get("set_warn_statuses") or [])
    rep_warn_statuses = set(settings.get("rep_warn_statuses") or [])
    for set_candidate in analysis.get("sets") or []:
        set_status = str(set_candidate.get("status") or "unknown")
        status = "warn" if set_status in set_warn_statuses else "pass"
        evidence.append(
            _evidence(
                f"set:{set_candidate.get('set_id')}",
                status=status,
                scope="set",
                source="chain",
                reasons=[f"set_status:{set_status}"] if status != "pass" else [],
                start_frame=int(set_candidate.get("start_frame") or 0),
                end_frame=int(set_candidate.get("end_frame") or 0),
                target_id=set_candidate.get("set_id"),
                value=float(set_candidate.get("chain_confidence") or 0.0),
                threshold=settings.get("minimum_set_chain_confidence"),
            )
        )
        for rep in set_candidate.get("reps") or []:
            rep_status = str(rep.get("status") or "unknown")
            flags = [str(value) for value in rep.get("quality_flags") or []]
            status = "warn" if rep_status in rep_warn_statuses else "pass"
            if any(flag in set(settings.get("rep_warn_flags") or []) for flag in flags):
                status = "warn"
            evidence.append(
                _evidence(
                    f"rep:{set_candidate.get('set_id')}:{rep.get('rep_id')}",
                    status=status,
                    scope="rep",
                    source="chain",
                    reasons=(
                        [f"rep_status:{rep_status}", *flags]
                        if status != "pass"
                        else []
                    ),
                    start_frame=int(rep.get("start_frame") or 0),
                    end_frame=int(rep.get("end_frame") or 0),
                    target_id=rep.get("rep_id"),
                    value=float((rep.get("confidence") or {}).get("combined") or 0.0),
                    threshold=settings.get("minimum_rep_combined_confidence"),
                )
            )
    return evidence


def worst_status(
    evidence: Iterable[dict[str, Any]],
    *,
    scopes: Iterable[str] | None = None,
) -> str:
    allowed = set(scopes or ["clip", "set", "rep", "frame"])
    statuses = [
        base_quality.normalize_status(item.get("status"))
        for item in evidence
        if item.get("scope") in allowed
    ]
    if not statuses:
        return "unknown"
    return max(statuses, key=lambda value: STATUS_RANK[value])


def apply_quality_lite(
    analysis: dict[str, Any],
    gate: dict[str, Any] | None,
    config: dict[str, Any],
    *,
    frame_count: int | None = None,
) -> dict[str, Any]:
    """Apply existing T1-6 gates and attach C-lite evidence."""

    source_gate = gate or analysis.get("quality_gate") or base_quality.make_quality_gate(
        "pass", source="pose", scope="clip"
    )
    normalized_gate = base_quality.make_quality_gate(
        source_gate.get("status"),
        reasons=source_gate.get("reasons"),
        source=str(source_gate.get("source") or "pose"),
        scope=str(source_gate.get("scope") or "clip"),
        target_id=source_gate.get("target_id"),
    )
    evidence = build_quality_evidence(
        analysis,
        normalized_gate,
        config,
        frame_count=frame_count,
    )
    output = base_quality.apply_quality_gates(copy.deepcopy(analysis), [normalized_gate])
    output["schema_version"] = "task1-7-clip-analysis-v3"
    output["quality_evidence"] = evidence
    output.setdefault("diagnostics", {})["c_lite_quality"] = {
        "clip_status": normalized_gate["status"],
        "worst_clip_set_status": worst_status(evidence, scopes=("clip", "set")),
        "evidence_count": len(evidence),
        "new_blocking_rules": False,
    }
    return output
