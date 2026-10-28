#!/usr/bin/env python3
"""Normalize and apply T1-6 quality gates without source-specific rules."""

from __future__ import annotations

import copy
from typing import Any, Iterable


SCHEMA_VERSION = "task1-6-quality-gate-v1"
STATUSES = ("pass", "warn", "reject", "unknown")
SCOPES = ("clip", "set", "rep")
BLOCKING_STATUSES = {"reject", "unknown"}


def _string_list(values: Iterable[Any] | None) -> list[str]:
    return [str(value) for value in (values or []) if str(value).strip()]


def normalize_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    aliases = {
        "ok": "pass",
        "accepted": "pass",
        "warning": "warn",
        "rejected": "reject",
        "rejected_quality": "reject",
        "not_evaluable": "unknown",
    }
    status = aliases.get(status, status)
    return status if status in STATUSES else "unknown"


def make_quality_gate(
    status: Any,
    *,
    reasons: Iterable[Any] | None = None,
    source: str = "unknown",
    scope: str = "clip",
    target_id: Any = None,
) -> dict[str, Any]:
    if scope not in SCOPES:
        raise ValueError(f"unknown quality gate scope: {scope}")
    normalized = normalize_status(status)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": normalized,
        "scope": scope,
        "target_id": target_id,
        "reasons": _string_list(reasons),
        "source": str(source),
        "blocking": normalized in BLOCKING_STATUSES,
    }


def from_legacy_quality(
    legacy_analysis: dict[str, Any] | None,
    *,
    scope: str = "clip",
) -> dict[str, Any]:
    if not isinstance(legacy_analysis, dict):
        return make_quality_gate(
            "unknown",
            reasons=["quality_evidence_missing"],
            source="pose",
            scope=scope,
        )
    quality = legacy_analysis.get("quality")
    if not isinstance(quality, dict):
        quality = legacy_analysis if legacy_analysis.get("status") else {}
    status = (
        legacy_analysis.get("quality_status")
        or quality.get("status")
        or "unknown"
    )
    normalized = normalize_status(status)
    if normalized == "reject":
        reasons = quality.get("failures") or quality.get("reasons") or []
    elif normalized == "warn":
        reasons = quality.get("warnings") or quality.get("reasons") or []
    elif normalized == "unknown":
        reasons = quality.get("reasons") or ["quality_evidence_missing"]
    else:
        reasons = []
    return make_quality_gate(
        normalized,
        reasons=reasons,
        source="pose",
        scope=scope,
    )


def _refresh_default_set(sets: list[dict[str, Any]]) -> None:
    if not sets:
        return
    default = max(
        sets,
        key=lambda item: (
            len(item.get("reps") or []),
            float(item.get("chain_confidence") or 0.0),
            -int(item.get("start_frame") or 0),
        ),
    )
    for set_candidate in sets:
        set_candidate["is_default"] = set_candidate is default


def _clear_reportable_structure(
    analysis: dict[str, Any],
    gate: dict[str, Any],
) -> None:
    diagnostics = analysis.setdefault("diagnostics", {})
    diagnostics["pre_gate_structure"] = {
        "status": analysis.get("status"),
        "set_count": len(analysis.get("sets") or []),
        "rep_count": sum(
            len(set_candidate.get("reps") or [])
            for set_candidate in analysis.get("sets") or []
        ),
        "selected_top_event_count": len(
            diagnostics.get("selected_top_events") or []
        ),
    }
    analysis["sets"] = []
    analysis["rejected_candidates"] = []
    analysis["non_rep_intervals"] = []
    diagnostics["selected_top_events"] = []
    diagnostics["suppressed_by_quality_gate"] = gate


def apply_quality_gates(
    analysis: dict[str, Any],
    gates: Iterable[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Return a gated copy while preserving the input for raw evaluation."""
    output = copy.deepcopy(analysis)
    normalized = [
        make_quality_gate(
            gate.get("status"),
            reasons=gate.get("reasons"),
            source=str(gate.get("source") or "unknown"),
            scope=str(gate.get("scope") or "clip"),
            target_id=gate.get("target_id"),
        )
        for gate in (gates or [])
    ]
    if not normalized:
        normalized = [
            make_quality_gate("pass", source="pose", scope="clip")
        ]

    blocking = [gate for gate in normalized if gate["blocking"]]
    clip_block = next(
        (gate for gate in blocking if gate["scope"] == "clip"),
        None,
    )
    removed = {"set_ids": [], "rep_ids": []}
    if clip_block is not None:
        _clear_reportable_structure(output, clip_block)
    else:
        for gate in blocking:
            if gate["scope"] == "rep":
                target = str(gate["target_id"])
                for set_candidate in output.get("sets") or []:
                    kept = []
                    for rep in set_candidate.get("reps") or []:
                        if str(rep.get("rep_id")) == target:
                            removed["rep_ids"].append(rep.get("rep_id"))
                        else:
                            kept.append(rep)
                    set_candidate["reps"] = kept
                output["sets"] = [
                    set_candidate
                    for set_candidate in output.get("sets") or []
                    if set_candidate.get("reps")
                ]
            elif gate["scope"] == "set":
                target = str(gate["target_id"])
                kept_sets = []
                for set_candidate in output.get("sets") or []:
                    if str(set_candidate.get("set_id")) == target:
                        removed["set_ids"].append(set_candidate.get("set_id"))
                    else:
                        kept_sets.append(set_candidate)
                output["sets"] = kept_sets
        _refresh_default_set(output.get("sets") or [])

    statuses = [gate["status"] for gate in normalized]
    effective_status = "pass"
    for candidate in ("unknown", "reject", "warn"):
        if candidate in statuses:
            effective_status = candidate
            break
    effective = next(
        gate for gate in normalized if gate["status"] == effective_status
    )
    if clip_block is not None:
        output["status"] = (
            "rejected_quality"
            if clip_block["status"] == "reject"
            else "not_evaluable_quality"
        )
    elif blocking and not output.get("sets"):
        output["status"] = (
            "rejected_quality"
            if any(gate["status"] == "reject" for gate in blocking)
            else "not_evaluable_quality"
        )

    output["quality_gate"] = effective
    output["quality_gates"] = normalized
    output.setdefault("diagnostics", {})["quality_gate"] = effective
    if removed["set_ids"] or removed["rep_ids"]:
        output["diagnostics"]["quality_gate_removed"] = removed
    return output
