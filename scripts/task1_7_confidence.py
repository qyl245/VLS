#!/usr/bin/env python3
"""C-lite selective confidence built on the frozen T1-6 calibrator."""

from __future__ import annotations

import copy
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import task1_6_confidence as base_confidence
import task1_7_quality as quality_lite


DEFAULT_CONFIG = ROOT / "configs" / "task1_7_confidence_frozen.json"
CONFIG_SCHEMA_VERSION = "task1-7c-confidence-config-v1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_json(path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"unexpected confidence config: {config.get('schema_version')}")
    core = config.get("core") or {}
    if not core.get("dataset_independent"):
        raise ValueError("confidence config must be dataset independent")
    return config


def load_calibrator(config: dict[str, Any]) -> dict[str, Any]:
    core = config["core"]
    path = Path(core["calibrator_config"])
    if not path.is_absolute():
        path = ROOT / path
    observed = sha256_path(path)
    expected = str(core["calibrator_sha256"])
    if observed != expected:
        raise ValueError(
            f"calibrator hash mismatch: expected {expected}, observed {observed}"
        )
    payload = load_json(path)
    return payload["core"]


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


def _evidence_by_id(
    evidence: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("evidence_id")): item
        for item in evidence
        if item.get("evidence_id") is not None
    }


def _multiplier(status: str, multipliers: dict[str, Any]) -> float:
    return float(multipliers.get(status, multipliers.get("unknown", 0.0)))


def _quality_factor(
    evidence: list[dict[str, Any]],
    multipliers: dict[str, Any],
    *,
    scopes: tuple[str, ...],
) -> tuple[float, str]:
    status = quality_lite.worst_status(evidence, scopes=scopes)
    return _multiplier(status, multipliers), status


def _clip_quality_factor(
    evidence: list[dict[str, Any]],
    multipliers: dict[str, Any],
) -> tuple[float, str]:
    """Return the conservative output-level factor for all quality evidence."""

    return _quality_factor(
        evidence,
        multipliers,
        scopes=("clip", "set", "rep"),
    )


def _rep_context_quality_factor(
    evidence: list[dict[str, Any]],
    multipliers: dict[str, Any],
) -> tuple[float, str]:
    """Return the clip/set factor inherited by every rep."""

    return _quality_factor(
        evidence,
        multipliers,
        scopes=("clip", "set"),
    )


def _rep_quality_factors(
    evidence: list[dict[str, Any]],
    multipliers: dict[str, Any],
) -> dict[tuple[int, int], float]:
    factors: dict[tuple[int, int], float] = {}
    for item in evidence:
        if item.get("scope") != "rep":
            continue
        parts = str(item.get("evidence_id") or "").split(":")
        if len(parts) != 3:
            continue
        try:
            key = (int(parts[1]), int(parts[2]))
        except ValueError:
            continue
        factors[key] = _multiplier(str(item.get("status")), multipliers)
    return factors


def _set_quality_factors(
    evidence: list[dict[str, Any]],
    multipliers: dict[str, Any],
) -> dict[int, float]:
    factors: dict[int, float] = {}
    for item in evidence:
        if item.get("scope") != "set":
            continue
        try:
            set_id = int(item.get("target_id"))
        except (TypeError, ValueError):
            continue
        factors[set_id] = _multiplier(str(item.get("status")), multipliers)
    return factors


def _downweight_structure(
    analysis: dict[str, Any],
    evidence: list[dict[str, Any]],
    multipliers: dict[str, Any],
) -> None:
    clip_factor, _ = _rep_context_quality_factor(evidence, multipliers)
    set_factors = _set_quality_factors(evidence, multipliers)
    rep_factors = _rep_quality_factors(evidence, multipliers)
    for set_candidate in analysis.get("sets") or []:
        try:
            set_id = int(set_candidate.get("set_id"))
        except (TypeError, ValueError):
            set_id = 0
        set_factor = min(clip_factor, set_factors.get(set_id, 1.0))
        if set_factor < 1.0:
            set_candidate["chain_confidence"] = round(
                max(
                    0.0,
                    min(
                        1.0,
                        float(set_candidate.get("chain_confidence") or 0.0)
                        * set_factor,
                    ),
                ),
                6,
            )
            flags = set(set_candidate.get("quality_flags") or [])
            flags.add("c_lite_set_confidence_downweighted")
            set_candidate["quality_flags"] = sorted(flags)
        for rep in set_candidate.get("reps") or []:
            try:
                rep_id = int(rep.get("rep_id"))
            except (TypeError, ValueError):
                rep_id = 0
            factor = min(
                set_factor,
                rep_factors.get((set_id, rep_id), 1.0),
            )
            if factor >= 1.0:
                continue
            confidence = rep.get("confidence") or {}
            for layer in ("complete", "in_set", "combined"):
                confidence[layer] = round(
                    max(0.0, min(1.0, float(confidence.get(layer) or 0.0) * factor)),
                    6,
                )
            rep["confidence"] = confidence
            flags = set(rep.get("quality_flags") or [])
            flags.add("c_lite_confidence_downweighted")
            rep["quality_flags"] = sorted(flags)


def _base_components(
    analysis: dict[str, Any],
    signal_evidence: dict[str, Any] | None,
) -> dict[str, float]:
    set_candidate = _default_set(analysis)
    reps = _complete_reps(set_candidate)
    rep_score = (
        float(
            statistics.median(
                float((rep.get("confidence") or {}).get("combined") or 0.0)
                for rep in reps
            )
        )
        if reps
        else 0.0
    )
    chain_score = float((set_candidate or {}).get("chain_confidence") or 0.0)
    evidence = signal_evidence or {}
    signal_score = float(evidence.get("consensus_score") or 0.0)
    if signal_score <= 0.0:
        signal_score = float((analysis.get("diagnostics") or {}).get("periodicity") or 0.0)
    return {
        "signal": max(0.0, min(1.0, signal_score)),
        "chain": max(0.0, min(1.0, chain_score)),
        "rep": max(0.0, min(1.0, rep_score)),
    }


def _calibrate(value: float, calibrator: dict[str, Any]) -> float:
    return base_confidence.calibrate_score(max(0.0, min(1.0, float(value))), calibrator)


def build_output_confidence(
    analysis: dict[str, Any],
    *,
    quality_evidence: list[dict[str, Any]],
    action_verification: dict[str, Any] | None,
    signal_evidence: dict[str, Any] | None,
    config: dict[str, Any],
    calibrator: dict[str, Any],
) -> dict[str, Any]:
    core = config["core"]
    components = _base_components(analysis, signal_evidence)
    quality_factor, quality_status = _clip_quality_factor(
        quality_evidence, core["quality_multipliers"]
    )
    action_status = str((action_verification or {}).get("status") or "ambiguous")
    action_factor = _multiplier(action_status, core["action_multipliers"])
    action_score = float((action_verification or {}).get("confidence") or 0.0)
    components.update(
        {
            "quality": quality_factor,
            "action": action_score,
        }
    )
    complete_raw = components["rep"] * quality_factor * action_factor
    in_set_raw = components["chain"] * quality_factor * action_factor
    combined_raw = complete_raw * in_set_raw
    calibrated = {
        "complete": round(_calibrate(complete_raw, calibrator), 6),
        "in_set": round(_calibrate(in_set_raw, calibrator), 6),
        "combined": round(_calibrate(combined_raw, calibrator), 6),
    }
    abstain_reason = None
    if quality_status in {"reject", "unknown"}:
        abstain_reason = f"quality_{quality_status}"
    elif bool((signal_evidence or {}).get("ambiguous")):
        abstain_reason = "phase_ambiguous"
    elif action_status != "supported":
        abstain_reason = f"action_{action_status}"
    elif not _complete_reps(_default_set(analysis)):
        abstain_reason = "no_confident_set"
    elif calibrated["combined"] < float(core["minimum_reportable_confidence"]):
        abstain_reason = "low_confidence"
    return {
        "components": {key: round(float(value), 6) for key, value in components.items()},
        "calibrated": calibrated,
        "abstained": abstain_reason is not None,
        "abstain_reason": abstain_reason,
    }


def apply_confidence_lite(
    analysis: dict[str, Any],
    *,
    quality_evidence: list[dict[str, Any]],
    action_verification: dict[str, Any] | None,
    signal_evidence: dict[str, Any] | None,
    config: dict[str, Any],
    calibrator: dict[str, Any],
) -> dict[str, Any]:
    confidence = build_output_confidence(
        analysis,
        quality_evidence=quality_evidence,
        action_verification=action_verification,
        signal_evidence=signal_evidence,
        config=config,
        calibrator=calibrator,
    )
    output = copy.deepcopy(analysis)
    _downweight_structure(
        output,
        quality_evidence,
        config["core"]["quality_multipliers"],
    )
    output["confidence"] = confidence
    output.setdefault("diagnostics", {})["c_lite_confidence"] = {
        "method": "frozen_t1_6_calibrator_with_quality_action_gate",
        "quality_status": quality_lite.worst_status(
            quality_evidence, scopes=("clip", "set", "rep")
        ),
        "rep_context_quality_status": quality_lite.worst_status(
            quality_evidence, scopes=("clip", "set")
        ),
        "action_status": (action_verification or {}).get("status"),
        "raw_layers": {
            "complete": round(
                float(confidence["components"]["rep"])
                * float(confidence["components"]["quality"])
                * _multiplier(
                    str((action_verification or {}).get("status") or "ambiguous"),
                    config["core"]["action_multipliers"],
                ),
                6,
            ),
            "in_set": round(
                float(confidence["components"]["chain"])
                * float(confidence["components"]["quality"])
                * _multiplier(
                    str((action_verification or {}).get("status") or "ambiguous"),
                    config["core"]["action_multipliers"],
                ),
                6,
            ),
        },
        "calibrator_config": config["core"]["calibrator_config"],
        "calibrator_sha256": config["core"]["calibrator_sha256"],
        "abstained": confidence["abstained"],
        "abstain_reason": confidence["abstain_reason"],
    }
    return output
