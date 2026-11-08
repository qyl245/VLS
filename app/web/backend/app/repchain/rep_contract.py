"""Shared semantic contract for final rep selection and cycle counts."""

from __future__ import annotations

from typing import Any, Iterable


SCHEMA_VERSION = "vls-rep-selection-v1"
ACCEPTED_CYCLE_STATUSES = frozenset({"complete", "endpoint_partial"})
SELECTION_MODES = frozenset(
    {"primary", "boundary_refinement", "fallback", "none"}
)


def accepted_cycle_reps(reps: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return reportable cycles, including trustworthy endpoint partials."""

    return [
        rep
        for rep in reps
        if str(rep.get("status") or "") in ACCEPTED_CYCLE_STATUSES
    ]


def count_cycles(
    reps: Iterable[dict[str, Any]],
    *,
    rejected_candidate_count: int = 0,
) -> dict[str, int]:
    """Count reps using the public contract's cycle vocabulary."""

    rep_list = list(reps)
    complete = sum(rep.get("status") == "complete" for rep in rep_list)
    endpoint_partial = sum(
        rep.get("status") == "endpoint_partial" for rep in rep_list
    )
    accepted = complete + endpoint_partial
    return {
        "accepted_cycle": accepted,
        "complete_cycle": complete,
        "endpoint_partial_cycle": endpoint_partial,
        "rejected_candidate": max(0, int(rejected_candidate_count)),
    }


def rep_provenance(selection_mode: str) -> dict[str, str]:
    """Describe which algorithm supplied each final cycle boundary."""

    if selection_mode == "fallback":
        bottom_source = "legacy_candidate"
    elif selection_mode in {"primary", "boundary_refinement"}:
        bottom_source = (
            "legacy_candidate"
            if selection_mode == "boundary_refinement"
            else "repchain_phase"
        )
    else:
        bottom_source = "none"
    return {
        "selection": selection_mode,
        "top_boundary_source": (
            "repchain_phase" if selection_mode != "none" else "none"
        ),
        "bottom_boundary_source": bottom_source,
    }


def attach_rep_provenance(
    reps: Iterable[dict[str, Any]],
    *,
    selection_mode: str,
) -> None:
    """Attach the canonical per-rep source contract in place."""

    provenance = rep_provenance(selection_mode)
    for rep in reps:
        rep["provenance"] = dict(provenance)


def build_rep_selection(
    reps: Iterable[dict[str, Any]],
    *,
    mode: str,
    reason: str,
    rejected_candidate_count: int = 0,
) -> dict[str, Any]:
    """Build the canonical final-rep selection payload."""

    if mode not in SELECTION_MODES:
        raise ValueError(f"unknown rep selection mode: {mode}")
    rep_list = list(reps)
    if mode == "none" and accepted_cycle_reps(rep_list):
        raise ValueError("mode 'none' cannot contain accepted cycles")
    if mode != "none" and not accepted_cycle_reps(rep_list):
        raise ValueError(f"mode '{mode}' requires at least one accepted cycle")
    selected_source = {
        "primary": "repchain",
        "boundary_refinement": "repchain",
        "fallback": "legacy",
        "none": None,
    }[mode]
    counts = count_cycles(
        rep_list,
        rejected_candidate_count=rejected_candidate_count,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "selected_source": selected_source,
        "reason": reason,
        "counts": counts,
        # `rep_count` remains a compatibility alias for accepted cycles.
        "aliases": {"rep_count": counts["accepted_cycle"]},
    }


def cycle_count_warning(accepted_cycle_count: int) -> str | None:
    """Return the canonical warning code for a final accepted rep count."""

    if accepted_cycle_count <= 0:
        return "no_accepted_cycles"
    if accepted_cycle_count < 3:
        return "fewer_than_three_accepted_cycles"
    return None
