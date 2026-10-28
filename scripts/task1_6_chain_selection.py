#!/usr/bin/env python3
"""Select continuous rep chains and set candidates from T1-6B phase states."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import task1_6_phase_states as phase_states
import task1_6_quality as quality_gates


DEFAULT_SOURCES = ROOT / "configs" / "task1_6_sources.json"
DEFAULT_SPECS = ROOT / "configs" / "task1_4_exercise_specs.json"
DEFAULT_SEGMENTER_CONFIG = ROOT / "configs" / "task1_6_segmenter.json"
DEFAULT_CHAIN_CONFIG = ROOT / "configs" / "task1_6_chain.json"
DEFAULT_FIT3D_MANIFEST = ROOT / "configs" / "task1_5_fit3d_clips.json"
DEFAULT_FIT3D_ANALYSIS_DIR = ROOT / "results" / "task1_5" / "clip_analysis"
DEFAULT_QEVD_ANALYSIS_DIR = ROOT / "results" / "clip_analysis_v1"
DEFAULT_RESULTS_DIR = ROOT / "results" / "task1_6" / "dev"

CONFIG_SCHEMA_VERSION = "task1-6-chain-config-v1"
SUMMARY_SCHEMA_VERSION = "task1-6c-chain-eval-v1"
ANALYSIS_SCHEMA_VERSION = "task1-6-clip-analysis-v2"
DEV_DATASET = "Fit3D"
QEVD_DATASET = "QEVD-FIT-COACH"
EXPECTED_FIT3D_DEV_CLIPS = 32
EXPECTED_FIT3D_DEV_CLUSTERS = 8
EXPECTED_QEVD_DEV_CLIPS = 13
HIGH_CONFIDENCE_THRESHOLD = 0.50
EVENT_GATE_PERIOD_FRACTION = 0.25
TAIL_ANCHOR_POLICY = "earlier_of_plateau_midpoint_and_phase_zero_crossing"
TAIL_ANCHOR_SCOPE = "last_selected_event_per_chain"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    parser.add_argument(
        "--segmenter-config",
        type=Path,
        default=DEFAULT_SEGMENTER_CONFIG,
    )
    parser.add_argument("--chain-config", type=Path, default=DEFAULT_CHAIN_CONFIG)
    parser.add_argument(
        "--fit3d-manifest", type=Path, default=DEFAULT_FIT3D_MANIFEST
    )
    parser.add_argument(
        "--fit3d-analysis-dir",
        type=Path,
        default=DEFAULT_FIT3D_ANALYSIS_DIR,
    )
    parser.add_argument(
        "--qevd-analysis-dir",
        type=Path,
        default=DEFAULT_QEVD_ANALYSIS_DIR,
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def load_chain_config(path: Path) -> dict[str, Any]:
    payload = phase_states.load_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"unexpected chain config schema: {payload.get('schema_version')}"
        )
    core = payload.get("core") or {}
    if not core.get("dataset_independent"):
        raise ValueError("chain config must be dataset independent")
    return payload


def state_duration_fraction(state: dict[str, Any]) -> float:
    evidence = state.get("evidence") or {}
    try:
        return float(evidence.get("duration_fraction", 0.0))
    except (TypeError, ValueError):
        return 0.0


def state_fallback_penalty(state: dict[str, Any]) -> float:
    return 0.85 if state.get("status") == "fallback" else 1.0


def terminal_anchor_frame(
    node: dict[str, Any],
    policy: str = TAIL_ANCHOR_POLICY,
) -> int:
    if policy != TAIL_ANCHOR_POLICY:
        raise ValueError(f"unknown tail anchor policy: {policy}")
    anchors = (node.get("top") or {}).get("anchors") or {}
    midpoint = int(anchors.get("plateau_midpoint", node["event_frame"]))
    zero_crossing = int(anchors.get("phase_zero_crossing", midpoint))
    return min(midpoint, zero_crossing)


def tail_anchor_policy(config: dict[str, Any]) -> tuple[str, str]:
    tail = (config.get("core") or {}).get("tail_anchor") or {}
    policy = str(tail.get("policy", TAIL_ANCHOR_POLICY))
    scope = str(tail.get("scope", TAIL_ANCHOR_SCOPE))
    if policy != TAIL_ANCHOR_POLICY:
        raise ValueError(f"unknown tail anchor policy: {policy}")
    if scope != TAIL_ANCHOR_SCOPE:
        raise ValueError(f"unknown tail anchor scope: {scope}")
    return policy, scope


def strongest_bottom_between(
    bottoms: list[dict[str, Any]],
    left_frame: float,
    right_frame: float,
) -> dict[str, Any] | None:
    candidates = [
        state
        for state in bottoms
        if left_frame
        < 0.5 * (state["start_frame"] + state["end_frame"])
        < right_frame
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda state: (
            float(state.get("stable_fraction", 0.0)),
            -abs(
                0.5 * (left_frame + right_frame)
                - 0.5 * (state["start_frame"] + state["end_frame"])
            ),
        ),
    )


def build_candidate_nodes(
    top_states: list[dict[str, Any]],
    bottom_states: list[dict[str, Any]],
    period_frames: float,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    core = config["core"]
    nodes = []
    for index, top in enumerate(top_states):
        top_mid = 0.5 * (top["start_frame"] + top["end_frame"])
        if index == 0 and len(top_states) > 1:
            left_frame = top_mid
            right_frame = 0.5 * (
                top_states[1]["start_frame"] + top_states[1]["end_frame"]
            )
        elif index > 0:
            left_frame = 0.5 * (
                top_states[index - 1]["start_frame"]
                + top_states[index - 1]["end_frame"]
            )
            right_frame = top_mid
        else:
            left_frame = top_mid - period_frames
            right_frame = top_mid + period_frames
        bottom = strongest_bottom_between(
            bottom_states, left_frame, right_frame
        )
        if bottom is None and bottom_states:
            bottom = min(
                bottom_states,
                key=lambda state: abs(
                    top_mid
                    - 0.5 * (state["start_frame"] + state["end_frame"])
                ),
            )
        amplitude = (
            max(0.0, float(top["mean_angle"]) - float(bottom["mean_angle"]))
            if bottom is not None
            else 0.0
        )
        nodes.append(
            {
                "index": index,
                "top": top,
                "bottom": bottom,
                "event_frame": int(top["anchors"]["plateau_midpoint"]),
                "output_frame": int(top["anchors"]["plateau_midpoint"]),
                "amplitude": amplitude,
                "bottom_duration_fraction": state_duration_fraction(bottom)
                if bottom is not None
                else 0.0,
                "top_stability": float(top.get("stable_fraction", 0.0)),
                "bottom_stability": float(bottom.get("stable_fraction", 0.0))
                if bottom is not None
                else 0.0,
                "fallback_penalty": state_fallback_penalty(top)
                * state_fallback_penalty(bottom)
                if bottom is not None
                else state_fallback_penalty(top),
                "scale_support": int(
                    (top.get("evidence") or {}).get("scale_support") or 0
                ),
                "multi_scale_consensus": bool(
                    (top.get("evidence") or {}).get(
                        "multi_scale_consensus", False
                    )
                ),
                "partial_endpoint": top.get("kind") != "interior",
            }
        )

    positive_amplitudes = [
        node["amplitude"] for node in nodes if node["amplitude"] > 1e-8
    ]
    median_amplitude = (
        float(np.median(positive_amplitudes)) if positive_amplitudes else 0.0
    )
    weights = core["node_quality_weights"]
    for node in nodes:
        ratio = (
            node["amplitude"] / median_amplitude if median_amplitude > 0 else 0.0
        )
        relative_amplitude_score = min(1.0, max(0.0, ratio))
        quality = node["fallback_penalty"] * (
            float(weights["top_stability"]) * node["top_stability"]
            + float(weights["bottom_stability"]) * node["bottom_stability"]
            + float(weights["relative_amplitude"]) * relative_amplitude_score
        )
        node["relative_amplitude"] = ratio
        node["eligible"] = (
            node["bottom"] is not None
            and ratio >= float(core["minimum_relative_amplitude"])
        )
        node["quality"] = float(np.clip(quality, 0.0, 1.0))
        node["initial_confidence"] = {
            "complete": round(node["quality"], 6),
            "in_set": round(node["quality"], 6),
            "combined": round(node["quality"], 6),
        }
    return nodes


def local_extrema_stability_bridge(
    nodes: list[dict[str, Any]],
    bottom_states: list[dict[str, Any]],
    period_frames: float,
    config: dict[str, Any],
    chain_policy: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Enable one stable partial endpoint on each side from local peak/trough evidence."""

    if chain_policy is None:
        return []
    policy = chain_policy.get("local_extrema_stability") or {}
    if not bool(policy.get("enabled")) or period_frames <= 0:
        return []
    minimum_relative_amplitude = float(
        policy.get("minimum_relative_amplitude", 0.9)
    )
    minimum_top_stability = float(policy.get("minimum_top_stability", 0.7))
    minimum_bottom_stability = float(
        policy.get("minimum_bottom_stability", 0.4)
    )
    minimum_bottom_support = int(
        policy.get("minimum_bottom_scale_support", 2)
    )
    minimum_gap, maximum_gap = (
        float(value) for value in policy.get("gap_period_range", [0.75, 1.35])
    )
    positive_amplitudes = [
        float(node["amplitude"])
        for node in nodes
        if float(node.get("amplitude") or 0.0) > 0
    ]
    median_amplitude = (
        float(np.median(positive_amplitudes)) if positive_amplitudes else 0.0
    )
    if median_amplitude <= 0:
        return []

    candidates: dict[str, list[tuple[tuple[float, ...], dict[str, Any]]]] = {
        "leading": [],
        "trailing": [],
    }
    for node in nodes:
        kind = node["top"].get("kind")
        if kind not in ("leading_partial", "trailing_partial"):
            continue
        if node.get("eligible") or not bool(node.get("multi_scale_consensus")):
            continue
        if float(node.get("top_stability") or 0.0) < minimum_top_stability:
            continue
        role = "leading" if kind == "leading_partial" else "trailing"
        index = int(node["index"])
        if role == "leading":
            if index + 1 >= len(nodes) or not nodes[index + 1].get("eligible"):
                continue
            local_bottom = strongest_bottom_between(
                bottom_states,
                node["event_frame"],
                nodes[index + 1]["event_frame"],
            )
            gap_frames = nodes[index + 1]["event_frame"] - node["event_frame"]
        else:
            if index == 0 or not nodes[index - 1].get("eligible"):
                continue
            local_bottom = node.get("bottom")
            gap_frames = node["event_frame"] - nodes[index - 1]["event_frame"]
        if local_bottom is None:
            continue
        relative_amplitude = float(
            node["top"]["mean_angle"] - local_bottom["mean_angle"]
        ) / median_amplitude
        if relative_amplitude < minimum_relative_amplitude:
            continue
        bottom_stability = float(local_bottom.get("stable_fraction") or 0.0)
        if bottom_stability < minimum_bottom_stability:
            continue
        bottom_support = int(
            (local_bottom.get("evidence") or {}).get("scale_support") or 0
        )
        if bottom_support < minimum_bottom_support:
            continue
        gap_ratio = float(gap_frames) / period_frames
        if not minimum_gap <= gap_ratio <= maximum_gap:
            continue
        strength = (
            relative_amplitude,
            float(node.get("top_stability") or 0.0),
            bottom_stability,
            float(bottom_support),
        )
        candidates[role].append(
            (
                strength,
                {
                    "node": node,
                    "local_bottom": local_bottom,
                    "relative_amplitude": relative_amplitude,
                    "bottom_stability": bottom_stability,
                    "bottom_support": bottom_support,
                    "gap_frames": int(gap_frames),
                    "gap_ratio": gap_ratio,
                },
            )
        )

    records = []
    for role in ("leading", "trailing"):
        if not candidates[role]:
            continue
        _, selected = max(candidates[role], key=lambda item: item[0])
        node = selected["node"]
        node["eligible"] = True
        node["local_extrema_stability_bridged"] = True
        node["local_extrema_relative_amplitude"] = round(
            float(selected["relative_amplitude"]), 6
        )
        node["local_extrema_bottom_stability"] = round(
            float(selected["bottom_stability"]), 6
        )
        evidence = dict(node["top"].get("evidence") or {})
        evidence.update(
            {
                "local_extrema_stable": True,
                "local_extrema_role": role,
                "local_extrema_relative_amplitude": round(
                    float(selected["relative_amplitude"]), 6
                ),
                "local_extrema_bottom_stability": round(
                    float(selected["bottom_stability"]), 6
                ),
                "local_extrema_bottom_scale_support": int(
                    selected["bottom_support"]
                ),
                "local_extrema_gap_period_ratio": round(
                    float(selected["gap_ratio"]), 6
                ),
            }
        )
        node["top"]["evidence"] = evidence
        records.append(
            {
                "index": int(node["index"]),
                "frame": int(node["event_frame"]),
                "role": role,
                "kind": node["top"].get("kind"),
                "relative_amplitude": round(
                    float(selected["relative_amplitude"]), 6
                ),
                "top_stability": round(
                    float(node.get("top_stability") or 0.0), 6
                ),
                "bottom_stability": round(
                    float(selected["bottom_stability"]), 6
                ),
                "bottom_scale_support": int(selected["bottom_support"]),
                "gap_frames": int(selected["gap_frames"]),
                "gap_ratio": round(float(selected["gap_ratio"]), 6),
            }
        )
    return records


def build_edge(
    left: dict[str, Any],
    right: dict[str, Any],
    period_frames: float,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    core = config["core"]
    gap_frames = right["event_frame"] - left["event_frame"]
    if gap_frames <= 0 or period_frames <= 0:
        return None
    gap_ratio = gap_frames / period_frames
    minimum_ratio, maximum_ratio = core["gap_period_range"]
    if not float(minimum_ratio) <= gap_ratio <= float(maximum_ratio):
        return None
    period_score = math.exp(
        -0.5
        * (
            math.log(max(gap_ratio, 1e-6))
            / float(core["period_log_sigma"])
        )
        ** 2
    )
    amplitude_sum = left["amplitude"] + right["amplitude"]
    amplitude_denominator = 0.25 * amplitude_sum
    amplitude_score = (
        math.exp(
            -0.5
            * (
                (right["amplitude"] - left["amplitude"])
                / max(amplitude_denominator, 1e-6)
            )
            ** 2
        )
        if amplitude_sum > 1e-8
        else 0.0
    )
    duration_sum = (
        left["bottom_duration_fraction"]
        + right["bottom_duration_fraction"]
    )
    duration_score = (
        math.exp(
            -0.5
            * (
                (
                    right["bottom_duration_fraction"]
                    - left["bottom_duration_fraction"]
                )
                / max(0.15 * duration_sum, 1e-6)
            )
            ** 2
        )
        if duration_sum > 1e-8
        else 0.0
    )
    weights = core["edge_quality_weights"]
    quality = (
        float(weights["period"]) * period_score
        + float(weights["relative_amplitude"]) * amplitude_score
        + float(weights["bottom_duration"]) * duration_score
    )
    return {
        "gap_frames": int(gap_frames),
        "gap_ratio": float(gap_ratio),
        "period_score": float(period_score),
        "amplitude_score": float(amplitude_score),
        "duration_score": float(duration_score),
        "quality": float(np.clip(quality, 0.0, 1.0)),
        "allowed": quality >= float(core["minimum_edge_quality"]),
    }


def _apply_edge_policy(
    edge: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
    chain_policy: dict[str, Any] | None,
) -> None:
    for endpoint, other in ((left, right), (right, left)):
        allowed_neighbors = endpoint.get("context_bridge_allowed_neighbors")
        if (
            allowed_neighbors is not None
            and other["index"] not in allowed_neighbors
        ):
            edge["allowed"] = False
            edge["context_bridge_neighbor_rejected"] = True
            break
    if chain_policy is not None and (
        "endpoint_closure_gap_period_range" in chain_policy
    ) and (
        left["partial_endpoint"] or right["partial_endpoint"]
    ):
        edge["endpoint_closure"] = True
        singleton = (
            left["partial_endpoint"] and not left["multi_scale_consensus"]
        ) or (
            right["partial_endpoint"] and not right["multi_scale_consensus"]
        )
        if singleton:
            closure_range = chain_policy["endpoint_closure_gap_period_range"]
            if not (
                float(closure_range[0])
                <= edge["gap_ratio"]
                <= float(closure_range[1])
            ):
                edge["allowed"] = False
                edge["endpoint_closure_gap_rejected"] = True


def _local_period_context(
    edges: dict[tuple[int, int], dict[str, Any]],
    left: dict[str, Any],
    right: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any] | None:
    """Estimate the cycle period from reliable edges immediately around a gap."""

    intervals_per_side = int(policy.get("context_intervals_per_side", 2))
    minimum_intervals = int(policy.get("minimum_context_intervals", 3))
    maximum_relative_mad = float(policy.get("maximum_relative_mad", 0.25))
    candidates: list[tuple[int, int, int]] = []
    for (left_index, right_index), edge in edges.items():
        if not edge["allowed"]:
            continue
        candidates.append(
            (left_index, right_index, int(edge["gap_frames"]))
        )

    before = sorted(
        [
            item
            for item in candidates
            if item[1] <= left["index"]
        ],
        key=lambda item: item[1],
        reverse=True,
    )[:intervals_per_side]
    after = sorted(
        [
            item
            for item in candidates
            if item[0] >= right["index"]
        ],
        key=lambda item: item[0],
    )[:intervals_per_side]
    if (
        not before
        or not after
        or before[0][1] != left["index"]
        or after[0][0] != right["index"]
    ):
        return None
    intervals = [item[2] for item in before + after]
    if len(intervals) < minimum_intervals:
        return None
    local_period = float(np.median(intervals))
    if local_period <= 0:
        return None
    relative_mad = float(
        np.median([abs(value - local_period) for value in intervals])
        / local_period
    )
    if relative_mad > maximum_relative_mad:
        return None
    return {
        "period_frames": local_period,
        "context_intervals_frames": intervals,
        "context_edges": [
            {"left_index": int(item[0]), "right_index": int(item[1])}
            for item in before + after
        ],
        "relative_mad": relative_mad,
    }


def _build_local_period_gap_edge(
    nodes: list[dict[str, Any]],
    edges: dict[tuple[int, int], dict[str, Any]],
    left: dict[str, Any],
    right: dict[str, Any],
    period_frames: float,
    config: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any] | None:
    if right["index"] != left["index"] + 1:
        return None
    if (left["index"], right["index"]) in edges:
        return None
    gap_frames = int(right["event_frame"] - left["event_frame"])
    if gap_frames <= 0 or period_frames <= 0:
        return None
    context = _local_period_context(edges, left, right, policy)
    if context is None:
        return None
    local_period = float(context["period_frames"])
    local_gap_ratio = gap_frames / local_period
    minimum_ratio, maximum_ratio = config["core"]["gap_period_range"]
    if not float(minimum_ratio) <= local_gap_ratio <= float(maximum_ratio):
        return None
    edge = build_edge(left, right, local_period, config)
    if edge is None:
        return None
    edge.update(
        {
            "local_period_gap": True,
            "left_index": int(left["index"]),
            "right_index": int(right["index"]),
            "left_frame": int(left["event_frame"]),
            "right_frame": int(right["event_frame"]),
            "global_period_frames": round(float(period_frames), 6),
            "global_gap_ratio": round(
                float(gap_frames) / max(float(period_frames), 1e-6), 6
            ),
            "local_period_frames": round(local_period, 6),
            "local_gap_ratio": round(float(local_gap_ratio), 6),
            "local_period_context_intervals": context[
                "context_intervals_frames"
            ],
            "local_period_context_edges": context["context_edges"],
            "local_period_relative_mad": round(
                float(context["relative_mad"]), 6
            ),
        }
    )
    return edge


def build_edges(
    nodes: list[dict[str, Any]],
    period_frames: float,
    config: dict[str, Any],
    *,
    chain_policy: dict[str, Any] | None = None,
) -> dict[tuple[int, int], dict[str, Any]]:
    edges = {}
    eligible = [node for node in nodes if node["eligible"]]
    for left_position, left in enumerate(eligible):
        for right in eligible[left_position + 1 :]:
            edge = build_edge(left, right, period_frames, config)
            if edge is not None:
                _apply_edge_policy(edge, left, right, chain_policy)
                edges[(left["index"], right["index"])] = edge
    policy = (
        (chain_policy or {}).get("local_period_gap") or {}
    )
    if bool(policy.get("enabled")):
        standard_edges = dict(edges)
        for left, right in zip(nodes[:-1], nodes[1:]):
            if not left["eligible"] or not right["eligible"]:
                continue
            edge = _build_local_period_gap_edge(
                nodes,
                standard_edges,
                left,
                right,
                period_frames,
                config,
                policy,
            )
            if edge is None:
                continue
            _apply_edge_policy(edge, left, right, chain_policy)
            edges[(left["index"], right["index"])] = edge
    return edges


def connected_components(
    indices: list[int],
    edges: dict[tuple[int, int], dict[str, Any]],
) -> list[list[int]]:
    parent = {index: index for index in indices}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in edges:
        if edges[(left, right)]["allowed"]:
            union(left, right)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in indices:
        groups[find(index)].append(index)
    return sorted(
        [sorted(group) for group in groups.values()],
        key=lambda group: group[0],
    )


def viterbi_path(
    indices: list[int],
    node_quality: dict[int, float],
    edges: dict[tuple[int, int], dict[str, Any]],
    edge_support_weight: float,
) -> list[int]:
    scores = {}
    previous = {}
    for position, index in enumerate(indices):
        best_score = float(node_quality[index])
        best_previous = None
        for prior in indices[:position]:
            edge = edges.get((prior, index))
            if edge is None or not edge["allowed"]:
                continue
            candidate = (
                scores[prior]
                + float(node_quality[index])
                + edge_support_weight * float(edge["quality"])
            )
            if candidate > best_score:
                best_score = candidate
                best_previous = prior
        scores[index] = best_score
        previous[index] = best_previous
    cursor = max(indices, key=lambda index: (scores[index], -index))
    path = []
    while cursor is not None:
        path.append(cursor)
        cursor = previous[cursor]
    return list(reversed(path))


def select_chains(
    nodes: list[dict[str, Any]],
    edges: dict[tuple[int, int], dict[str, Any]],
    config: dict[str, Any],
) -> list[list[int]]:
    core = config["core"]
    eligible_indices = [node["index"] for node in nodes if node["eligible"]]
    components = connected_components(eligible_indices, edges)
    node_quality = {node["index"]: node["quality"] for node in nodes}
    paths = []
    for component in components:
        path = viterbi_path(
            component,
            node_quality,
            edges,
            float(core["edge_support_weight"]),
        )
        paths.append(path)
    return sorted(paths, key=lambda path: path[0])


def chain_confidence(
    path: list[int],
    nodes_by_index: dict[int, dict[str, Any]],
    edges: dict[tuple[int, int], dict[str, Any]],
) -> float:
    node_values = [nodes_by_index[index]["quality"] for index in path]
    edge_values = [
        edges[(left, right)]["quality"]
        for left, right in zip(path[:-1], path[1:])
        if (left, right) in edges
    ]
    node_mean = float(np.mean(node_values)) if node_values else 0.0
    edge_mean = float(np.mean(edge_values)) if edge_values else node_mean
    return round(float(np.clip(0.6 * node_mean + 0.4 * edge_mean, 0.0, 1.0)), 6)


def build_set_candidate(
    set_id: int,
    path: list[int],
    nodes: list[dict[str, Any]],
    bottom_states: list[dict[str, Any]],
    edges: dict[tuple[int, int], dict[str, Any]],
    period_frames: float,
    periodicity: float,
    action: dict[str, Any],
    phase_signal: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    nodes_by_index = {node["index"]: node for node in nodes}
    selected_nodes = [nodes_by_index[index] for index in path]
    selected_tops = [node["top"] for node in selected_nodes]
    confidence = chain_confidence(path, nodes_by_index, edges)
    reps = phase_states.build_rep_candidates(
        selected_tops,
        bottom_states,
        phase_signal=phase_signal,
        rep_unit=action.get("rep_unit", "cycle"),
        period_frames=period_frames,
        periodicity=periodicity,
    )
    for rep_id, rep in enumerate(reps, start=1):
        rep["rep_id"] = rep_id
        in_set = float(rep["confidence"]["in_set"]) * confidence
        rep["confidence"]["in_set"] = round(float(np.clip(in_set, 0.0, 1.0)), 6)
        rep["confidence"]["combined"] = round(
            float(
                np.clip(
                    float(rep["confidence"]["complete"]) * in_set,
                    0.0,
                    1.0,
                )
            ),
            6,
        )
        rep["quality_flags"] = sorted(
            set(rep.get("quality_flags") or []) | {"selected_chain"}
        )
    start_frame = int(reps[0]["start_frame"]) if reps else int(selected_tops[0]["start_frame"])
    end_frame = int(reps[-1]["end_frame"]) if reps else int(selected_tops[-1]["end_frame"])
    return (
        {
            "set_id": set_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "status": "complete_set",
            "chain_confidence": confidence,
            "is_default": False,
            "reps": reps,
            "rejected_candidates": [],
            "non_rep_intervals": [],
        },
        selected_nodes,
    )


def _endpoint_timing_features(
    path: list[int],
    nodes_by_index: dict[int, dict[str, Any]],
    period_frames: float,
    role: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    intervals = [
        nodes_by_index[right]["event_frame"]
        - nodes_by_index[left]["event_frame"]
        for left, right in zip(path[:-1], path[1:])
    ]
    context_count = int(policy.get("context_intervals", 4))
    minimum_context = int(policy.get("minimum_context_intervals", 3))
    if role == "leading":
        endpoint_index = path[0]
        endpoint_gap = intervals[0]
        context = intervals[1 : 1 + context_count]
    else:
        endpoint_index = path[-1]
        endpoint_gap = intervals[-1]
        context = intervals[max(0, len(intervals) - 1 - context_count) : -1]
    node = nodes_by_index[endpoint_index]
    context_median = float(np.median(context)) if context else 0.0
    context_mad = (
        float(
            np.median([abs(value - context_median) for value in context])
            / context_median
        )
        if context and context_median > 0
        else float("inf")
    )
    local_gap_ratio = (
        float(endpoint_gap) / context_median if context_median > 0 else None
    )
    global_gap_ratio = (
        float(endpoint_gap) / period_frames if period_frames > 0 else None
    )
    context_stable = bool(
        len(context) >= minimum_context
        and context_mad
        <= float(policy.get("maximum_context_relative_mad", 0.25))
    )
    partial = node["top"].get("kind") != "interior"
    retain_global = [
        float(value)
        for value in policy.get(
            "retain_partial_gap_period_range", [0.9, 1.1]
        )
    ]
    retain_local = [
        float(value)
        for value in policy.get(
            "retain_partial_local_gap_ratio_range", [0.72, 1.08]
        )
    ]
    timing_plausible = bool(
        partial
        and context_stable
        and global_gap_ratio is not None
        and retain_global[0] <= global_gap_ratio <= retain_global[1]
        and local_gap_ratio is not None
        and retain_local[0] <= local_gap_ratio <= retain_local[1]
    )
    short_partial_outlier = bool(
        partial
        and context_stable
        and len(path) - 1
        >= int(policy.get("minimum_retained_events", 4))
        and local_gap_ratio is not None
        and local_gap_ratio
        < float(policy.get("short_partial_local_gap_ratio", 0.75))
    )
    long_partial_outlier = bool(
        partial
        and context_stable
        and len(path) - 1
        >= int(policy.get("minimum_retained_events", 4))
        and local_gap_ratio is not None
        and local_gap_ratio
        > float(policy.get("long_partial_local_gap_ratio", 1.45))
    )
    return {
        "role": role,
        "frame": int(node["event_frame"]),
        "kind": node["top"].get("kind"),
        "partial": partial,
        "stable_fraction": round(
            float(node["top"].get("stable_fraction") or 0.0), 6
        ),
        "bottom_stability": round(
            float(node.get("bottom_stability") or 0.0), 6
        ),
        "scale_support": int(node.get("scale_support") or 0),
        "multi_scale_consensus": bool(node.get("multi_scale_consensus")),
        "endpoint_gap_frames": int(endpoint_gap),
        "endpoint_gap_ratio": (
            round(float(global_gap_ratio), 6)
            if global_gap_ratio is not None
            else None
        ),
        "endpoint_local_gap_ratio": (
            round(float(local_gap_ratio), 6)
            if local_gap_ratio is not None
            else None
        ),
        "context_intervals_frames": [int(value) for value in context],
        "context_interval_median": round(context_median, 6),
        "context_interval_mad": (
            round(context_mad, 6) if math.isfinite(context_mad) else None
        ),
        "neighbor_gap_ratios": [
            round(float(value) / context_median, 6)
            for value in context
            if context_median > 0
        ],
        "context_stable": context_stable,
        "timing_plausible": timing_plausible,
        "short_partial_outlier": short_partial_outlier,
        "long_partial_outlier": long_partial_outlier,
    }


def evaluate_endpoint_timing(
    path: list[int],
    nodes_by_index: dict[int, dict[str, Any]],
    period_frames: float,
    chain_policy: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Evaluate both chain ends against the group's top-to-top timing."""

    if chain_policy is None:
        return None
    policy = chain_policy.get("endpoint_timing_model") or {}
    if not bool(policy.get("enabled")) or len(path) < 2 or period_frames <= 0:
        return None
    endpoints = {
        role: _endpoint_timing_features(
            path, nodes_by_index, period_frames, role, policy
        )
        for role in ("leading", "trailing")
    }
    prune_candidates = []
    short_threshold = float(policy.get("short_partial_local_gap_ratio", 0.75))
    long_threshold = float(policy.get("long_partial_local_gap_ratio", 1.45))
    for endpoint in endpoints.values():
        local_ratio = endpoint["endpoint_local_gap_ratio"]
        if endpoint["short_partial_outlier"]:
            prune_candidates.append(
                (
                    short_threshold - float(local_ratio),
                    "short_partial_endpoint",
                    endpoint,
                )
            )
        elif endpoint["long_partial_outlier"]:
            prune_candidates.append(
                (
                    float(local_ratio) - long_threshold,
                    "long_partial_endpoint",
                    endpoint,
                )
            )
    if prune_candidates:
        _, reason, selected = max(
            prune_candidates,
            key=lambda item: (item[0], -float(item[2]["stable_fraction"])),
        )
        return {
            "decision": "prune",
            "reason": reason,
            "prune_role": selected["role"],
            "period_frames": round(float(period_frames), 6),
            "endpoints": endpoints,
        }
    retained = [
        endpoint
        for endpoint in endpoints.values()
        if endpoint["timing_plausible"]
    ]
    return {
        "decision": "keep",
        "reason": (
            "timing_plausible_endpoint" if retained else "no_timing_outlier"
        ),
        "prune_role": None,
        "period_frames": round(float(period_frames), 6),
        "endpoints": endpoints,
    }


def endpoint_timing_pruning(
    path: list[int],
    nodes_by_index: dict[int, dict[str, Any]],
    period_frames: float,
    chain_policy: dict[str, Any] | None,
) -> tuple[list[int], dict[str, Any]] | None:
    """Trim one partial endpoint whose cycle is too short for the local tempo."""

    timing = evaluate_endpoint_timing(
        path, nodes_by_index, period_frames, chain_policy
    )
    if timing is None or timing["decision"] != "prune":
        return None
    role = timing["prune_role"]
    endpoint = timing["endpoints"][role]
    removed_index = path[0] if role == "leading" else path[-1]
    pruned_path = path[1:] if role == "leading" else path[:-1]
    record = {
        "role": role,
        "frame": int(endpoint["frame"]),
        "kind": endpoint["kind"],
        "stable_fraction": float(endpoint["stable_fraction"]),
        "scale_support": int(endpoint["scale_support"]),
        "reason": timing["reason"],
        "endpoint_gap_ratio": endpoint["endpoint_gap_ratio"],
        "endpoint_local_gap_ratio": endpoint["endpoint_local_gap_ratio"],
        "context_interval_median": endpoint["context_interval_median"],
        "context_interval_mad": endpoint["context_interval_mad"],
        "removed_index": int(removed_index),
    }
    return pruned_path, record


def low_sampling_endpoint_pruning(
    path: list[int],
    nodes_by_index: dict[int, dict[str, Any]],
    chain_policy: dict[str, Any] | None,
    period_frames: float | None = None,
) -> tuple[list[int], dict[str, Any]] | None:
    """Trim one weak multi-scale endpoint under low sampling only."""

    if chain_policy is None:
        return None
    policy = chain_policy.get("low_sampling_endpoint_pruning") or {}
    if not bool(policy.get("enabled")):
        return None
    minimum_retained = int(policy.get("minimum_retained_events") or 4)
    if len(path) - 1 < minimum_retained:
        return None
    left = nodes_by_index[path[0]]
    right = nodes_by_index[path[-1]]
    endpoint_nodes = (left, right)
    if any(
        not bool(node.get("multi_scale_consensus"))
        or bool(
            (node["top"].get("evidence") or {}).get("half_cycle_recovered")
        )
        for node in endpoint_nodes
    ):
        return None
    if (
        left["top"].get("kind") == "interior"
        and right["top"].get("kind") == "interior"
    ):
        return None
    removed = min(
        endpoint_nodes,
        key=lambda node: (
            float(node["top"].get("stable_fraction") or 0.0),
            float(node["quality"]),
        ),
    )
    if float(removed["top"].get("stable_fraction") or 0.0) > float(
        policy.get("maximum_pruned_endpoint_stability", 0.6)
    ):
        return None
    if period_frames is not None:
        timing = evaluate_endpoint_timing(
            path, nodes_by_index, period_frames, chain_policy
        )
        role = "leading" if removed is left else "trailing"
        if (
            timing is not None
            and timing["endpoints"][role]["timing_plausible"]
        ):
            return None
    pruned_path = path[1:] if removed is left else path[:-1]
    record = {
        "role": "leading" if removed is left else "trailing",
        "frame": int(removed["event_frame"]),
        "kind": removed["top"].get("kind"),
        "stable_fraction": round(
            float(removed["top"].get("stable_fraction") or 0.0), 6
        ),
        "scale_support": int(removed.get("scale_support") or 0),
    }
    return pruned_path, record


def low_amplitude_context_bridge(
    nodes: list[dict[str, Any]],
    edges: dict[tuple[int, int], dict[str, Any]],
    period_frames: float,
    config: dict[str, Any],
    chain_policy: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Extend exactly one viable chain through a low-amplitude interior node.

    A low-amplitude node can be a real, multi-scale-consistent event that was
    rejected by the amplitude-only node gate. It is safe to restore only when
    it extends one already viable chain. If both sides become viable after
    adding the node, the node may be joining two independent sets, so it is
    left rejected rather than merging them.
    """

    if chain_policy is None:
        return []
    policy = chain_policy.get("low_amplitude_context_bridge") or {}
    if not bool(policy.get("enabled")):
        return []
    core = config["core"]
    minimum_relative_amplitude = float(core["minimum_relative_amplitude"])
    minimum_bridge_amplitude = float(
        policy.get("minimum_relative_amplitude", 0.2)
    )
    minimum_chain_events = int(core["minimum_chain_reps"]) + 1
    minimum_top_stability = float(policy.get("minimum_top_stability", 0.4))
    minimum_bottom_stability = float(
        policy.get("minimum_bottom_stability", 0.2)
    )
    minimum_support_fraction = float(
        policy.get("minimum_scale_support_fraction", 0.5)
    )
    minimum_bridge_confidence = float(
        policy.get("minimum_bridge_confidence", 0.45)
    )
    eligible_indices = [node["index"] for node in nodes if node["eligible"]]
    components = connected_components(eligible_indices, edges)
    component_by_index = {
        index: component for component in components for index in component
    }

    def viable_with_candidate(component: list[int] | None) -> bool:
        return bool(
            component is not None
            and len(component) + 1 >= minimum_chain_events
        )

    records = []
    for node in nodes:
        if node["eligible"] or node.get("bottom") is None:
            continue
        top = node["top"]
        if top.get("kind") != "interior":
            continue
        evidence = top.get("evidence") or {}
        if not bool(evidence.get("multi_scale_consensus")):
            continue
        if bool(evidence.get("half_cycle_recovered")):
            continue
        relative_amplitude = float(node.get("relative_amplitude") or 0.0)
        if not (
            minimum_bridge_amplitude
            <= relative_amplitude
            < minimum_relative_amplitude
        ):
            continue
        if float(node.get("top_stability") or 0.0) < minimum_top_stability:
            continue
        if (
            float(node.get("bottom_stability") or 0.0)
            < minimum_bottom_stability
        ):
            continue
        support_fraction = float(evidence.get("scale_support_fraction") or 0.0)
        if support_fraction < minimum_support_fraction:
            continue

        left_index = node["index"] - 1 if node["index"] > 0 else None
        right_index = (
            node["index"] + 1 if node["index"] + 1 < len(nodes) else None
        )
        left_component = component_by_index.get(left_index)
        right_component = component_by_index.get(right_index)
        left_viable = viable_with_candidate(left_component)
        right_viable = viable_with_candidate(right_component)
        if left_viable and right_viable:
            continue
        if not left_viable and not right_viable:
            continue
        primary_side = "left" if left_viable else "right"
        primary_component = left_component if left_viable else right_component
        secondary_component = right_component if left_viable else left_component
        primary_index = left_index if left_viable else right_index
        if primary_index is None:
            continue
        primary_edge = build_edge(
            nodes[primary_index],
            node,
            period_frames,
            config,
        )
        if primary_edge is None or not primary_edge["allowed"]:
            continue
        allowed_neighbors = set(primary_component or [])
        if secondary_component is not None:
            allowed_neighbors.update(secondary_component)

        amplitude_score = min(
            1.0,
            relative_amplitude / max(minimum_relative_amplitude, 1e-6),
        )
        stability_score = (
            0.6 * float(node.get("top_stability") or 0.0)
            + 0.4 * float(node.get("bottom_stability") or 0.0)
        )
        bridge_confidence = float(
            np.clip(
                0.45 * amplitude_score
                + 0.35 * stability_score
                + 0.20 * support_fraction,
                0.0,
                1.0,
            )
        )
        if bridge_confidence < minimum_bridge_confidence:
            continue

        node["eligible"] = True
        node["low_amplitude_context_bridged"] = True
        node["bridge_confidence"] = round(bridge_confidence, 6)
        node["context_bridge_allowed_neighbors"] = sorted(allowed_neighbors)
        neighbor_frames = [
            int(nodes[index]["event_frame"])
            for index in (left_index, right_index)
            if index is not None
        ]
        records.append(
            {
                "index": int(node["index"]),
                "frame": int(node["event_frame"]),
                "relative_amplitude": round(relative_amplitude, 6),
                "relative_amplitude_threshold": round(
                    minimum_relative_amplitude, 6
                ),
                "top_stability": round(
                    float(node.get("top_stability") or 0.0), 6
                ),
                "bottom_stability": round(
                    float(node.get("bottom_stability") or 0.0), 6
                ),
                "scale_support": int(node.get("scale_support") or 0),
                "scale_support_fraction": round(support_fraction, 6),
                "primary_side": primary_side,
                "primary_component_indices": [
                    int(index) for index in primary_component or []
                ],
                "secondary_component_indices": [
                    int(index) for index in secondary_component or []
                ],
                "neighbor_frames": neighbor_frames,
                "primary_edge_gap_ratio": round(
                    float(primary_edge["gap_ratio"]), 6
                ),
                "primary_edge_quality": round(
                    float(primary_edge["quality"]), 6
                ),
                "bridge_confidence": round(bridge_confidence, 6),
            }
        )
    return records


def build_rejected_candidates(
    nodes: list[dict[str, Any]],
    selected_indices: set[int],
    accepted_paths: list[list[int]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    in_any_path = {index for path in accepted_paths for index in path}
    rows = []
    for node in nodes:
        if node["index"] in selected_indices:
            continue
        if not node["eligible"]:
            reasons = ["shape_mismatch"]
        elif node["index"] in in_any_path:
            reasons = ["outside_set"]
        else:
            reasons = ["no_confident_set"]
        rows.append(
            {
                "candidate_id": f"top_state_{node['index'] + 1}",
                "start_frame": int(node["top"]["start_frame"]),
                "end_frame": int(node["top"]["end_frame"]),
                "reasons": reasons,
                "confidence": node["initial_confidence"],
                "evidence": {
                    "event_frame": int(node["event_frame"]),
                    "relative_amplitude": round(
                        float(node["relative_amplitude"]), 6
                    ),
                    "eligible": bool(node["eligible"]),
                },
            }
        )
    return rows


def build_non_rep_intervals(
    sets: list[dict[str, Any]],
    frame_count: int,
) -> list[dict[str, Any]]:
    if not sets or frame_count <= 0:
        return []
    ordered = sorted(sets, key=lambda item: item["start_frame"])
    intervals = []
    first_start = max(0, int(ordered[0]["start_frame"]))
    if first_start > 0:
        intervals.append(
            {
                "start_frame": 0,
                "end_frame": first_start - 1,
                "kind": "setup",
                "confidence": 0.8,
            }
        )
    for left, right in zip(ordered[:-1], ordered[1:]):
        start = int(left["end_frame"]) + 1
        end = int(right["start_frame"]) - 1
        if end >= start:
            intervals.append(
                {
                    "start_frame": start,
                    "end_frame": end,
                    "kind": "rest",
                    "confidence": 0.75,
                }
            )
    last_end = min(frame_count - 1, int(ordered[-1]["end_frame"]))
    if last_end < frame_count - 1:
        intervals.append(
            {
                "start_frame": last_end + 1,
                "end_frame": frame_count - 1,
                "kind": "recovery",
                "confidence": 0.8,
            }
        )
    return intervals


def mark_default_set(sets: list[dict[str, Any]], config: dict[str, Any]) -> None:
    if not sets:
        return
    default = max(
        sets,
        key=lambda item: (
            len(item["reps"]),
            float(item["chain_confidence"]),
            -int(item["start_frame"]),
        ),
    )
    for set_candidate in sets:
        set_candidate["is_default"] = set_candidate is default


def analyze_phase_states(
    clip_id: str,
    top_states: list[dict[str, Any]],
    bottom_states: list[dict[str, Any]],
    *,
    period_frames: float,
    periodicity: float,
    action: dict[str, Any],
    phase_signal: str,
    frame_count: int,
    config: dict[str, Any],
    chain_policy: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tail_policy, tail_scope = tail_anchor_policy(config)
    nodes = build_candidate_nodes(
        top_states, bottom_states, period_frames, config
    )
    nodes_by_index = {node["index"]: node for node in nodes}
    local_extrema_bridged_events = local_extrema_stability_bridge(
        nodes,
        bottom_states,
        period_frames,
        config,
        chain_policy,
    )
    edges = build_edges(
        nodes, period_frames, config, chain_policy=chain_policy
    )
    context_bridged_events = low_amplitude_context_bridge(
        nodes,
        edges,
        period_frames,
        config,
        chain_policy,
    )
    if context_bridged_events:
        edges = build_edges(
            nodes, period_frames, config, chain_policy=chain_policy
        )
    paths = select_chains(nodes, edges, config)
    sets = []
    selected_nodes = []
    accepted_paths = []
    endpoint_pruned_events = []
    endpoint_timing_decisions = []
    endpoint_timing_pruned_events = []
    minimum_chain_reps = int(config["core"]["minimum_chain_reps"])
    minimum_consensus_events = (
        int(chain_policy.get("minimum_chain_consensus_events", 0))
        if chain_policy is not None
        else 0
    )
    for path in paths:
        consensus_event_count = sum(
            bool(nodes_by_index[index]["multi_scale_consensus"])
            for index in path
        )
        if consensus_event_count < minimum_consensus_events:
            continue
        set_candidate, path_nodes = build_set_candidate(
            len(sets) + 1,
            path,
            nodes,
            bottom_states,
            edges,
            period_frames,
            periodicity,
            action,
            phase_signal,
        )
        timing = evaluate_endpoint_timing(
            path, nodes_by_index, period_frames, chain_policy
        )
        accepted_pruned_record = None
        accepted_timing_record = None
        pruned = endpoint_timing_pruning(
            path, nodes_by_index, period_frames, chain_policy
        )
        if pruned is not None:
            pruned_path, pruned_record = pruned
            path = pruned_path
            set_candidate, path_nodes = build_set_candidate(
                len(sets) + 1,
                path,
                nodes,
                bottom_states,
                edges,
                period_frames,
                periodicity,
                action,
                phase_signal,
            )
            accepted_timing_record = pruned_record
        else:
            pruned = low_sampling_endpoint_pruning(
                path, nodes_by_index, chain_policy, period_frames
            )
            if pruned is not None:
                pruned_path, pruned_record = pruned
                path = pruned_path
                set_candidate, path_nodes = build_set_candidate(
                    len(sets) + 1,
                    path,
                    nodes,
                    bottom_states,
                    edges,
                    period_frames,
                    periodicity,
                    action,
                    phase_signal,
                )
                accepted_pruned_record = pruned_record
        if len(set_candidate["reps"]) < minimum_chain_reps:
            continue
        closure_edges = [
            edges[(left, right)]
            for left, right in zip(path[:-1], path[1:])
            if edges[(left, right)].get("endpoint_closure")
        ]
        set_candidate["consensus_event_count"] = consensus_event_count
        set_candidate["endpoint_closure_count"] = len(closure_edges)
        if closure_edges and any(
            rep.get("status") == "endpoint_partial"
            for rep in set_candidate["reps"]
        ):
            set_candidate["status"] = "partial_set"
        sets.append(set_candidate)
        accepted_paths.append(path)
        selected_nodes.extend(path_nodes)
        if accepted_pruned_record is not None:
            endpoint_pruned_events.append(accepted_pruned_record)
        if accepted_timing_record is not None:
            endpoint_timing_pruned_events.append(accepted_timing_record)
        if timing is not None:
            endpoint_timing_decisions.append(
                {
                    "set_id": int(set_candidate["set_id"]),
                    "decision": timing["decision"],
                    "reason": timing["reason"],
                    "prune_role": timing["prune_role"],
                    "period_frames": timing["period_frames"],
                    "endpoints": timing["endpoints"],
                }
            )
    for path in accepted_paths:
        terminal = nodes_by_index[path[-1]]
        terminal["output_frame"] = terminal_anchor_frame(terminal, tail_policy)
        terminal["tail_anchor_applied"] = True
        for left, right in zip(path[:-1], path[1:]):
            edge = edges[(left, right)]
            if edge.get("endpoint_closure"):
                nodes_by_index[left]["endpoint_closure"] = True
                nodes_by_index[right]["endpoint_closure"] = True
            if edge.get("local_period_gap"):
                nodes_by_index[left]["local_period_gap"] = True
                nodes_by_index[right]["local_period_gap"] = True
    mark_default_set(sets, config)
    selected_indices = {node["index"] for node in selected_nodes}
    rejected = build_rejected_candidates(
        nodes, selected_indices, accepted_paths, config
    )
    non_rep_intervals = build_non_rep_intervals(sets, frame_count)
    status = "ok" if sets else "no_confident_set"
    selected_top_events = sorted(
        [
            {
                "frame": int(node["output_frame"]),
                "graph_frame": int(node["event_frame"]),
                "is_terminal": bool(node.get("tail_anchor_applied")),
                "anchor_policy": (
                    tail_policy
                    if node.get("tail_anchor_applied")
                    else "plateau_midpoint"
                ),
                "scale_support": int(node.get("scale_support") or 0),
                "multi_scale_consensus": bool(
                    node.get("multi_scale_consensus")
                ),
                "low_amplitude_context_bridged": bool(
                    node.get("low_amplitude_context_bridged")
                ),
                "bridge_confidence": node.get("bridge_confidence"),
                "local_extrema_stable": bool(
                    (node["top"].get("evidence") or {}).get(
                        "local_extrema_stable"
                    )
                ),
                "local_extrema_relative_amplitude": (
                    node["top"].get("evidence") or {}
                ).get("local_extrema_relative_amplitude"),
                "local_period_gap": bool(node.get("local_period_gap")),
                "endpoint_closure": bool(node.get("endpoint_closure")),
                "confidence": node["initial_confidence"],
                "set_ids": [
                    set_candidate["set_id"]
                    for set_candidate in sets
                    if set_candidate["start_frame"]
                    <= node["output_frame"]
                    <= set_candidate["end_frame"]
                ],
            }
            for node in selected_nodes
        ],
        key=lambda item: item["frame"],
    )
    analysis = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "clip_id": clip_id,
        "algorithm": {"name": "repchain", "version": "v2"},
        "anchor_policy": "plateau_midpoint",
        "event_anchor_policy": "plateau_midpoint_with_terminal_early",
        "status": status,
        "sets": sets,
        "rejected_candidates": rejected,
        "non_rep_intervals": non_rep_intervals,
        "diagnostics": {
            "selected_phase_signal": phase_signal,
            "phase_signal_candidates": [phase_signal],
            "chain_source": "phase_state_viterbi",
            "graph_anchor_policy": "plateau_midpoint",
            "tail_anchor_policy": tail_policy,
            "tail_anchor_scope": tail_scope,
            "selected_top_events": selected_top_events,
            "period_frames": round(float(period_frames), 6),
            "periodicity": round(float(periodicity), 6),
            "chain_policy": chain_policy,
            "consensus_event_count": sum(
                bool(node.get("multi_scale_consensus"))
                for node in selected_nodes
            ),
            "endpoint_closure_edges": [
                {
                    "left_index": int(left),
                    "right_index": int(right),
                    "left_frame": int(nodes_by_index[left]["event_frame"]),
                    "right_frame": int(nodes_by_index[right]["event_frame"]),
                    "gap_ratio": round(
                        float(edges[(left, right)]["gap_ratio"]), 6
                    ),
                    "quality": round(
                        float(edges[(left, right)]["quality"]), 6
                    ),
                }
                for path in accepted_paths
                for left, right in zip(path[:-1], path[1:])
                if edges[(left, right)].get("endpoint_closure")
            ],
            "local_period_gap_edges": [
                {
                    "left_index": int(left),
                    "right_index": int(right),
                    "left_frame": int(nodes_by_index[left]["event_frame"]),
                    "right_frame": int(nodes_by_index[right]["event_frame"]),
                    "gap_frames": int(edges[(left, right)]["gap_frames"]),
                    "global_period_frames": float(
                        edges[(left, right)]["global_period_frames"]
                    ),
                    "global_gap_ratio": float(
                        edges[(left, right)]["global_gap_ratio"]
                    ),
                    "local_period_frames": float(
                        edges[(left, right)]["local_period_frames"]
                    ),
                    "local_gap_ratio": float(
                        edges[(left, right)]["local_gap_ratio"]
                    ),
                    "local_period_context_intervals": edges[(left, right)][
                        "local_period_context_intervals"
                    ],
                    "local_period_relative_mad": float(
                        edges[(left, right)]["local_period_relative_mad"]
                    ),
                    "quality": round(
                        float(edges[(left, right)]["quality"]), 6
                    ),
                }
                for path in accepted_paths
                for left, right in zip(path[:-1], path[1:])
                if edges[(left, right)].get("local_period_gap")
            ],
            "low_sampling_endpoint_pruned_events": endpoint_pruned_events,
            "endpoint_timing_decisions": endpoint_timing_decisions,
            "endpoint_timing_pruned_events": endpoint_timing_pruned_events,
            "local_extrema_stability_bridged_events": (
                local_extrema_bridged_events
            ),
            "low_amplitude_context_bridged_events": context_bridged_events,
        },
    }
    meta = {
        "selected_top_events": selected_top_events,
        "selected_indices": selected_indices,
        "rejected_candidates": rejected,
        "period_frames": period_frames,
        "periodicity": periodicity,
        "chain_policy": chain_policy,
    }
    return analysis, meta


def legacy_sets_from_analysis(
    clip_id: str,
    legacy_analysis: dict[str, Any],
    frame_count: int,
    reason: str,
    frame_offset: int = 0,
) -> dict[str, Any]:
    segmentation = legacy_analysis.get("segmentation") or {}
    reps = []
    for index, source in enumerate(segmentation.get("reps") or [], start=1):
        if source.get("status", "complete") != "complete":
            continue
        confidence = float(source.get("confidence") or 0.5)
        confidence = float(np.clip(confidence, 0.0, 1.0))
        reps.append(
            {
                "rep_id": index,
                "start_frame": max(
                    0, int(source["start_frame"]) - int(frame_offset)
                ),
                "end_frame": min(
                    frame_count - 1,
                    int(source["end_frame"]) - int(frame_offset),
                ),
                "top": None,
                "bottom": None,
                "phase_signal": "legacy_t1_4",
                "rep_unit": "cycle",
                "status": "complete",
                "partial_side": None,
                "confidence": {
                    "complete": round(confidence, 6),
                    "in_set": round(confidence, 6),
                    "combined": round(confidence, 6),
                },
                "quality_flags": ["legacy_fallback"],
            }
        )
    sets = []
    if reps:
        chain_confidence = round(
            float(
                np.mean(
                    [rep["confidence"]["combined"] for rep in reps]
                )
            ),
            6,
        )
        sets.append(
            {
                "set_id": 1,
                "start_frame": min(rep["start_frame"] for rep in reps),
                "end_frame": max(rep["end_frame"] for rep in reps),
                "status": "complete_set",
                "chain_confidence": chain_confidence,
                "is_default": True,
                "reps": reps,
                "rejected_candidates": [],
                "non_rep_intervals": [],
            }
        )
    non_rep = build_non_rep_intervals(sets, frame_count)
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "clip_id": clip_id,
        "algorithm": {"name": "repchain", "version": "v2"},
        "anchor_policy": "plateau_midpoint",
        "status": "ok" if sets else "no_confident_set",
        "sets": sets,
        "rejected_candidates": [],
        "non_rep_intervals": non_rep,
        "diagnostics": {
            "selected_phase_signal": None,
            "phase_signal_candidates": [],
            "chain_source": "legacy_fallback",
            "fallback_reason": reason,
            "selected_top_events": [],
        },
    }


def analyze_clip_signals(
    clip: dict[str, Any],
    signals: dict[str, np.ndarray],
    action: dict[str, Any],
    *,
    prior_signal: str | None,
    phase_policy: str,
    chain_config: dict[str, Any],
    legacy_analysis: dict[str, Any] | None = None,
    allow_legacy_fallback: bool = False,
    quality_gate: dict[str, Any] | None = None,
    detector: dict[str, Any] | None = None,
    phase_resolution: dict[str, Any] | None = None,
    phase_state_override: dict[str, Any] | None = None,
    chain_policy: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frame_count = (
        int(next(iter(signals.values())).shape[0]) if signals else 0
    )
    if phase_resolution is None:
        phase_signal = phase_states.resolve_phase_signal(
            signals,
            float(clip["fps"]),
            action,
            phase_policy,
            prior_signal=prior_signal,
            detector=detector,
        )
        effective_detector = detector
        phase_evidence = None
    else:
        phase_signal = phase_resolution.get("selected_phase_signal")
        effective_detector = phase_resolution.get("detector") or detector
        phase_evidence = phase_resolution.get("evidence") or phase_resolution
        if phase_resolution.get("status") == "ambiguous":
            analysis = {
                "schema_version": ANALYSIS_SCHEMA_VERSION,
                "clip_id": clip["clip_id"],
                "algorithm": {"name": "repchain", "version": "v2"},
                "anchor_policy": "plateau_midpoint",
                "status": "phase_ambiguous",
                "motion_status": "unknown",
                "sets": [],
                "rejected_candidates": [],
                "non_rep_intervals": [],
                "diagnostics": {
                    "selected_phase_signal": phase_signal,
                    "phase_signal_candidates": [],
                    "chain_source": "phase_resolution_ambiguous",
                    "phase_signal_evidence": phase_evidence,
                },
            }
            analysis = quality_gates.apply_quality_gates(
                analysis,
                [quality_gate] if quality_gate is not None else None,
            )
            return analysis, {
                "selected_top_events": [],
                "selected_indices": set(),
                "rejected_candidates": [],
                "period_frames": None,
                "periodicity": None,
                "phase_signal": phase_signal,
                "phase_resolution_status": "ambiguous",
            }
    if phase_signal is None:
        evidence = phase_states.assess_motion_signals(
            signals,
            float(clip["fps"]),
            action,
            effective_detector,
        )
        status = (
            "no_motion"
            if evidence.get("status") != "ok"
            else "no_confident_set"
        )
        analysis = {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "clip_id": clip["clip_id"],
            "algorithm": {"name": "repchain", "version": "v2"},
            "anchor_policy": "plateau_midpoint",
            "status": status,
            "motion_status": evidence.get("status"),
            "sets": [],
            "rejected_candidates": [],
            "non_rep_intervals": [],
            "diagnostics": {
                "selected_phase_signal": None,
                "phase_signal_candidates": [],
                "chain_source": "no_phase_signal",
                "motion_evidence": evidence,
                "phase_signal_evidence": phase_evidence,
            },
        }
        analysis = quality_gates.apply_quality_gates(
            analysis,
            [quality_gate] if quality_gate is not None else None,
        )
        return analysis, {
            "selected_top_events": [],
            "selected_indices": set(),
            "rejected_candidates": [],
            "period_frames": None,
            "periodicity": None,
        }
    if phase_state_override is None:
        states = phase_states.detect_phase_states(
            signals[phase_signal],
            float(clip["fps"]),
            action,
            phase_signal,
            "plateau_midpoint",
            effective_detector,
        )
    else:
        states = phase_state_override
    minimum_periodicity = float(
        chain_config["core"]["low_evidence"]["minimum_phase_periodicity"]
    )
    periodicity = float(states.get("periodicity") or 0.0)
    if states.get("status") == "no_motion":
        analysis = {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "clip_id": clip["clip_id"],
            "algorithm": {"name": "repchain", "version": "v2"},
            "anchor_policy": "plateau_midpoint",
            "status": "no_motion",
            "motion_status": "no_motion",
            "sets": [],
            "rejected_candidates": [],
            "non_rep_intervals": [],
            "diagnostics": {
                "selected_phase_signal": phase_signal,
                "phase_signal_candidates": [],
                "chain_source": "motion_gate",
                "motion_reasons": (
                    (states.get("motion_evidence") or {}).get("reasons") or []
                ),
                "motion_evidence": states.get("motion_evidence"),
                "phase_signal_evidence": phase_evidence,
            },
        }
        analysis = quality_gates.apply_quality_gates(
            analysis,
            [quality_gate] if quality_gate is not None else None,
        )
        return analysis, {
            "selected_top_events": [],
            "selected_indices": set(),
            "rejected_candidates": [],
            "period_frames": states.get("period_frames"),
            "periodicity": periodicity,
            "phase_signal": phase_signal,
            "states": states,
            "phase_resolution_status": (
                phase_resolution.get("status")
                if phase_resolution is not None
                else None
            ),
        }
    if (
        allow_legacy_fallback
        and legacy_analysis is not None
        and periodicity < minimum_periodicity
    ):
        analysis = legacy_sets_from_analysis(
            clip["clip_id"],
            legacy_analysis,
            frame_count,
            "phase_periodicity_below_threshold",
            int((clip.get("trim") or {}).get("start_frame", 0)),
        )
        analysis = quality_gates.apply_quality_gates(
            analysis,
            [quality_gate] if quality_gate is not None else None,
        )
        return analysis, {
            "selected_top_events": [],
            "selected_indices": set(),
            "rejected_candidates": [],
            "period_frames": states.get("period_frames"),
            "periodicity": periodicity,
            "fallback_used": True,
        }
    analysis, meta = analyze_phase_states(
        clip["clip_id"],
        states.get("top_states") or [],
        states.get("bottom_states") or [],
        period_frames=float(states.get("period_frames") or 0.0),
        periodicity=periodicity,
        action=action,
        phase_signal=phase_signal,
        frame_count=frame_count,
        config=chain_config,
        chain_policy=states.get("chain_policy") or chain_policy,
    )
    analysis["diagnostics"]["phase_signal_candidates"] = [
        name
        for name in action.get("phase_signal_candidates") or []
        if name in signals
    ] + [
        pair["name"]
        for pair in action.get("phase_signal_pairs") or []
        if pair["name"] in signals
    ]
    analysis["motion_status"] = states.get("motion_status")
    analysis["diagnostics"]["motion_evidence"] = states.get("motion_evidence")
    if phase_evidence is not None:
        analysis["diagnostics"]["phase_signal_evidence"] = phase_evidence
    if phase_state_override is not None:
        analysis["diagnostics"]["chain_source"] = (
            "phase_state_multiscale_consensus"
        )
        analysis["diagnostics"]["event_consensus_evidence"] = states.get(
            "evidence"
        )
    meta["phase_signal"] = phase_signal
    meta["states"] = states
    meta["phase_state_source"] = (
        "multiscale_consensus"
        if phase_state_override is not None
        else "single_scale"
    )
    meta["phase_resolution_status"] = (
        phase_resolution.get("status")
        if phase_resolution is not None
        else None
    )
    analysis = quality_gates.apply_quality_gates(
        analysis,
        [quality_gate] if quality_gate is not None else None,
    )
    if not analysis.get("sets"):
        meta["selected_top_events"] = []
        meta["selected_indices"] = set()
    return analysis, meta


def event_gate_frames(gt_frames: list[int]) -> int:
    return phase_states.event_gate_frames(gt_frames)


def evaluate_fit3d_events(
    gt_frames: list[int],
    selected_top_events: list[dict[str, Any]],
    high_confidence_threshold: float,
) -> dict[str, Any]:
    events = sorted(selected_top_events, key=lambda item: item["frame"])
    pred_frames = [int(event["frame"]) for event in events]
    gate = event_gate_frames(gt_frames)
    matches = phase_states.match_events(gt_frames, pred_frames, gate)
    matched_gt = {gt_index for gt_index, _, _ in matches}
    matched_pred = {pred_index for _, pred_index, _ in matches}
    errors = [error for _, _, error in matches]
    signed_errors = [
        pred_frames[pred_index] - gt_frames[gt_index]
        for gt_index, pred_index, _ in matches
    ]
    high_confidence_fp = 0
    outside_set_high_confidence_fp = 0
    for pred_index, event in enumerate(events):
        if pred_index in matched_pred:
            continue
        if float(event["confidence"]["combined"]) >= high_confidence_threshold:
            high_confidence_fp += 1
            if not event.get("set_ids"):
                outside_set_high_confidence_fp += 1
    return {
        "event_gate_frames": gate,
        "gt_event_count": len(gt_frames),
        "pred_event_count": len(events),
        "tp": len(matches),
        "fp": len(events) - len(matches),
        "fn": len(gt_frames) - len(matches),
        "event_recall": phase_states.ratio(len(matches), len(gt_frames)),
        "event_precision": phase_states.ratio(len(matches), len(events)),
        "event_abs_error_median_frames": phase_states.percentile(errors, 50),
        "event_abs_error_p90_frames": phase_states.percentile(errors, 90),
        "event_signed_error_median_frames": phase_states.percentile(
            signed_errors, 50
        ),
        "first_gt_matched": bool(gt_frames) and 0 in matched_gt,
        "last_gt_matched": bool(gt_frames) and len(gt_frames) - 1 in matched_gt,
        "high_confidence_fp": high_confidence_fp,
        "outside_set_high_confidence_fp": outside_set_high_confidence_fp,
        "abs_errors": errors,
    }


def aggregate_fit3d_rows(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    gt_count = sum(int(row["gt_event_count"]) for row in rows)
    pred_count = sum(int(row["pred_event_count"]) for row in rows)
    tp = sum(int(row["tp"]) for row in rows)
    errors = [error for row in rows for error in row.get("_abs_errors", [])]
    count_errors = [int(row["count_error"]) for row in rows]
    first_recall = phase_states.ratio(
        sum(bool(row["first_gt_matched"]) for row in rows), len(rows)
    )
    last_recall = phase_states.ratio(
        sum(bool(row["last_gt_matched"]) for row in rows), len(rows)
    )
    precision = phase_states.ratio(tp, pred_count)
    recall = phase_states.ratio(tp, gt_count)
    return {
        "clip_count": len(rows),
        "cluster_count": len({row["cluster_id"] for row in rows}),
        "gt_event_count": gt_count,
        "pred_event_count": pred_count,
        "tp": tp,
        "fp": pred_count - tp,
        "fn": gt_count - tp,
        "event_recall": recall,
        "event_precision": precision,
        "event_f1": phase_states.f1_score(precision, recall),
        "first_event_recall": first_recall,
        "last_event_recall": last_recall,
        "event_abs_error_median_frames": phase_states.percentile(errors, 50),
        "event_abs_error_p90_frames": phase_states.percentile(errors, 90),
        "count_mae": round(float(np.mean(np.abs(count_errors))), 6)
        if count_errors
        else None,
        "count_exact_correct": sum(error == 0 for error in count_errors),
        "high_confidence_event_fp": sum(
            int(row["high_confidence_fp"]) for row in rows
        ),
        "outside_set_high_confidence_fp": sum(
            int(row["outside_set_high_confidence_fp"]) for row in rows
        ),
        "clips_with_set": sum(int(row["set_count"]) > 0 for row in rows),
        "clips_with_multiple_sets": sum(int(row["set_count"]) > 1 for row in rows),
        "no_confident_set_clips": sum(
            row["status"] == "no_confident_set" for row in rows
        ),
        "set_count_distribution": dict(
            Counter(int(row["set_count"]) for row in rows)
        ),
    }


def load_fit3d_clips(
    sources: dict[str, Any],
) -> list[dict[str, Any]]:
    clips = [
        clip
        for clip in sources.get("clips", [])
        if clip.get("dataset") == DEV_DATASET
        and clip.get("split_role") == "dev"
        and clip.get("has_gt")
        and not clip.get("smoke_only")
    ]
    if len(clips) != EXPECTED_FIT3D_DEV_CLIPS:
        raise ValueError(
            f"expected {EXPECTED_FIT3D_DEV_CLIPS} Fit3D dev clips, found {len(clips)}"
        )
    clusters = {clip["cluster_id"] for clip in clips}
    if len(clusters) != EXPECTED_FIT3D_DEV_CLUSTERS:
        raise ValueError(
            f"expected {EXPECTED_FIT3D_DEV_CLUSTERS} clusters, found {len(clusters)}"
        )
    return clips


def complete_rep_count(analysis: dict[str, Any], *, default_only: bool = False) -> int:
    sets = analysis.get("sets") or []
    if default_only:
        sets = [item for item in sets if item.get("is_default")]
    return sum(
        rep.get("status") == "complete"
        for set_candidate in sets
        for rep in set_candidate.get("reps") or []
    )


def fit3d_evaluation_row(
    clip: dict[str, Any],
    analysis: dict[str, Any],
    selected_events: list[dict[str, Any]],
    *,
    chain_config_hash: str,
) -> dict[str, Any]:
    gt_frames = sorted(
        int(event["clip_frame"])
        for event in (clip.get("gt") or {}).get("events", [])
        if event.get("event") == "top"
    )
    evaluation = evaluate_fit3d_events(
        gt_frames,
        selected_events,
        HIGH_CONFIDENCE_THRESHOLD,
    )
    rep_count = sum(
        len(set_candidate["reps"])
        for set_candidate in analysis.get("sets") or []
    )
    default_set = next(
        (
            set_candidate
            for set_candidate in analysis.get("sets") or []
            if set_candidate["is_default"]
        ),
        None,
    )
    diagnostics = analysis.get("diagnostics") or {}
    return {
        "_abs_errors": evaluation.pop("abs_errors"),
        "dataset": clip.get("dataset"),
        "clip_id": clip["clip_id"],
        "cluster_id": clip["cluster_id"],
        "subject_id": clip.get("subject_id"),
        "camera_id": clip.get("camera_id"),
        "action": clip.get("action"),
        "view": clip.get("view"),
        "status": analysis["status"],
        "motion_status": analysis.get("motion_status"),
        "chain_source": diagnostics.get("chain_source"),
        "phase_signal": diagnostics.get("selected_phase_signal"),
        "period_frames": diagnostics.get("period_frames"),
        "periodicity": diagnostics.get("periodicity"),
        "set_count": len(analysis.get("sets") or []),
        "rep_count": rep_count,
        "default_set_rep_count": len(default_set["reps"])
        if default_set
        else 0,
        "complete_rep_count": complete_rep_count(analysis),
        "default_complete_rep_count": complete_rep_count(
            analysis, default_only=True
        ),
        "default_set_confidence": default_set["chain_confidence"]
        if default_set
        else None,
        "rejected_candidate_count": len(
            analysis.get("rejected_candidates") or []
        ),
        "count_error": evaluation["pred_event_count"] - len(gt_frames),
        "chain_config_sha256": chain_config_hash,
        **evaluation,
    }


def run_fit3d_dual_evaluation(
    clips: list[dict[str, Any]],
    *,
    specs: dict[str, Any],
    segmenter_config: dict[str, Any],
    chain_config: dict[str, Any],
    analysis_dir: Path,
    chain_config_hash: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    raw_rows = []
    raw_analyses = []
    gated_rows = []
    gated_analyses = []
    detector = segmenter_config["core"].get("detector")
    for clip in clips:
        clip_dir = analysis_dir / clip["clip_id"]
        analysis = phase_states.load_json(clip_dir / "analysis.json")
        signals = phase_states.load_angle_signals(
            clip_dir / "angles" / "angles_wide.csv"
        )
        action = phase_states.action_config(clip["action"], specs)
        phase_policy = segmenter_config["core"]["phase_signal"]["policy"]
        chain_analysis, meta = analyze_clip_signals(
            clip,
            signals,
            action,
            prior_signal=(analysis.get("segmentation") or {}).get(
                "phase_angle"
            ),
            phase_policy=phase_policy,
            chain_config=chain_config,
            detector=detector,
        )
        chain_analysis["diagnostics"]["chain_config_sha256"] = chain_config_hash
        gate = quality_gates.from_legacy_quality(analysis)
        gated_analysis = quality_gates.apply_quality_gates(
            chain_analysis, [gate]
        )
        gated_analysis["diagnostics"]["chain_config_sha256"] = chain_config_hash
        raw_row = fit3d_evaluation_row(
            clip,
            chain_analysis,
            meta.get("selected_top_events") or [],
            chain_config_hash=chain_config_hash,
        )
        gated_events = (
            (gated_analysis.get("diagnostics") or {}).get("selected_top_events")
            or []
        )
        gated_row = fit3d_evaluation_row(
            clip,
            gated_analysis,
            gated_events,
            chain_config_hash=chain_config_hash,
        )
        gated_row.update(
            {
                "quality_status": gate["status"],
                "quality_source": gate["source"],
                "quality_action": (
                    "blocked" if gate["blocking"] else "reported"
                ),
                "quality_reasons": "|".join(gate["reasons"]),
                "excluded_by_quality": bool(gate["blocking"]),
                "excluded_by_motion": bool(
                    not gate["blocking"]
                    and gated_analysis.get("status") == "no_motion"
                ),
                "raw_status": raw_row["status"],
                "raw_motion_status": raw_row["motion_status"],
                "raw_set_count": raw_row["set_count"],
                "raw_rep_count": raw_row["rep_count"],
                "raw_default_set_rep_count": raw_row[
                    "default_set_rep_count"
                ],
                "raw_complete_rep_count": raw_row["complete_rep_count"],
                "raw_default_complete_rep_count": raw_row[
                    "default_complete_rep_count"
                ],
                "raw_pred_event_count": raw_row["pred_event_count"],
                "raw_tp": raw_row["tp"],
                "raw_fp": raw_row["fp"],
                "raw_fn": raw_row["fn"],
            }
        )
        raw_rows.append(raw_row)
        raw_analyses.append(chain_analysis)
        gated_rows.append(gated_row)
        gated_analyses.append(gated_analysis)
    return raw_rows, raw_analyses, gated_rows, gated_analyses


def run_fit3d_evaluation(
    clips: list[dict[str, Any]],
    *,
    specs: dict[str, Any],
    segmenter_config: dict[str, Any],
    chain_config: dict[str, Any],
    analysis_dir: Path,
    chain_config_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    _, _, gated_rows, gated_analyses = run_fit3d_dual_evaluation(
        clips,
        specs=specs,
        segmenter_config=segmenter_config,
        chain_config=chain_config,
        analysis_dir=analysis_dir,
        chain_config_hash=chain_config_hash,
    )
    return gated_rows, gated_analyses, []


def annotation_intervals(clip: dict[str, Any]) -> list[dict[str, Any]]:
    sets = (clip.get("gt") or {}).get("annotation_sets") or []
    if not sets:
        return []
    chosen = next(
        (
            annotation_set
            for annotation_set in sets
            if annotation_set.get("annotation_set_id") == "t1_3_blind"
        ),
        sets[-1],
    )
    return list(chosen.get("intervals") or [])


def match_intervals_by_iou(
    gt_intervals: list[dict[str, Any]],
    pred_intervals: list[dict[str, Any]],
    minimum_iou: float = 0.30,
) -> tuple[list[tuple[int, int, float]], list[float], list[float]]:
    gt_spans = [
        (int(interval["start_frame"]), int(interval["end_frame"]))
        for interval in gt_intervals
    ]
    pred_spans = [
        (int(interval["start_frame"]), int(interval["end_frame"]))
        for interval in pred_intervals
    ]
    if not gt_spans or not pred_spans:
        return [], [], []
    iou = np.zeros((len(gt_spans), len(pred_spans)), dtype=np.float64)
    for gt_index, (gt_start, gt_end) in enumerate(gt_spans):
        for pred_index, (pred_start, pred_end) in enumerate(pred_spans):
            intersection = max(
                0, min(gt_end, pred_end) - max(gt_start, pred_start) + 1
            )
            union = max(gt_end, pred_end) - min(gt_start, pred_start) + 1
            iou[gt_index, pred_index] = intersection / union if union else 0.0
    rows, columns = linear_sum_assignment(-iou)
    matches = [
        (int(row), int(column), float(iou[row, column]))
        for row, column in zip(rows, columns)
        if iou[row, column] >= minimum_iou
    ]
    start_errors = []
    end_errors = []
    for gt_index, pred_index, _ in matches:
        gt_start, gt_end = gt_spans[gt_index]
        pred_start, pred_end = pred_spans[pred_index]
        start_errors.append(abs(pred_start - gt_start))
        end_errors.append(abs(pred_end - gt_end))
    return sorted(matches), start_errors, end_errors


def run_qevd_regression(
    clips: list[dict[str, Any]],
    *,
    specs: dict[str, Any],
    segmenter_config: dict[str, Any],
    chain_config: dict[str, Any],
    analysis_dir: Path,
    output_analysis_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    pooled_start_errors: list[float] = []
    pooled_end_errors: list[float] = []
    phase_policy = segmenter_config["core"]["phase_signal"]["policy"]
    for clip in clips:
        clip_dir = analysis_dir / clip["clip_id"]
        legacy = phase_states.load_json(clip_dir / "analysis.json")
        signals = phase_states.load_angle_signals(
            clip_dir / "angles" / "angles_wide.csv"
        )
        action = phase_states.action_config(clip["action"], specs)
        prior_signal = (legacy.get("segmentation") or {}).get("phase_angle")
        no_fallback, no_meta = analyze_clip_signals(
            clip,
            signals,
            action,
            prior_signal=prior_signal,
            phase_policy=phase_policy,
            chain_config=chain_config,
            legacy_analysis=legacy,
            allow_legacy_fallback=False,
        )
        fallback, fallback_meta = analyze_clip_signals(
            clip,
            signals,
            action,
            prior_signal=prior_signal,
            phase_policy=phase_policy,
            chain_config=chain_config,
            legacy_analysis=legacy,
            allow_legacy_fallback=True,
            quality_gate=quality_gates.from_legacy_quality(legacy),
            detector=segmenter_config["core"].get("detector"),
        )
        output_analysis_dir.mkdir(parents=True, exist_ok=True)
        phase_states.write_json(
            output_analysis_dir / f"{clip['clip_id']}.json", fallback
        )
        old_count = len(
            [
                rep
                for rep in (legacy.get("segmentation") or {}).get("reps", [])
                if rep.get("status", "complete") == "complete"
            ]
        )
        no_fallback_count = sum(
            len(set_candidate["reps"]) for set_candidate in no_fallback["sets"]
        )
        fallback_count = sum(
            len(set_candidate["reps"]) for set_candidate in fallback["sets"]
        )
        intervals = annotation_intervals(clip)
        trim_start = int(clip["trim"]["start_frame"])
        gt_intervals = [
            {
                "start_frame": int(interval["start_frame"]) - trim_start,
                "end_frame": int(interval["end_frame"]) - trim_start,
            }
            for interval in intervals
        ]
        pred_intervals = [
            {
                "start_frame": int(rep["start_frame"]),
                "end_frame": int(rep["end_frame"]),
            }
            for set_candidate in fallback["sets"]
            for rep in set_candidate["reps"]
        ]
        matches, start_errors, end_errors = match_intervals_by_iou(
            gt_intervals, pred_intervals
        )
        pooled_start_errors.extend(start_errors)
        pooled_end_errors.extend(end_errors)
        rows.append(
            {
                "clip_id": clip["clip_id"],
                "action": clip["action"],
                "periodicity": round(float(no_meta.get("periodicity") or 0.0), 6),
                "chain_source": fallback["diagnostics"].get("chain_source"),
                "fallback_used": bool(fallback_meta.get("fallback_used")),
                "legacy_pipeline_count": old_count,
                "v2_no_fallback_count": no_fallback_count,
                "fallback_enabled_count": fallback_count,
                "v2_no_fallback_abs_delta": abs(no_fallback_count - old_count),
                "fallback_enabled_abs_delta": abs(fallback_count - old_count),
                "annotation_rep_count": len(gt_intervals),
                "annotation_matches": len(matches),
                "boundary_match_fraction": phase_states.ratio(
                    len(matches), len(gt_intervals)
                ),
                "start_abs_error_median_frames": phase_states.percentile(
                    start_errors, 50
                ),
                "end_abs_error_median_frames": phase_states.percentile(
                    end_errors, 50
                ),
                "boundary_p95_abs_error_frames": phase_states.percentile(
                    start_errors + end_errors, 95
                ),
            }
        )
    deltas = [
        int(row["fallback_enabled_abs_delta"]) for row in rows
    ]
    boundary_values = pooled_start_errors + pooled_end_errors
    annotated_clip_count = sum(
        int(row["annotation_rep_count"]) > 0 for row in rows
    )
    summary = {
        "clip_count": len(rows),
        "annotated_clip_count": annotated_clip_count,
        "annotated_interval_count": len(boundary_values) // 2,
        "fallback_used_count": sum(bool(row["fallback_used"]) for row in rows),
        "legacy_pipeline_count_mae": round(
            float(np.mean(deltas)), 6
        )
        if deltas
        else None,
        "legacy_pipeline_max_count_delta": max(deltas) if deltas else None,
        "boundary_median_abs_error_frames": phase_states.percentile(
            boundary_values, 50
        ),
        "boundary_p95_abs_error_frames": phase_states.percentile(
            boundary_values, 95
        ),
        "boundary_statistics_scope": (
            "pooled_blind_intervals_from_annotated_dev_clips"
        ),
        "count_mae_lte_0_5": bool(deltas) and float(np.mean(deltas)) <= 0.5,
        "max_count_delta_lte_2": bool(deltas) and max(deltas) <= 2,
    }
    return rows, summary


def write_chain_outputs(
    results_dir: Path,
    rows: list[dict[str, Any]],
    analyses: list[dict[str, Any]],
    summary: dict[str, Any],
    qevd_rows: list[dict[str, Any]],
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    audit_fields = [
        "dataset",
        "clip_id",
        "cluster_id",
        "subject_id",
        "camera_id",
        "action",
        "view",
        "status",
        "chain_source",
        "phase_signal",
        "period_frames",
        "periodicity",
        "set_count",
        "rep_count",
        "default_set_rep_count",
        "default_set_confidence",
        "rejected_candidate_count",
        "gt_event_count",
        "pred_event_count",
        "tp",
        "fp",
        "fn",
        "event_recall",
        "event_precision",
        "first_gt_matched",
        "last_gt_matched",
        "event_abs_error_median_frames",
        "event_abs_error_p90_frames",
        "event_signed_error_median_frames",
        "high_confidence_fp",
        "outside_set_high_confidence_fp",
        "count_error",
        "event_gate_frames",
    ]
    public_rows = [
        {field: row.get(field) for field in audit_fields} for row in rows
    ]
    phase_states.write_csv(results_dir / "chain_audit.csv", public_rows, audit_fields)
    qevd_fields = list(qevd_rows[0]) if qevd_rows else []
    phase_states.write_csv(
        results_dir / "qevd_chain_regression.csv",
        qevd_rows,
        qevd_fields,
    )
    analysis_dir = results_dir / "clip_analysis_v2"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    for analysis in analyses:
        phase_states.write_json(
            analysis_dir / f"{analysis['clip_id']}.json", analysis
        )
    phase_states.write_json(results_dir / "chain_eval_summary.json", summary)
    phase_states.write_checksums(
        results_dir,
        [
            "chain_audit.csv",
            "qevd_chain_regression.csv",
            "chain_eval_summary.json",
        ],
    )


def main() -> int:
    args = parse_args()
    sources = phase_states.load_json(args.sources)
    specs = phase_states.load_json(args.specs)
    segmenter_config = phase_states.load_json(args.segmenter_config)
    chain_config = load_chain_config(args.chain_config)
    fit3d_clips = load_fit3d_clips(sources)
    qevd_clips = [
        clip
        for clip in sources.get("clips", [])
        if clip.get("dataset") == QEVD_DATASET
        and clip.get("split_role") == "dev"
    ]
    if len(qevd_clips) != EXPECTED_QEVD_DEV_CLIPS:
        raise ValueError(
            f"expected {EXPECTED_QEVD_DEV_CLIPS} QEVD dev clips, found {len(qevd_clips)}"
        )
    rows, analyses, _ = run_fit3d_evaluation(
        fit3d_clips,
        specs=specs,
        segmenter_config=segmenter_config,
        chain_config=chain_config,
        analysis_dir=args.fit3d_analysis_dir,
        chain_config_hash=phase_states.sha256_path(args.chain_config),
    )
    all_cameras = aggregate_fit3d_rows(rows)
    stage_gates = {
        "first_event_recall_gte_0_75": bool(
            all_cameras["first_event_recall"] is not None
            and all_cameras["first_event_recall"] >= 0.75
        ),
        "last_event_recall_gte_0_75": bool(
            all_cameras["last_event_recall"] is not None
            and all_cameras["last_event_recall"] >= 0.75
        ),
        "outside_set_high_confidence_fp_zero": (
            all_cameras["outside_set_high_confidence_fp"] == 0
        ),
    }
    qevd_rows, qevd_summary = run_qevd_regression(
        qevd_clips,
        specs=specs,
        segmenter_config=segmenter_config,
        chain_config=chain_config,
        analysis_dir=args.qevd_analysis_dir,
        output_analysis_dir=args.results_dir / "qevd_clip_analysis_v2",
    )
    qevd_summary["smoke_gates"] = {
        "count_not_degraded": bool(
            qevd_summary["count_mae_lte_0_5"]
            and qevd_summary["max_count_delta_lte_2"]
        ),
        "boundary_median_lte_12_frames": bool(
            qevd_summary["boundary_median_abs_error_frames"] is not None
            and qevd_summary["boundary_median_abs_error_frames"] <= 12
        ),
        "boundary_p95_lte_25_frames": bool(
            qevd_summary["boundary_p95_abs_error_frames"] is not None
            and qevd_summary["boundary_p95_abs_error_frames"] <= 25
        ),
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "task": "T1-6C",
        "generated_at": phase_states.utc_now(),
        "source": {
            "sources": str(args.sources.resolve()),
            "sources_sha256": phase_states.sha256_path(args.sources),
            "segmenter_config": str(args.segmenter_config.resolve()),
            "segmenter_config_sha256": phase_states.sha256_path(
                args.segmenter_config
            ),
            "chain_config": str(args.chain_config.resolve()),
            "chain_config_sha256": phase_states.sha256_path(args.chain_config),
            "fit3d_manifest": str(args.fit3d_manifest.resolve()),
            "fit3d_manifest_sha256": phase_states.sha256_path(
                args.fit3d_manifest
            ),
        },
        "fit3d_dev": {
            "all_cameras": all_cameras,
            "stage_gates": stage_gates,
            "stage_passed": all(stage_gates.values()),
        },
        "qevd_dev": {
            **qevd_summary,
            "smoke_passed": all(qevd_summary["smoke_gates"].values()),
            "formal_stage_gate_evaluable": (
                qevd_summary["annotated_clip_count"] > 1
            ),
            "stage_passed": None,
        },
        "artifacts": {
            "chain_audit": "chain_audit.csv",
            "chain_summary": "chain_eval_summary.json",
            "clip_analysis_v2": "clip_analysis_v2/",
            "qevd_regression": "qevd_chain_regression.csv",
        },
    }
    write_chain_outputs(
        args.results_dir,
        rows,
        analyses,
        summary,
        qevd_rows,
    )
    print(
        "fit3d "
        f"recall={all_cameras['event_recall']} "
        f"precision={all_cameras['event_precision']} "
        f"first={all_cameras['first_event_recall']} "
        f"last={all_cameras['last_event_recall']} "
        f"median={all_cameras['event_abs_error_median_frames']} "
        f"p90={all_cameras['event_abs_error_p90_frames']}"
    )
    print(
        "fit3d "
        f"count_mae={all_cameras['count_mae']} "
        f"exact={all_cameras['count_exact_correct']}/32 "
        f"high_conf_fp={all_cameras['high_confidence_event_fp']}"
    )
    print(
        f"fit3d_stage={'pass' if summary['fit3d_dev']['stage_passed'] else 'fail'} "
        f"qevd_count_mae={qevd_summary['legacy_pipeline_count_mae']} "
        f"qevd_boundary_median={qevd_summary['boundary_median_abs_error_frames']} "
        f"qevd_smoke={'pass' if summary['qevd_dev']['smoke_passed'] else 'fail'} "
        f"qevd_formal_gate={'evaluable' if summary['qevd_dev']['formal_stage_gate_evaluable'] else 'not_evaluable'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
