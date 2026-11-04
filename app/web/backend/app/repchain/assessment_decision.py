#!/usr/bin/env python3
"""Apply shadow issue decisions to phase-aware measurements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RULES = ROOT / "configs" / "assessment_rules.json"
DEFAULT_TAXONOMY = ROOT / "configs" / "issue_taxonomy.json"
DEFAULT_CONTRACTS = ROOT / "configs" / "metric_contracts.json"
DEFAULT_MEASUREMENT_DIR = ROOT / "results" / "assessment" / "measurement"
DEFAULT_ANNOTATIONS = ROOT / "results" / "assessment" / "annotations" / "adjudicated.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "assessment" / "decision"

ACTIVE_OBSERVATION_STATUSES = {"ok", "low_confidence"}
SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
SEVERITY_BY_RANK = {rank: name for name, rank in SEVERITY_RANK.items()}
EVALUATION_FIELDS = [
    "clip_id",
    "split",
    "exercise",
    "view",
    "quality_status",
    "measurement_id",
    "metric",
    "scope",
    "rep_id",
    "phase",
    "raw_value",
    "unit",
    "observation_status",
    "observation_confidence",
    "eligibility",
    "report_view_allowed",
    "physical_valid",
    "threshold_used",
    "direction",
    "deviation",
    "magnitude_severity",
    "adjusted_confidence",
    "decision",
    "reason",
    "anchor_frame",
    "anchor_source_frame",
    "evidence_start_frame",
    "evidence_end_frame",
    "source_file",
    "source_signals",
]
ISSUE_FIELDS = [
    "clip_id",
    "split",
    "exercise",
    "view",
    "quality_status",
    "issue_id",
    "issue_code",
    "issue_family",
    "scope",
    "rep_id",
    "support_rep_ids",
    "support_rep_count",
    "support_metric_count",
    "metrics",
    "support_measurement_ids",
    "severity",
    "raw_severity",
    "confidence",
    "decision",
    "reportable",
    "reasons",
    "anchor_frame",
    "anchor_source_frame",
    "anchor_time_s",
    "evidence_start_frame",
    "evidence_end_frame",
    "threshold_policy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-dir", type=Path, default=DEFAULT_MEASUREMENT_DIR)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clip-id", nargs="+", default=None)
    parser.add_argument(
        "--split",
        nargs="+",
        default=["dev", "calibration"],
        choices=["dev", "calibration", "holdout"],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--generated-at", default=None)
    parser.add_argument("--skip-sweep", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def rounded(value: float | None, digits: int = 6) -> float | None:
    return round(float(value), digits) if value is not None and math.isfinite(value) else None


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def severity_at_least(severity: str, minimum: str) -> bool:
    return SEVERITY_RANK.get(severity, -1) >= SEVERITY_RANK.get(minimum, 99)


def severity_cap(severity: str, maximum: str) -> str:
    rank = min(SEVERITY_RANK.get(severity, 0), SEVERITY_RANK.get(maximum, 3))
    return SEVERITY_BY_RANK[rank]


def drop_severity(severity: str, levels: int) -> str:
    rank = max(0, SEVERITY_RANK.get(severity, 0) - max(0, levels))
    return SEVERITY_BY_RANK[rank]


def validate_rules(
    rules: dict[str, Any],
    taxonomy: dict[str, Any],
    contracts: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    policy = rules.get("policy") or {}
    if policy.get("holdout_allowed") is not False:
        errors.append("policy.holdout_allowed must be false")
    if policy.get("expert_calibrated") is not False:
        errors.append("expert_calibrated must be false")
    if policy.get("product_actions_enabled") is not False:
        errors.append("product_actions_enabled must be false")
    thresholds = rules.get("thresholds") or {}
    physical = rules.get("physical_ranges") or {}
    metric_policy = rules.get("metric_policy") or {}
    issue_rules = rules.get("issue_rules") or {}
    taxonomy_issues = taxonomy.get("issues") or {}
    contract_metrics = {
        metric
        for metrics in (contracts.get("exercise_metrics") or {}).values()
        for metric in metrics
    }
    contract_metrics.update((contracts.get("metrics") or {}).keys())
    for metric, threshold in thresholds.items():
        if metric not in contract_metrics:
            errors.append(f"threshold metric is not in contracts: {metric}")
        if metric not in physical:
            errors.append(f"threshold metric has no physical range: {metric}")
        if not {"low", "medium", "high", "direction"} <= set(threshold):
            errors.append(f"threshold metric has incomplete levels: {metric}")
        if threshold.get("direction") not in {"low_bad", "high_bad"}:
            errors.append(f"threshold metric has invalid direction: {metric}")
        bounds = physical.get(metric) or []
        if len(bounds) != 2 or float(bounds[0]) >= float(bounds[1]):
            errors.append(f"threshold metric has invalid physical range: {metric}")
    for metric, entry in metric_policy.items():
        if metric not in contract_metrics:
            errors.append(f"metric policy references unknown metric: {metric}")
        if entry.get("eligibility") not in {
            "provisional",
            "diagnostic_only",
            "experimental",
        }:
            errors.append(f"metric policy has invalid eligibility: {metric}")
        if entry.get("eligibility") == "provisional" and not entry.get("report_views"):
            errors.append(f"provisional metric has no report_views: {metric}")
    if set(issue_rules) != set(taxonomy_issues):
        errors.append("issue_rules must cover exactly the taxonomy issues")
    for issue_code, entry in issue_rules.items():
        taxonomy_issue = taxonomy_issues.get(issue_code) or {}
        if entry.get("family") != taxonomy_issue.get("family"):
            errors.append(f"issue family mismatch: {issue_code}")
        if entry.get("scope") not in (taxonomy_issue.get("annotation_scope") or []):
            errors.append(f"issue scope mismatch: {issue_code}")
        for metric in entry.get("metrics") or []:
            if metric not in thresholds:
                errors.append(f"issue metric has no threshold: {issue_code}:{metric}")
            if metric not in metric_policy:
                errors.append(f"issue metric has no metric policy: {issue_code}:{metric}")
    for preferred, lower in (rules.get("arbitration", {}).get("family_precedence") or {}).items():
        if preferred not in issue_rules:
            errors.append(f"unknown precedence issue: {preferred}")
        for issue_code in lower:
            if issue_code not in issue_rules:
                errors.append(f"unknown suppressed precedence issue: {issue_code}")
    confidence = rules.get("confidence") or {}
    checks = (
        float(confidence.get("suppress_below", -1)),
        float(confidence.get("low_below", -1)),
        float(confidence.get("report_below", -1)),
    )
    if not (0 <= checks[0] <= checks[1] <= checks[2] <= 1):
        errors.append("confidence gates must satisfy 0 <= suppress <= low <= report <= 1")
    return errors


def metric_policy_for(rules: dict[str, Any], metric: str) -> dict[str, Any]:
    return (rules.get("metric_policy") or {}).get(metric) or {
        "eligibility": "diagnostic_only",
        "reason": "metric has no product eligibility policy",
    }


def threshold_severity(
    value: float,
    threshold: dict[str, Any],
    scale: float,
) -> tuple[str, float | None, float]:
    low = float(threshold["low"]) * scale
    medium = float(threshold["medium"]) * scale
    high = float(threshold["high"]) * scale
    direction = str(threshold["direction"])
    if direction == "low_bad":
        levels = ((high, "high"), (medium, "medium"), (low, "low"))
        for boundary, severity in levels:
            if value <= boundary:
                return severity, boundary, max(0.0, boundary - value)
        return "none", None, 0.0
    levels = ((high, "high"), (medium, "medium"), (low, "low"))
    for boundary, severity in levels:
        if value >= boundary:
            return severity, boundary, max(0.0, value - boundary)
    return "none", None, 0.0


def physical_range_valid(
    value: float,
    metric: str,
    physical_ranges: dict[str, Any],
) -> bool:
    bounds = physical_ranges.get(metric)
    if not bounds:
        return False
    return float(bounds[0]) <= value <= float(bounds[1])


def evaluation_evidence(
    observation: dict[str, Any],
    rep_by_id: dict[int, dict[str, Any]],
    radius: int,
    frame_count: int,
) -> tuple[int | None, int | None, int | None]:
    anchor = observation.get("anchor_frame")
    rep_id = observation.get("rep_id")
    rep = rep_by_id.get(int(rep_id)) if rep_id is not None else None
    if anchor is None and rep is not None:
        anchor = rep.get("bottom_frame", rep.get("start_frame"))
    if anchor is None:
        return None, None, None
    start = int(anchor) - radius
    end = int(anchor) + radius
    if rep is not None:
        start = max(int(rep["start_frame"]), start)
        end = min(int(rep["end_frame"]), end)
    else:
        start = max(0, start)
        end = min(max(0, frame_count - 1), end)
    return int(anchor), start, end


def evaluate_observation(
    observation: dict[str, Any],
    clip: dict[str, Any],
    measurement: dict[str, Any],
    rules: dict[str, Any],
    threshold_scale: float,
) -> dict[str, Any]:
    metric = observation["metric"]
    policy = metric_policy_for(rules, metric)
    threshold = (rules.get("thresholds") or {}).get(metric)
    confidence_cfg = rules["confidence"]
    value = as_float(observation.get("normalized_value"))
    confidence = as_float(observation.get("confidence"))
    observation_status = observation.get("status")
    physical_valid = False
    severity = "none"
    threshold_used = None
    deviation = 0.0
    decision = observation_status or "unavailable"
    reasons: list[str] = []
    report_view_allowed = bool(
        policy.get("eligibility") == "provisional"
        and clip["view"] in (policy.get("report_views") or [])
    )

    if observation_status == "disabled":
        reasons.append(observation.get("reason") or "view_unsupported")
    elif observation_status in {"unavailable", "insufficient_reps", "conflict"}:
        reasons.append(observation.get("reason") or str(observation_status))
    elif observation_status not in ACTIVE_OBSERVATION_STATUSES:
        decision = "unavailable"
        reasons.append(f"unsupported_observation_status:{observation_status}")
    elif value is None:
        decision = "unavailable"
        reasons.append("missing_numeric_value")
    elif threshold is None:
        decision = "suppressed"
        reasons.append("missing_threshold")
    else:
        physical_valid = physical_range_valid(
            value, metric, rules.get("physical_ranges") or {}
        )
        if not physical_valid:
            decision = "suppressed"
            reasons.append("physical_range_invalid")
        elif confidence is None:
            decision = "unavailable"
            reasons.append("missing_confidence")
        elif confidence < float(confidence_cfg["suppress_below"]):
            decision = "suppressed"
            reasons.append("confidence_below_suppress")
        else:
            severity, threshold_used, deviation = threshold_severity(
                value, threshold, threshold_scale
            )
            eligibility = policy.get("eligibility")
            if severity == "none":
                decision = "no_issue"
            elif eligibility in {"diagnostic_only", "experimental"}:
                decision = "diagnostic"
                reasons.append(f"metric_policy_{eligibility}")
            elif not report_view_allowed:
                decision = "diagnostic"
                reasons.append("view_outside_report_scope")
            else:
                decision = "candidate"

    rep_by_id = {
        int(rep["rep_id"]): rep for rep in measurement.get("reps") or []
    }
    radius = int((rules.get("evidence") or {}).get("radius_frames", 8))
    frame_count = int((measurement.get("input_contract") or {}).get("frame_count", 0))
    anchor, evidence_start, evidence_end = evaluation_evidence(
        observation, rep_by_id, radius, frame_count
    )
    trace = observation.get("trace") or {}
    return {
        "measurement_id": observation["measurement_id"],
        "metric": metric,
        "scope": observation["scope"],
        "rep_id": observation.get("rep_id"),
        "phase": observation["phase"],
        "raw_value": rounded(value),
        "unit": observation.get("unit", ""),
        "observation_status": observation_status,
        "observation_confidence": rounded(confidence),
        "eligibility": policy.get("eligibility"),
        "report_view_allowed": report_view_allowed,
        "physical_valid": physical_valid,
        "threshold_policy": (rules.get("policy") or {}).get("threshold_policy"),
        "threshold": {
            "low": rounded(float(threshold["low"]) * threshold_scale) if threshold else None,
            "medium": rounded(float(threshold["medium"]) * threshold_scale) if threshold else None,
            "high": rounded(float(threshold["high"]) * threshold_scale) if threshold else None,
            "direction": threshold.get("direction") if threshold else None,
        },
        "threshold_used": rounded(threshold_used),
        "direction": threshold.get("direction") if threshold else None,
        "deviation": rounded(deviation),
        "magnitude_severity": severity,
        "adjusted_confidence": rounded(confidence),
        "decision": decision,
        "reason": ";".join(reasons) if reasons else None,
        "anchor_frame": anchor,
        "anchor_source_frame": observation.get("anchor_source_frame"),
        "anchor_time_s": observation.get("anchor_time_s"),
        "evidence_start_frame": evidence_start,
        "evidence_end_frame": evidence_end,
        "trace": trace,
    }


def issue_support_evaluations(
    evaluations: list[dict[str, Any]],
    issue_rule: dict[str, Any],
    minimum_severity: str,
) -> list[dict[str, Any]]:
    metrics = set(issue_rule.get("metrics") or [])
    return [
        evaluation
        for evaluation in evaluations
        if evaluation["metric"] in metrics
        and evaluation["observation_status"] in ACTIVE_OBSERVATION_STATUSES
        and evaluation["raw_value"] is not None
        and evaluation["physical_valid"]
        and severity_at_least(evaluation["magnitude_severity"], minimum_severity)
    ]


def contiguous_rep_blocks(rep_ids: Iterable[int]) -> list[list[int]]:
    ordered = sorted(set(int(rep_id) for rep_id in rep_ids))
    blocks: list[list[int]] = []
    for rep_id in ordered:
        if not blocks or rep_id != blocks[-1][-1] + 1:
            blocks.append([rep_id])
        else:
            blocks[-1].append(rep_id)
    return blocks


def persistence_factor(confidence_cfg: dict[str, Any], support_count: int) -> float:
    factors = confidence_cfg.get("persistence_factors") or {}
    if support_count >= 3:
        return float(factors.get("3_or_more", 1.0))
    return float(factors.get(str(max(1, support_count)), 0.8))


def representative_evaluation(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        evaluations,
        key=lambda item: (
            SEVERITY_RANK.get(item["magnitude_severity"], 0),
            item.get("adjusted_confidence") or 0.0,
            item.get("deviation") or 0.0,
        ),
    )


def build_issue_group(
    clip: dict[str, Any],
    measurement: dict[str, Any],
    rules: dict[str, Any],
    taxonomy: dict[str, Any],
    issue_code: str,
    issue_rule: dict[str, Any],
    evaluations: list[dict[str, Any]],
    rep_ids: list[int],
    report_min_severity: str | None,
) -> dict[str, Any]:
    confidence_cfg = rules["confidence"]
    quality_status = measurement["quality"]["status"]
    representative = representative_evaluation(evaluations)
    raw_severity = representative["magnitude_severity"]
    severity = raw_severity
    reasons: list[str] = []

    support_count = len(rep_ids) if issue_rule["scope"] == "rep" else int(
        measurement["summary"]["rep_count"]
    )
    factor = (
        persistence_factor(confidence_cfg, support_count)
        if issue_rule["scope"] == "rep"
        else 1.0
    )
    confidence = (representative.get("adjusted_confidence") or 0.0) * factor
    if issue_rule["scope"] == "rep" and len(rep_ids) == 1:
        severity = severity_cap(
            severity, issue_rule.get("single_rep_max_severity", "medium")
        )
        if severity != raw_severity:
            reasons.append("single_rep_severity_cap")
    if quality_status == "warn":
        severity = drop_severity(
            severity, int(confidence_cfg.get("quality_warn_drop_levels", 1))
        )
        severity = severity_cap(
            severity,
            confidence_cfg.get("quality_warn_max_severity", "medium"),
        )
        reasons.append("quality_warn_severity_cap")
    if confidence < float(confidence_cfg["low_below"]):
        severity = severity_cap(severity, "low")
        reasons.append("low_confidence_severity_cap")

    product_eligible = any(
        evaluation["eligibility"] == "provisional"
        and evaluation["report_view_allowed"]
        for evaluation in evaluations
    )
    if not product_eligible:
        decision = "diagnostic"
        reasons.append("no_product_eligible_metric")
        reportable = False
    elif confidence < float(confidence_cfg["suppress_below"]):
        decision = "suppressed"
        reasons.append("issue_confidence_below_suppress")
        reportable = False
    else:
        minimum_report_severity = report_min_severity or issue_rule.get(
            "report_min_severity", "high"
        )
        report_confidence = max(
            float(issue_rule.get("report_min_confidence", confidence_cfg["report_below"])),
            float(confidence_cfg["report_below"]),
        )
        persistence_ok = support_count >= int(issue_rule.get("min_persistent_reps", 1))
        evidence_ready = (
            representative.get("anchor_frame") is not None
            and representative.get("evidence_start_frame") is not None
            and representative.get("evidence_end_frame") is not None
        )
        if not persistence_ok:
            reasons.append("insufficient_persistence")
        if confidence < report_confidence:
            reasons.append("confidence_below_report_gate")
        if not severity_at_least(severity, minimum_report_severity):
            reasons.append("severity_below_report_gate")
        if not evidence_ready:
            reasons.append("evidence_missing")
        reportable = (
            persistence_ok
            and confidence >= report_confidence
            and severity_at_least(severity, minimum_report_severity)
            and evidence_ready
        )
        decision = "reportable" if reportable else "candidate"

    if quality_status == "reject":
        decision = "suppressed"
        reportable = False
        reasons.append("quality_reject")
    elif severity == "none":
        decision = "suppressed"
        reportable = False
        reasons.append("severity_suppressed_by_quality_or_confidence")

    metrics = sorted({evaluation["metric"] for evaluation in evaluations})
    return {
        "issue_id": "",
        "issue_code": issue_code,
        "issue_family": issue_rule["family"],
        "definition": (taxonomy.get("issues") or {}).get(issue_code, {}).get("definition"),
        "scope": issue_rule["scope"],
        "rep_id": rep_ids[0] if issue_rule["scope"] == "rep" and len(rep_ids) == 1 else None,
        "support_rep_ids": rep_ids if issue_rule["scope"] == "rep" else [],
        "support_rep_count": support_count,
        "support_metric_count": len(metrics),
        "metrics": metrics,
        "support_measurement_ids": sorted(
            {evaluation["measurement_id"] for evaluation in evaluations}
        ),
        "severity": severity,
        "raw_severity": raw_severity,
        "confidence": rounded(confidence),
        "persistence_factor": rounded(factor),
        "decision": decision,
        "reportable": reportable,
        "reasons": reasons,
        "threshold_policy": (rules.get("policy") or {}).get("threshold_policy"),
        "anchor_frame": representative.get("anchor_frame"),
        "anchor_source_frame": representative.get("anchor_source_frame"),
        "anchor_time_s": representative.get("anchor_time_s"),
        "evidence_interval": [
            representative.get("evidence_start_frame"),
            representative.get("evidence_end_frame"),
        ],
        "representative_measurement_id": representative["measurement_id"],
        "support_metric_evaluations": evaluations,
    }


def issue_groups(
    clip: dict[str, Any],
    measurement: dict[str, Any],
    rules: dict[str, Any],
    taxonomy: dict[str, Any],
    evaluations: list[dict[str, Any]],
    report_min_severity: str | None = None,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    exercise = clip["exercise"]
    for issue_code, issue_rule in (rules.get("issue_rules") or {}).items():
        taxonomy_issue = (taxonomy.get("issues") or {}).get(issue_code) or {}
        if taxonomy_issue.get("action") != exercise:
            continue
        support = issue_support_evaluations(
            evaluations,
            issue_rule,
            issue_rule.get("candidate_min_severity", "low"),
        )
        if not support:
            continue
        if issue_rule["scope"] == "clip":
            groups.append(
                build_issue_group(
                    clip,
                    measurement,
                    rules,
                    taxonomy,
                    issue_code,
                    issue_rule,
                    support,
                    [],
                    report_min_severity,
                )
            )
            continue
        by_rep: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for evaluation in support:
            if evaluation.get("rep_id") is not None:
                by_rep[int(evaluation["rep_id"])].append(evaluation)
        for block in contiguous_rep_blocks(by_rep):
            block_evaluations = [
                evaluation
                for rep_id in block
                for evaluation in by_rep.get(rep_id, [])
            ]
            groups.append(
                build_issue_group(
                    clip,
                    measurement,
                    rules,
                    taxonomy,
                    issue_code,
                    issue_rule,
                    block_evaluations,
                    block,
                    report_min_severity,
                )
            )
    return groups


def issue_overlap(first: dict[str, Any], second: dict[str, Any]) -> bool:
    if first["scope"] == "clip" or second["scope"] == "clip":
        return True
    return bool(set(first["support_rep_ids"]) & set(second["support_rep_ids"]))


def arbitrate_groups(
    groups: list[dict[str, Any]], rules: dict[str, Any]
) -> list[dict[str, Any]]:
    precedence = (rules.get("arbitration") or {}).get("family_precedence") or {}
    for preferred, lower_issues in precedence.items():
        preferred_groups = [group for group in groups if group["issue_code"] == preferred]
        for group in groups:
            if group["issue_code"] not in lower_issues:
                continue
            if any(issue_overlap(group, candidate) for candidate in preferred_groups):
                if group["decision"] == "reportable":
                    group["decision"] = "suppressed"
                elif group["decision"] == "candidate":
                    group["decision"] = "diagnostic"
                group["reportable"] = False
                group["reasons"].append(f"family_precedence:{preferred}")

    if bool((rules.get("arbitration") or {}).get("max_one_reportable_issue_per_family")):
        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for group in groups:
            if group["decision"] == "reportable":
                by_family[group["issue_family"]].append(group)
        for family_groups in by_family.values():
            family_groups.sort(
                key=lambda group: (
                    SEVERITY_RANK.get(group["severity"], 0),
                    group.get("confidence") or 0.0,
                    group["support_rep_count"],
                ),
                reverse=True,
            )
            for group in family_groups[1:]:
                group["decision"] = "suppressed"
                group["reportable"] = False
                group["reasons"].append("max_one_reportable_issue_per_family")
    return groups


def assign_issue_ids(groups: list[dict[str, Any]]) -> None:
    decision_order = {"reportable": 0, "candidate": 1, "suppressed": 2, "diagnostic": 3}
    groups.sort(
        key=lambda group: (
            decision_order.get(group["decision"], 9),
            -SEVERITY_RANK.get(group["severity"], 0),
            -(group.get("confidence") or 0.0),
            group["issue_code"],
            group.get("anchor_frame") or -1,
        )
    )
    counters: Counter[str] = Counter()
    for group in groups:
        counters[group["issue_code"]] += 1
        group["issue_id"] = f"{group['issue_code']}_{counters[group['issue_code']]:02d}"


def build_assessment(
    measurement: dict[str, Any],
    clip: dict[str, Any],
    rules: dict[str, Any],
    taxonomy: dict[str, Any],
    generated_at: str,
    *,
    threshold_scale: float = 1.0,
    report_confidence: float | None = None,
    report_min_severity: str | None = None,
) -> dict[str, Any]:
    local_rules = json.loads(json.dumps(rules))
    if report_confidence is not None:
        local_rules["confidence"]["report_below"] = float(report_confidence)
    observations = list(measurement.get("rep_measurements") or []) + list(
        measurement.get("clip_measurements") or []
    )
    evaluations = [
        evaluate_observation(
            observation,
            clip,
            measurement,
            local_rules,
            threshold_scale,
        )
        for observation in observations
    ]
    groups = issue_groups(
        clip,
        measurement,
        local_rules,
        taxonomy,
        evaluations,
        report_min_severity=report_min_severity,
    )
    groups = arbitrate_groups(groups, local_rules)
    assign_issue_ids(groups)
    decision_counts = Counter(group["decision"] for group in groups)
    reportable = [group for group in groups if group["decision"] == "reportable"]
    quality_status = measurement["quality"]["status"]
    evaluable_metric_count = int(measurement["summary"].get("evaluable_metric_count", 0))
    if quality_status == "reject":
        abstained = True
        abstain_reason = "quality_reject"
    elif evaluable_metric_count == 0:
        abstained = True
        abstain_reason = "no_evaluable_metric"
    else:
        abstained = False
        abstain_reason = None
    if abstained:
        conclusion = abstain_reason
    elif reportable:
        conclusion = "shadow_reportable_issue"
    else:
        conclusion = "no_reliable_issue"
    return {
        "schema_version": "vls-assessment-v2",
        "component": "assessment_decision",
        "generated_at": generated_at,
        "clip_id": clip["clip_id"],
        "subject_id": clip["subject_id"],
        "exercise": clip["exercise"],
        "view": clip["view"],
        "split": clip["split"],
        "workflow_stage": "decision_shadow",
        "source_measurement": {
            "schema_version": measurement.get("schema_version"),
            "component": measurement.get("component"),
            "input_contract": measurement.get("input_contract"),
            "rep_count": measurement.get("summary", {}).get("rep_count", 0),
        },
        "policy": {
            **local_rules["policy"],
            "threshold_scale": threshold_scale,
            "holdout_used": False,
        },
        "quality": measurement["quality"],
        "view_capability": measurement.get("view_capability"),
        "reps": measurement.get("reps") or [],
        "rep_measurements": measurement.get("rep_measurements") or [],
        "clip_measurements": measurement.get("clip_measurements") or [],
        "metric_evaluations": evaluations,
        "issues": groups,
        "summary": {
            "rep_count": measurement["summary"].get("rep_count", 0),
            "observation_count": len(observations),
            "evaluable_metric_count": evaluable_metric_count,
            "metric_evaluation_count": len(evaluations),
            "metric_decision_counts": dict(
                sorted(Counter(item["decision"] for item in evaluations).items())
            ),
            "issue_count": len(groups),
            "candidate_issue_count": sum(
                group["decision"] in {"candidate", "reportable"} for group in groups
            ),
            "reportable_issue_count": len(reportable),
            "suppressed_issue_count": decision_counts.get("suppressed", 0),
            "diagnostic_issue_count": decision_counts.get("diagnostic", 0),
            "candidate_issue_codes": sorted(
                {
                    group["issue_code"]
                    for group in groups
                    if group["decision"] in {"candidate", "reportable"}
                }
            ),
            "reportable_issue_codes": sorted(
                {group["issue_code"] for group in reportable}
            ),
            "abstained": abstained,
            "abstain_reason": abstain_reason,
            "conclusion": conclusion,
            "shadow_only": True,
            "product_actions_enabled": False,
        },
        "confidence": {
            "calibrated": None,
            "calibration_status": "provisional_engineering_not_expert_calibrated",
            "mean_evaluated_metric_confidence": rounded(
                sum(
                    item["adjusted_confidence"]
                    for item in evaluations
                    if item.get("adjusted_confidence") is not None
                )
                / max(
                    1,
                    sum(
                        item.get("adjusted_confidence") is not None
                        for item in evaluations
                    ),
                )
            ),
        },
    }


def evaluation_rows(assessment: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in assessment["metric_evaluations"]:
        trace = item.get("trace") or {}
        rows.append(
            {
                "clip_id": assessment["clip_id"],
                "split": assessment["split"],
                "exercise": assessment["exercise"],
                "view": assessment["view"],
                "quality_status": assessment["quality"]["status"],
                "measurement_id": item["measurement_id"],
                "metric": item["metric"],
                "scope": item["scope"],
                "rep_id": item["rep_id"],
                "phase": item["phase"],
                "raw_value": item["raw_value"],
                "unit": item["unit"],
                "observation_status": item["observation_status"],
                "observation_confidence": item["observation_confidence"],
                "eligibility": item["eligibility"],
                "report_view_allowed": item["report_view_allowed"],
                "physical_valid": item["physical_valid"],
                "threshold_used": item["threshold_used"],
                "direction": item["direction"],
                "deviation": item["deviation"],
                "magnitude_severity": item["magnitude_severity"],
                "adjusted_confidence": item["adjusted_confidence"],
                "decision": item["decision"],
                "reason": item["reason"],
                "anchor_frame": item["anchor_frame"],
                "anchor_source_frame": item["anchor_source_frame"],
                "evidence_start_frame": item["evidence_start_frame"],
                "evidence_end_frame": item["evidence_end_frame"],
                "source_file": trace.get("source_file"),
                "source_signals": ";".join(trace.get("source_signals") or []),
            }
        )
    return rows


def issue_rows(assessment: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for issue in assessment["issues"]:
        evidence = issue.get("evidence_interval") or [None, None]
        rows.append(
            {
                "clip_id": assessment["clip_id"],
                "split": assessment["split"],
                "exercise": assessment["exercise"],
                "view": assessment["view"],
                "quality_status": assessment["quality"]["status"],
                "issue_id": issue["issue_id"],
                "issue_code": issue["issue_code"],
                "issue_family": issue["issue_family"],
                "scope": issue["scope"],
                "rep_id": issue["rep_id"],
                "support_rep_ids": ";".join(str(value) for value in issue["support_rep_ids"]),
                "support_rep_count": issue["support_rep_count"],
                "support_metric_count": issue["support_metric_count"],
                "metrics": ";".join(issue["metrics"]),
                "support_measurement_ids": ";".join(issue["support_measurement_ids"]),
                "severity": issue["severity"],
                "raw_severity": issue["raw_severity"],
                "confidence": issue["confidence"],
                "decision": issue["decision"],
                "reportable": issue["reportable"],
                "reasons": ";".join(issue["reasons"]),
                "anchor_frame": issue["anchor_frame"],
                "anchor_source_frame": issue["anchor_source_frame"],
                "anchor_time_s": issue["anchor_time_s"],
                "evidence_start_frame": evidence[0],
                "evidence_end_frame": evidence[1],
                "threshold_policy": issue["threshold_policy"],
            }
        )
    return rows


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: "" if row.get(field) is None else row.get(field)
                    for field in fields
                }
            )


def validate_schema(payload: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    validator = Draft202012Validator(schema)
    return [
        f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(payload), key=lambda item: list(item.path))
    ]


def invariant_checks(
    assessment: dict[str, Any], rules: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    reportable_families: Counter[str] = Counter()
    for issue in assessment["issues"]:
        if issue["reportable"] != (issue["decision"] == "reportable"):
            errors.append(f"reportable flag mismatch: {issue['issue_id']}")
        if issue["decision"] == "reportable":
            reportable_families[issue["issue_family"]] += 1
            evidence = issue.get("evidence_interval") or []
            anchor = issue.get("anchor_frame")
            if len(evidence) != 2 or evidence[0] is None or evidence[1] is None:
                errors.append(f"reportable issue lacks evidence: {issue['issue_id']}")
            elif not int(evidence[0]) <= int(anchor) <= int(evidence[1]):
                errors.append(f"reportable anchor outside evidence: {issue['issue_id']}")
            if not any(
                item["eligibility"] == "provisional" and item["report_view_allowed"]
                for item in issue["support_metric_evaluations"]
            ):
                errors.append(f"reportable issue lacks product metric: {issue['issue_id']}")
        if issue["definition"] is None:
            errors.append(f"issue missing taxonomy definition: {issue['issue_id']}")
        if issue["confidence"] is None or not 0 <= float(issue["confidence"]) <= 1:
            errors.append(f"issue confidence invalid: {issue['issue_id']}")
    for family, count in reportable_families.items():
        if count > 1:
            errors.append(f"multiple reportable issues in family: {family}")
    if assessment["quality"]["status"] == "reject":
        if any(issue["decision"] == "reportable" for issue in assessment["issues"]):
            errors.append("quality reject must not produce reportable issues")
        if not assessment["summary"]["abstained"]:
            errors.append("quality reject must abstain")
    for evaluation in assessment["metric_evaluations"]:
        if evaluation["metric"] == "hip_depth_ratio":
            value = evaluation["raw_value"]
            if value is not None and value < 0:
                if evaluation["physical_valid"] or evaluation["decision"] in {
                    "candidate",
                    "diagnostic",
                    "reportable",
                }:
                    errors.append("negative hip_depth_ratio must fail closed")
        if (
            evaluation["observation_status"] in ACTIVE_OBSERVATION_STATUSES
            and evaluation["decision"] in {"candidate", "diagnostic", "reportable"}
            and evaluation["anchor_frame"] is None
        ):
            errors.append(f"active evaluation lacks anchor: {evaluation['measurement_id']}")
        if (evaluation["adjusted_confidence"] or 0.0) < float(
            rules["confidence"]["suppress_below"]
        ) and evaluation["decision"] in {"candidate", "diagnostic", "reportable"}:
            errors.append(f"low-confidence evaluation did not fail closed: {evaluation['measurement_id']}")
    if assessment["summary"]["product_actions_enabled"]:
        errors.append("product actions must remain disabled")
    return errors


def annotation_rows(path: Path, allowed_clips: set[str]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row.get("clip_id") in allowed_clips
            and row.get("split") not in {"holdout"}
        ]


def label_maps(
    rows: list[dict[str, str]],
    benchmark: dict[str, Any],
    taxonomy: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    clip_map = {clip["clip_id"]: clip for clip in benchmark.get("clips") or []}
    labels: dict[tuple[str, str], dict[str, Any]] = {}
    no_obvious_clips = {
        row["clip_id"]
        for row in rows
        if row.get("issue_code") == "no_obvious_issue"
        and row.get("presence") == "absent"
    }
    for clip_id in no_obvious_clips:
        for issue_code in (taxonomy.get("issues") or {}):
            labels[(clip_id, issue_code)] = {
                "presence": "absent",
                "severity": "none",
                "scope": "clip",
                "rep_id": None,
                "evidence_start_frame": None,
                "evidence_end_frame": None,
            }
    for row in rows:
        clip = clip_map.get(row["clip_id"]) or {}
        if clip.get("split") == "holdout":
            continue
        key = (row["clip_id"], row["issue_code"])
        existing = labels.get(key)
        candidate = {
            "presence": row["presence"],
            "severity": row["severity"],
            "scope": row["scope"],
            "rep_id": row.get("rep_id") or None,
            "evidence_start_frame": row.get("evidence_start_frame") or None,
            "evidence_end_frame": row.get("evidence_end_frame") or None,
        }
        rank = {"present": 4, "uncertain": 3, "not_evaluable": 2, "absent": 1}
        if existing is None or rank.get(candidate["presence"], 0) > rank.get(
            existing["presence"], 0
        ):
            labels[key] = candidate
    return labels, clip_map


def build_label_audit(
    assessments: list[dict[str, Any]],
    labels: dict[tuple[str, str], dict[str, Any]],
    clip_map: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audit_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    for assessment in assessments:
        for issue in assessment["issues"]:
            label = labels.get((assessment["clip_id"], issue["issue_code"]), {})
            presence = label.get("presence", "missing")
            if presence == "present":
                relation = "ai_present_overlap"
            elif presence in {"absent", "not_evaluable", "uncertain"}:
                relation = f"ai_{presence}_overlap"
            else:
                relation = "ai_label_missing"
            audit_rows.append(
                {
                    "clip_id": assessment["clip_id"],
                    "issue_id": issue["issue_id"],
                    "issue_code": issue["issue_code"],
                    "decision": issue["decision"],
                    "severity": issue["severity"],
                    "confidence": issue["confidence"],
                    "ai_label_presence": presence,
                    "ai_label_severity": label.get("severity"),
                    "relation": relation,
                    "source_quality_group": (clip_map.get(assessment["clip_id"]) or {}).get(
                        "quality_group"
                    ),
                    "evidence_basis": "ai_candidate_not_expert_ground_truth",
                }
            )
        for (clip_id, issue_code), label in labels.items():
            if clip_id != assessment["clip_id"] or label["presence"] == "absent":
                continue
            matched = [
                issue
                for issue in assessment["issues"]
                if issue["issue_code"] == issue_code
            ]
            best = max(
                matched,
                key=lambda item: SEVERITY_RANK.get(item["severity"], 0),
                default=None,
            )
            gap_rows.append(
                {
                    "clip_id": clip_id,
                    "issue_code": issue_code,
                    "ai_label_presence": label["presence"],
                    "ai_label_severity": label["severity"],
                    "best_engine_decision": best["decision"] if best else "not_detected",
                    "best_engine_severity": best["severity"] if best else "none",
                    "best_engine_confidence": best["confidence"] if best else None,
                    "exploratory_only": True,
                }
            )
    return audit_rows, gap_rows


def quality_group_audit(
    assessments: list[dict[str, Any]], clip_map: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for assessment in assessments:
        clip = clip_map.get(assessment["clip_id"]) or {}
        rows.append(
            {
                "clip_id": assessment["clip_id"],
                "split": assessment["split"],
                "exercise": assessment["exercise"],
                "view": assessment["view"],
                "quality_status": assessment["quality"]["status"],
                "source_quality_group": clip.get("quality_group"),
                "candidate_issue_count": assessment["summary"]["candidate_issue_count"],
                "reportable_issue_count": assessment["summary"]["reportable_issue_count"],
                "reportable_issue_codes": ";".join(
                    assessment["summary"]["reportable_issue_codes"]
                ),
                "source_group_usage": "sampling_stratification_only_not_ground_truth",
            }
        )
    return rows


def threshold_sweep(
    measurement_payloads: list[tuple[dict[str, Any], dict[str, Any]]],
    rules: dict[str, Any],
    taxonomy: dict[str, Any],
    clip_map: dict[str, dict[str, Any]],
    labels: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    sweep = rules.get("diagnostic_sweep") or {}
    rows: list[dict[str, Any]] = []
    for scale in sweep.get("threshold_scales") or [1.0]:
        for confidence in sweep.get("confidence_thresholds") or [0.65]:
            for minimum_severity in sweep.get("min_severities") or ["high"]:
                decisions = []
                for measurement, clip in measurement_payloads:
                    decisions.append(
                        build_assessment(
                            measurement,
                            clip,
                            rules,
                            taxonomy,
                            "diagnostic_sweep",
                            threshold_scale=float(scale),
                            report_confidence=float(confidence),
                            report_min_severity=minimum_severity,
                        )
                    )
                issues = [
                    (assessment, issue)
                    for assessment in decisions
                    for issue in assessment["issues"]
                ]
                reportable = [
                    (assessment, issue)
                    for assessment, issue in issues
                    if issue["decision"] == "reportable"
                ]
                label_counts = Counter()
                for assessment, issue in reportable:
                    label = labels.get((assessment["clip_id"], issue["issue_code"]), {})
                    label_counts[label.get("presence", "missing")] += 1
                source_group_counts = Counter(
                    (clip_map.get(assessment["clip_id"]) or {}).get("quality_group")
                    for assessment, _ in reportable
                )
                rows.append(
                    {
                        "threshold_scale": scale,
                        "confidence_threshold": confidence,
                        "min_severity": minimum_severity,
                        "clip_count": len(decisions),
                        "candidate_issue_count": sum(
                            issue["decision"] in {"candidate", "reportable"}
                            for _, issue in issues
                        ),
                        "reportable_issue_count": len(reportable),
                        "clips_with_reportable_issue": len(
                            {assessment["clip_id"] for assessment, _ in reportable}
                        ),
                        "abstained_clip_count": sum(
                            assessment["summary"]["abstained"]
                            for assessment in decisions
                        ),
                        "ai_label_present_overlap": label_counts.get("present", 0),
                        "ai_label_uncertain_overlap": label_counts.get("uncertain", 0),
                        "ai_label_absent_or_no_obvious_overlap": label_counts.get(
                            "absent", 0
                        ),
                        "ai_label_missing_overlap": label_counts.get("missing", 0),
                        "source_good_group_reportable": source_group_counts.get("good", 0),
                        "source_bad_group_reportable": source_group_counts.get("bad", 0),
                        "label_usage": "exploratory_ai_candidate_only",
                    }
                )
    return rows


def select_clips(
    benchmark: dict[str, Any], splits: list[str], clip_ids: list[str] | None
) -> list[dict[str, Any]]:
    if "holdout" in splits:
        raise ValueError("assessment refuses holdout; reserve it for final evaluation only")
    wanted = set(clip_ids or [])
    clips = [
        clip
        for clip in benchmark.get("clips") or []
        if clip["split"] in splits and (not wanted or clip["clip_id"] in wanted)
    ]
    missing = wanted - {clip["clip_id"] for clip in clips}
    if missing:
        raise ValueError("requested clips are unavailable in selected splits: " + ", ".join(sorted(missing)))
    return clips


def write_checksums(output_dir: Path) -> None:
    lines = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "checksums.sha256":
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(output_dir)}")
    (output_dir / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    rules = load_json(args.rules)
    taxonomy = load_json(args.taxonomy)
    contracts = load_json(args.contracts)
    benchmark = load_json(args.benchmark)
    schema = load_json(args.schema)
    rule_errors = validate_rules(rules, taxonomy, contracts)
    if rule_errors:
        raise ValueError("invalid assessment rules:\n" + "\n".join(rule_errors))
    clips = select_clips(benchmark, args.split, args.clip_id)
    if not clips:
        raise ValueError("no clips selected")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated_at = args.generated_at or utc_now()
    measurements: list[tuple[dict[str, Any], dict[str, Any]]] = []
    assessments: list[dict[str, Any]] = []
    for clip in clips:
        measurement_path = args.measurement_dir / clip["clip_id"] / "assessment.json"
        if not measurement_path.exists():
            raise FileNotFoundError(f"measurement missing: {measurement_path}")
        measurement = load_json(measurement_path)
        if measurement["split"] != clip["split"]:
            raise ValueError(f"split mismatch for {clip['clip_id']}")
        measurement["source_measurement_sha256"] = sha256_file(measurement_path)
        assessment = build_assessment(
            measurement, clip, rules, taxonomy, generated_at
        )
        schema_errors = validate_schema(assessment, schema)
        invariant_errors = invariant_checks(assessment, rules)
        clip_output = args.output_dir / clip["clip_id"]
        clip_output.mkdir(parents=True, exist_ok=True)
        write_json(clip_output / "assessment.json", assessment)
        write_json(
            clip_output / "issues.json",
            {
                "schema_version": "vls-issues-v2",
                "clip_id": clip["clip_id"],
                "issues": assessment["issues"],
                "summary": assessment["summary"],
            },
        )
        write_csv(clip_output / "evaluations.csv", EVALUATION_FIELDS, evaluation_rows(assessment))
        write_json(
            clip_output / "checks" / "schema_validation.json",
            {
                "schema_version": "vls-assessment-decision-schema-validation-v1",
                "valid": not schema_errors,
                "errors": schema_errors,
            },
        )
        write_json(
            clip_output / "checks" / "invariants.json",
            {
                "schema_version": "vls-assessment-decision-invariants-v1",
                "valid": not invariant_errors,
                "errors": invariant_errors,
            },
        )
        if schema_errors or invariant_errors:
            raise ValueError(
                f"assessment validation failed for {clip['clip_id']}: "
                f"schema={len(schema_errors)} invariants={len(invariant_errors)}"
            )
        measurements.append((measurement, clip))
        assessments.append(assessment)

    all_evaluations = [
        row for assessment in assessments for row in evaluation_rows(assessment)
    ]
    all_issues = [row for assessment in assessments for row in issue_rows(assessment)]
    write_csv(args.output_dir / "evaluations_all.csv", EVALUATION_FIELDS, all_evaluations)
    write_csv(args.output_dir / "issues_all.csv", ISSUE_FIELDS, all_issues)

    annotations = annotation_rows(args.annotations, {clip["clip_id"] for clip in clips})
    labels, clip_map = label_maps(annotations, benchmark, taxonomy)
    label_audit, label_gaps = build_label_audit(assessments, labels, clip_map)
    write_csv(
        args.output_dir / "label_audit.csv",
        [
            "clip_id",
            "issue_id",
            "issue_code",
            "decision",
            "severity",
            "confidence",
            "ai_label_presence",
            "ai_label_severity",
            "relation",
            "source_quality_group",
            "evidence_basis",
        ],
        label_audit,
    )
    write_csv(
        args.output_dir / "label_gap_audit.csv",
        [
            "clip_id",
            "issue_code",
            "ai_label_presence",
            "ai_label_severity",
            "best_engine_decision",
            "best_engine_severity",
            "best_engine_confidence",
            "exploratory_only",
        ],
        label_gaps,
    )
    quality_rows = quality_group_audit(assessments, clip_map)
    write_csv(
        args.output_dir / "quality_group_audit.csv",
        [
            "clip_id",
            "split",
            "exercise",
            "view",
            "quality_status",
            "source_quality_group",
            "candidate_issue_count",
            "reportable_issue_count",
            "reportable_issue_codes",
            "source_group_usage",
        ],
        quality_rows,
    )

    sweep_rows: list[dict[str, Any]] = []
    if not args.skip_sweep:
        sweep_rows = threshold_sweep(
            measurements, rules, taxonomy, clip_map, labels
        )
        if sweep_rows:
            write_csv(
                args.output_dir / "threshold_curve.csv",
                list(sweep_rows[0].keys()),
                sweep_rows,
            )
        write_json(
            args.output_dir / "diagnostic_sweep.json",
            {
                "schema_version": "vls-diagnostic-sweep-v1",
                "generated_at": generated_at,
                "threshold_scales": rules["diagnostic_sweep"]["threshold_scales"],
                "confidence_thresholds": rules["diagnostic_sweep"]["confidence_thresholds"],
                "min_severities": rules["diagnostic_sweep"]["min_severities"],
                "row_count": len(sweep_rows),
                "label_usage": "exploratory_ai_candidate_only",
                "limitations": [
                    "not expert ground truth",
                    "not precision or recall",
                    "thresholds remain provisional engineering values",
                ],
            },
        )

    write_json(args.output_dir / "rules_snapshot.json", rules)
    write_json(
        args.output_dir / "thresholds_used.json",
        {
            "schema_version": "vls-thresholds-used-v1",
            "threshold_policy": rules["policy"]["threshold_policy"],
            "expert_calibrated": False,
            "freeze_status": rules["policy"]["freeze_status"],
            "thresholds": rules["thresholds"],
            "confidence_gates": rules["confidence"],
            "physical_ranges": rules["physical_ranges"],
            "metric_policy": rules["metric_policy"],
        },
    )

    reportable = [issue for issue in all_issues if issue["decision"] == "reportable"]
    issue_decision_counts = Counter(issue["decision"] for issue in all_issues)
    metric_decision_counts = Counter(item["decision"] for item in all_evaluations)
    manifest = {
        "schema_version": "vls-assessment-decision-manifest-v1",
        "component": "assessment_decision",
        "generated_at": generated_at,
        "workflow_stage": "decision_shadow",
        "rules_path": relative_path(args.rules),
        "rules_sha256": sha256_file(args.rules),
        "taxonomy_path": relative_path(args.taxonomy),
        "taxonomy_sha256": sha256_file(args.taxonomy),
        "contracts_path": relative_path(args.contracts),
        "contracts_sha256": sha256_file(args.contracts),
        "benchmark_path": relative_path(args.benchmark),
        "benchmark_sha256": sha256_file(args.benchmark),
        "schema_path": relative_path(args.schema),
        "schema_sha256": sha256_file(args.schema),
        "measurement_dir": relative_path(args.measurement_dir),
        "selected_splits": args.split,
        "holdout_used": False,
        "clip_count": len(assessments),
        "rep_count": sum(item["summary"]["rep_count"] for item in assessments),
        "metric_evaluation_count": len(all_evaluations),
        "metric_decision_counts": dict(sorted(metric_decision_counts.items())),
        "issue_count": len(all_issues),
        "issue_decision_counts": dict(sorted(issue_decision_counts.items())),
        "reportable_issue_count": len(reportable),
        "reportable_issue_codes": dict(
            sorted(Counter(issue["issue_code"] for issue in reportable).items())
        ),
        "clip_reportable_counts": {
            assessment["clip_id"]: assessment["summary"]["reportable_issue_count"]
            for assessment in assessments
            if assessment["summary"]["reportable_issue_count"]
        },
        "abstained_clip_count": sum(
            assessment["summary"]["abstained"] for assessment in assessments
        ),
        "abstain_reason_counts": dict(
            sorted(
                Counter(
                    assessment["summary"]["abstain_reason"]
                    for assessment in assessments
                    if assessment["summary"]["abstain_reason"]
                ).items()
            )
        ),
        "source_quality_group_counts": dict(
            sorted(Counter(row["source_quality_group"] for row in quality_rows).items())
        ),
        "label_usage": "exploratory_ai_candidate_not_expert_ground_truth",
        "limitations": {
            "expert_calibrated": False,
            "product_actions_enabled": False,
            "input_is_repchain_v3": all(
                assessment["source_measurement"]["input_contract"]["upstream_contract"][
                    "is_repchain_v3"
                ]
                for assessment in assessments
            ),
            "holdout_excluded": True,
        },
        "clips": [
            {
                "clip_id": assessment["clip_id"],
                "split": assessment["split"],
                "exercise": assessment["exercise"],
                "view": assessment["view"],
                "quality_status": assessment["quality"]["status"],
                "rep_count": assessment["summary"]["rep_count"],
                "candidate_issue_count": assessment["summary"]["candidate_issue_count"],
                "reportable_issue_count": assessment["summary"]["reportable_issue_count"],
                "abstained": assessment["summary"]["abstained"],
                "abstain_reason": assessment["summary"]["abstain_reason"],
            }
            for assessment in assessments
        ],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    write_checksums(args.output_dir)
    print(
        f"assessment: clips={len(assessments)} evaluations={len(all_evaluations)} "
        f"issues={len(all_issues)} reportable={len(reportable)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
