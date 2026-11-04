#!/usr/bin/env python3
"""Build the T1-6E failure index, stratified tables, and SVG audit figures."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import task1_6_phase_states as phase_states


DEFAULT_EVALUATION_DIR = ROOT / "results" / "task1_6" / "evaluation"
DEFAULT_TABLES_DIR = ROOT / "results" / "task1_6" / "tables"
DEFAULT_FIGURES_DIR = ROOT / "results" / "task1_6" / "figures"
SCHEMA_VERSION = "task1-6e-failure-audit-v1"

FAILURE_LABELS = {
    "first_event_miss": "first event missed",
    "last_event_miss": "last event missed",
    "count_error": "count error",
    "event_high_conf_fp": "event high-confidence FP",
    "rep_high_conf_fp": "complete-rep high-confidence FP",
    "event_median_exceeds_gate": "primary/all-camera median error",
    "event_p90_exceeds_gate": "primary/all-camera p90 error",
    "no_confident_set": "no confident set",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-dir", type=Path, default=DEFAULT_EVALUATION_DIR
    )
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return phase_states.load_json(path)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_bool(value: Any) -> bool:
    return str(value).lower() == "true"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    phase_states.write_csv(path, rows, fields)


def write_json(path: Path, payload: Any) -> None:
    phase_states.write_json(path, payload)


def clip_failure_categories(
    row: dict[str, str],
    *,
    is_primary: bool,
) -> list[str]:
    categories = []
    if not as_bool(row.get("first_gt_matched")):
        categories.append("first_event_miss")
    if not as_bool(row.get("last_gt_matched")):
        categories.append("last_event_miss")
    if int(row.get("count_error") or 0) != 0:
        categories.append("count_error")
    if int(row.get("high_confidence_event_fp") or 0) > 0:
        categories.append("event_high_conf_fp")
    if int(row.get("high_confidence_rep_fp") or 0) > 0:
        categories.append("rep_high_conf_fp")
    if as_bool(row.get("no_confident_set")):
        categories.append("no_confident_set")
    median = as_float(row.get("event_abs_error_median_frames"))
    p90 = as_float(row.get("event_abs_error_p90_frames"))
    if median is not None and median > 5.0:
        categories.append("event_median_exceeds_gate")
    if p90 is not None and p90 > 12.0:
        categories.append("event_p90_exceeds_gate")
    if not is_primary:
        categories = [
            category
            for category in categories
            if category
            not in {"event_median_exceeds_gate", "event_p90_exceeds_gate"}
        ]
    return categories


def severity_score(row: dict[str, str], categories: list[str]) -> float:
    score = 0.0
    if "last_event_miss" in categories:
        score += 3.0
    if "first_event_miss" in categories:
        score += 2.0
    if "no_confident_set" in categories:
        score += 5.0
    score += 2.0 * int(row.get("high_confidence_rep_fp") or 0)
    score += 1.0 * int(row.get("high_confidence_event_fp") or 0)
    score += abs(int(row.get("count_error") or 0))
    p90 = as_float(row.get("event_abs_error_p90_frames"))
    if p90 is not None:
        score += max(0.0, (p90 - 12.0) / 5.0)
    return round(score, 6)


def evidence_frames(
    clip_id: str,
    matches: list[dict[str, str]],
    categories: list[str],
) -> tuple[int | None, int | None, str | None]:
    clip_rows = [row for row in matches if row["clip_id"] == clip_id]
    if "last_event_miss" in categories:
        failures = [
            row
            for row in clip_rows
            if row["match_status"] == "false_negative" and row.get("gt_frame")
        ]
        if failures:
            row = max(failures, key=lambda item: int(item["gt_frame"]))
            return int(row["gt_frame"]), None, "last_false_negative"
    if "event_high_conf_fp" in categories:
        failures = [
            row
            for row in clip_rows
            if row["match_status"] == "false_positive"
            and as_bool(row.get("high_confidence_false_positive"))
        ]
        if failures:
            row = max(failures, key=lambda item: int(item["pred_frame"]))
            return None, int(row["pred_frame"]), "high_confidence_false_positive"
    if "first_event_miss" in categories:
        failures = [
            row
            for row in clip_rows
            if row["match_status"] == "false_negative" and row.get("gt_frame")
        ]
        if failures:
            row = min(failures, key=lambda item: int(item["gt_frame"]))
            return int(row["gt_frame"]), None, "first_false_negative"
    matched = [row for row in clip_rows if row["match_status"] == "matched"]
    if matched:
        row = max(
            matched,
            key=lambda item: int(item.get("abs_error_frames") or 0),
        )
        return (
            int(row["gt_frame"]) if row.get("gt_frame") not in (None, "") else None,
            int(row["pred_frame"])
            if row.get("pred_frame") not in (None, "")
            else None,
            "largest_matched_error",
        )
    return None, None, None


def build_failure_index(
    clip_rows: list[dict[str, str]],
    match_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    output = []
    for row in clip_rows:
        is_primary = as_bool(row.get("is_primary_camera"))
        categories = clip_failure_categories(row, is_primary=is_primary)
        score = severity_score(row, categories)
        gt_frame, pred_frame, evidence_kind = evidence_frames(
            row["clip_id"], match_rows, categories
        )
        output.append(
            {
                "clip_id": row["clip_id"],
                "cluster_id": row["cluster_id"],
                "subject_id": row["subject_id"],
                "camera_id": row["camera_id"],
                "action": row["action"],
                "view": row["view"],
                "is_primary_camera": is_primary,
                "count_error": int(row.get("count_error") or 0),
                "first_gt_matched": as_bool(row.get("first_gt_matched")),
                "last_gt_matched": as_bool(row.get("last_gt_matched")),
                "event_abs_error_median_frames": as_float(
                    row.get("event_abs_error_median_frames")
                ),
                "event_abs_error_p90_frames": as_float(
                    row.get("event_abs_error_p90_frames")
                ),
                "high_confidence_event_fp": int(
                    row.get("high_confidence_event_fp") or 0
                ),
                "high_confidence_rep_fp": int(
                    row.get("high_confidence_rep_fp") or 0
                ),
                "failure_categories": "|".join(categories),
                "failure_labels": " | ".join(
                    FAILURE_LABELS[category] for category in categories
                ),
                "severity_score": score,
                "priority": (
                    "high"
                    if score >= 4.0
                    else "medium"
                    if score >= 1.0
                    else "low"
                    if score > 0.0
                    else "none"
                ),
                "evidence_kind": evidence_kind,
                "evidence_gt_frame": gt_frame,
                "evidence_pred_frame": pred_frame,
                "failure_count": len(categories),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            -float(row["severity_score"]),
            row["action"],
            row["clip_id"],
        ),
    )


def stratified_failures(
    failure_rows: list[dict[str, Any]], field: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in failure_rows:
        grouped[str(row.get(field))].append(row)
    output = []
    for value, rows in sorted(grouped.items()):
        output.append(
            {
                field: value,
                "clip_count": len(rows),
                "failed_clip_count": sum(row["failure_count"] > 0 for row in rows),
                "failed_clip_rate": phase_states.ratio(
                    sum(row["failure_count"] > 0 for row in rows), len(rows)
                ),
                "primary_failure_count": sum(
                    bool(row["is_primary_camera"]) and row["failure_count"] > 0
                    for row in rows
                ),
                "event_high_conf_fp": sum(
                    int(row["high_confidence_event_fp"]) for row in rows
                ),
                "rep_high_conf_fp": sum(
                    int(row["high_confidence_rep_fp"]) for row in rows
                ),
                "max_severity_score": max(
                    float(row["severity_score"]) for row in rows
                ),
            }
        )
    return output


def build_qevd_review(
    sealed_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    output = []
    for row in sealed_rows:
        status = row["status"]
        if as_bool(row.get("excluded_by_quality")):
            action = "excluded_by_quality_gate"
        elif str(row.get("quality_action")) == "blocked":
            action = "excluded_by_quality_gate"
        elif as_bool(row.get("excluded_by_motion")) or status == "no_motion":
            action = "motion_refused_no_reportable_reps"
        elif status in {"rejected_quality", "not_evaluable_quality"}:
            action = "review_quality_evidence_before_acceptance"
        elif status == "ok":
            if as_bool(row.get("quality_gate_conflict")):
                action = "review_quality_gate_and_annotate_before_acceptance"
            else:
                action = "pending_human_annotation"
        elif status == "unsupported_action_config":
            action = "add_frozen_action_contract_or_confirm_zero_rep_expectation"
        elif status == "preprocessed_input_missing":
            action = "generate_preprocessed_input"
        else:
            action = "manual_review"
        output.append(
            {
                "clip_id": row["clip_id"],
                "action": row["action"],
                "selection_role": row.get("selection_role"),
                "status": status,
                "input_quality_status": row.get("input_quality_status"),
                "quality_action": row.get("quality_action"),
                "quality_reasons": row.get("quality_reasons"),
                "excluded_by_quality": as_bool(row.get("excluded_by_quality")),
                "excluded_by_motion": as_bool(row.get("excluded_by_motion")),
                "raw_set_count": row.get("raw_set_count") or 0,
                "raw_complete_rep_count": row.get("raw_complete_rep_count") or 0,
                "set_count": row.get("set_count") or 0,
                "complete_rep_count": row.get("complete_rep_count") or 0,
                "high_confidence_complete_rep_count": (
                    row.get("high_confidence_complete_rep_count") or 0
                ),
                "quality_gate_conflict": as_bool(row.get("quality_gate_conflict")),
                "formal_metric_evaluable": as_bool(row.get("gt_evaluable")),
                "required_action": action,
            }
        )
    return output


def svg_bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    colors: list[str],
    *,
    value_suffix: str = "",
    display_values: list[str] | None = None,
) -> None:
    width = 920
    row_height = 30
    margin_left = 330
    margin_top = 54
    height = margin_top + row_height * max(1, len(labels)) + 30
    maximum = max([1.0] + values)
    if display_values is None:
        display_values = [f"{value:g}{value_suffix}" for value in values]
    bars = []
    for index, (label, value, color, display_value) in enumerate(
        zip(labels, values, colors, display_values)
    ):
        y = margin_top + index * row_height
        bar_width = max(1.0, (width - margin_left - 100) * value / maximum)
        bars.append(
            f'<text x="16" y="{y + 18}" font-size="13" fill="#243447">'
            f'{html.escape(label)}</text>'
            f'<rect x="{margin_left}" y="{y + 4}" width="{bar_width:.2f}" '
            f'height="19" rx="3" fill="{color}"/>'
            f'<text x="{margin_left + bar_width + 8}" y="{y + 18}" '
            f'font-size="12" fill="#243447">{html.escape(display_value)}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#f7f9fc"/>'
        f'<text x="16" y="30" font-size="20" font-weight="700" '
        f'fill="#14213d">{html.escape(title)}</text>'
        + "".join(bars)
        + "</svg>\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg)


def build_figures(
    figures_dir: Path,
    *,
    go_no_go: list[dict[str, str]],
    failure_rows: list[dict[str, Any]],
    robustness_summary: list[dict[str, str]],
) -> None:
    gate_colors = {
        "pass": "#2a9d8f",
        "fail": "#d1495b",
        "not_evaluable": "#8d99ae",
    }
    gate_positions = {"pass": 1.0, "fail": 0.0, "not_evaluable": 0.5}
    svg_bar_chart(
        figures_dir / "holdout_gates.svg",
        "T1-6E fixed gates - status only, not a numeric score",
        [f"{row['dataset']}: {row['gate_id']}" for row in go_no_go],
        [gate_positions[row["status"]] for row in go_no_go],
        [gate_colors[row["status"]] for row in go_no_go],
        display_values=[row["status"].replace("_", " ").upper() for row in go_no_go],
    )
    category_counts = Counter()
    for row in failure_rows:
        for category in str(row["failure_categories"]).split("|"):
            if category:
                category_counts[category] += 1
    categories = sorted(category_counts)
    svg_bar_chart(
        figures_dir / "failure_categories.svg",
        "Fit3D holdout failure categories",
        [FAILURE_LABELS[category] for category in categories],
        [float(category_counts[category]) for category in categories],
        ["#e76f51"] * len(categories),
    )
    svg_bar_chart(
        figures_dir / "robustness_structure_match.svg",
        "Structure-match rate under frozen transformations",
        [row["transform_id"] for row in robustness_summary],
        [100.0 * float(row["structure_match_rate"] or 0.0) for row in robustness_summary],
        ["#457b9d"] * len(robustness_summary),
        value_suffix="%",
    )


def main() -> int:
    args = parse_args()
    clip_rows = load_csv(args.evaluation_dir / "fit3d_holdout_clip_metrics.csv")
    match_rows = load_csv(args.evaluation_dir / "fit3d_holdout_event_matches.csv")
    go_no_go = load_csv(args.evaluation_dir / "go_no_go.csv")
    robustness_summary = load_csv(
        args.evaluation_dir / "robustness_summary.csv"
    )
    negative_controls = load_csv(args.evaluation_dir / "negative_controls.csv")
    sealed_rows = load_csv(args.evaluation_dir / "qevd_sealed_predictions.csv")
    summary = load_json(args.evaluation_dir / "summary.json")
    failure_rows = build_failure_index(clip_rows, match_rows)
    action_rows = stratified_failures(failure_rows, "action")
    view_rows = stratified_failures(failure_rows, "view")
    cluster_rows = stratified_failures(failure_rows, "cluster_id")
    qevd_review = build_qevd_review(sealed_rows)
    failure_fields = list(failure_rows[0]) if failure_rows else []
    action_fields = list(action_rows[0]) if action_rows else []
    view_fields = list(view_rows[0]) if view_rows else []
    cluster_fields = list(cluster_rows[0]) if cluster_rows else []
    qevd_fields = list(qevd_review[0]) if qevd_review else []
    write_csv(args.evaluation_dir / "failure_index.csv", failure_rows, failure_fields)
    write_csv(args.tables_dir / "failure_index.csv", failure_rows, failure_fields)
    write_csv(
        args.tables_dir / "failure_by_action.csv", action_rows, action_fields
    )
    write_csv(args.tables_dir / "failure_by_view.csv", view_rows, view_fields)
    write_csv(
        args.tables_dir / "failure_by_cluster.csv", cluster_rows, cluster_fields
    )
    write_csv(
        args.evaluation_dir / "qevd_review_index.csv",
        qevd_review,
        qevd_fields,
    )
    build_figures(
        args.figures_dir,
        go_no_go=go_no_go,
        failure_rows=failure_rows,
        robustness_summary=robustness_summary,
    )
    high_priority = [row for row in failure_rows if row["priority"] == "high"]
    last_event_recall = (
        summary.get("fit3d_holdout", {})
        .get("all_cameras", {})
        .get("last_event_recall")
    )
    last_recall_text = (
        f"{last_event_recall:g}"
        if isinstance(last_event_recall, (int, float))
        else "unknown"
    )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "task": "T1-6E",
        "generated_at": phase_states.utc_now(),
        "source": {
            "evaluation_summary": str(
                (args.evaluation_dir / "summary.json").resolve()
            ),
            "evaluation_summary_sha256": phase_states.sha256_path(
                args.evaluation_dir / "summary.json"
            ),
        },
        "clip_count": len(failure_rows),
        "failed_clip_count": sum(row["failure_count"] > 0 for row in failure_rows),
        "high_priority_clip_count": len(high_priority),
        "failure_category_counts": dict(sorted(Counter(
            category
            for row in failure_rows
            for category in str(row["failure_categories"]).split("|")
            if category
        ).items())),
        "top_failures": high_priority[:10],
        "negative_controls": {
            "count": len(negative_controls),
            "passed": sum(row["passed"].lower() == "true" for row in negative_controls),
            "failed_control_ids": [
                row["control_id"]
                for row in negative_controls
                if row["passed"].lower() != "true"
            ],
        },
        "qevd_review": {
            "segment_count": len(qevd_review),
            "quality_gate_conflict_count": sum(
                bool(row["quality_gate_conflict"]) for row in qevd_review
            ),
            "excluded_by_quality_count": sum(
                bool(row["excluded_by_quality"]) for row in qevd_review
            ),
            "excluded_by_motion_count": sum(
                bool(row["excluded_by_motion"]) for row in qevd_review
            ),
            "unsupported_action_count": sum(
                row["status"] == "unsupported_action_config"
                for row in qevd_review
            ),
            "formal_metric_evaluable_count": sum(
                bool(row["formal_metric_evaluable"]) for row in qevd_review
            ),
        },
        "integration_decision": summary.get("verdict", {}).get(
            "decision", "NO_GO"
        ),
        "recommended_next_actions": [
            "keep raw/gated exclusions and coverage visible before T1-7",
            (
                "improve chain tail/end anchor coverage; "
                f"last-event recall is {last_recall_text}"
            ),
            "separate anchor/confidence calibration for high-confidence event FP",
            "review quality-gate exclusions before accepting cross-domain counts",
            "complete QEVD sealed labels and add frozen contracts for 0207 controls",
        ],
    }
    write_json(args.evaluation_dir / "audit_summary.json", audit)
    write_json(args.tables_dir / "audit_summary.json", audit)
    checksum_names = sorted(
        path.name
        for path in args.evaluation_dir.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    phase_states.write_checksums(args.evaluation_dir, checksum_names)
    print(
        json.dumps(
            {
                "failed_clip_count": audit["failed_clip_count"],
                "high_priority_clip_count": audit["high_priority_clip_count"],
                "failure_category_counts": audit["failure_category_counts"],
                "qevd_review": audit["qevd_review"],
                "integration_decision": audit["integration_decision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
