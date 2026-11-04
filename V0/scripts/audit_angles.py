#!/usr/bin/env python3
"""Audit task1-2 angle tables, metadata, and body-frame landmarks."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANGLES_DIR = ROOT / "outputs" / "angles"
LONG_FIELDS = [
    "frame_id",
    "timestamp_ms",
    "joint",
    "feature",
    "angle_name",
    "plane",
    "unit",
    "angle",
    "angle_raw",
    "confidence",
    "source",
    "valid",
    "outlier",
    "invalid_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angles-dir", type=Path, default=DEFAULT_ANGLES_DIR)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def check(condition: bool, errors: list[str], message: str, checks: list[int]) -> None:
    checks[0] += 1
    if not condition:
        errors.append(message)


def load_vocabulary(path: Path, errors: list[str], checks: list[int]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text())
    definitions = payload.get("angles", [])
    names = [item.get("name") for item in definitions]
    check(len(definitions) > 0, errors, "angle_vocabulary.json has no angles", checks)
    check(len(names) == len(set(names)), errors, "angle vocabulary contains duplicate names", checks)
    check(all(isinstance(name, str) and name for name in names), errors, "angle vocabulary has an invalid name", checks)
    required = {"name", "joint", "feature", "plane", "unit", "landmarks", "description"}
    for item in definitions:
        check(required <= set(item), errors, f"vocabulary entry missing fields: {item.get('name')}", checks)
        check(item.get("unit") in {"deg", "m", "variance"}, errors, f"unexpected unit: {item.get('name')}", checks)
        check(
            isinstance(item.get("landmarks"), list) and len(item["landmarks"]) >= 2,
            errors,
            f"invalid landmark list: {item.get('name')}",
            checks,
        )
    by_name = {item["name"]: item for item in definitions}

    csv_path = path.with_suffix(".csv")
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        csv_rows = list(reader)
    check(reader.fieldnames == ["name", "joint", "feature", "plane", "unit", "landmarks", "description"],
          errors, "angle_vocabulary.csv header mismatch", checks)
    check(len(csv_rows) == len(names), errors, "angle_vocabulary.csv row count mismatch", checks)
    check([row["name"] for row in csv_rows] == names, errors, "angle_vocabulary.csv order mismatch", checks)
    return names, by_name


def make_body_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    shoulder_mid = (points[11] + points[12]) / 2.0
    hip_mid = (points[23] + points[24]) / 2.0
    y_norm = float(np.linalg.norm(hip_mid - shoulder_mid))
    if y_norm < 1e-8:
        return None
    y_axis = (hip_mid - shoulder_mid) / y_norm
    x_raw = points[12] - points[11]
    x_residual = x_raw - np.dot(x_raw, y_axis) * y_axis
    x_norm = float(np.linalg.norm(x_residual))
    if x_norm < 1e-8:
        return None
    x_axis = x_residual / x_norm
    z_raw = np.cross(x_axis, y_axis)
    z_norm = float(np.linalg.norm(z_raw))
    if z_norm < 1e-8:
        return None
    axes = np.stack([x_axis, y_axis, z_raw / z_norm], axis=0)
    return axes, hip_mid


def read_wide(
    path: Path,
    names: list[str],
    records: list[dict[str, Any]],
    errors: list[str],
    checks: list[int],
) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)
    check(columns == ["frame_id", "timestamp_ms", *names], errors, f"{path.parent.name}: wide header mismatch", checks)
    check(len(rows) == len(records), errors, f"{path.parent.name}: wide row count mismatch", checks)
    values = {name: np.full(len(rows), np.nan, dtype=np.float64) for name in names}
    previous_timestamp = -1
    for index, (row, record) in enumerate(zip(rows, records)):
        check(row.get("frame_id") == str(record["frame_id"]), errors,
              f"{path.parent.name}: wide frame mismatch at row {index}", checks)
        timestamp = int(row["timestamp_ms"])
        check(timestamp == int(record["timestamp_ms"]), errors,
              f"{path.parent.name}: wide timestamp mismatch at row {index}", checks)
        check(timestamp > previous_timestamp, errors,
              f"{path.parent.name}: wide timestamp not increasing at row {index}", checks)
        previous_timestamp = timestamp
        for name in names:
            values[name][index] = float(row[name])
    return values


def read_long(
    path: Path,
    names: list[str],
    vocabulary: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
    errors: list[str],
    checks: list[int],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list[str]], dict]:
    count = len(records)
    width = len(names)
    values = {name: np.full(count, np.nan, dtype=np.float64) for name in names}
    confidence = {name: np.full(count, np.nan, dtype=np.float64) for name in names}
    outliers = {name: np.zeros(count, dtype=bool) for name in names}
    reasons = {name: ["" for _ in range(count)] for name in names}
    reason_counts = {name: Counter() for name in names}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        check(reader.fieldnames == LONG_FIELDS, errors, f"{path.parent.name}: long header mismatch", checks)
        row_count = 0
        for row_count, row in enumerate(reader, start=1):
            zero_index = row_count - 1
            record_index = zero_index // width
            name_index = zero_index % width
            if record_index >= count:
                errors.append(f"{path.parent.name}: long table has surplus rows")
                break
            record = records[record_index]
            name = names[name_index]
            definition = vocabulary[name]
            check(row["frame_id"] == str(record["frame_id"]), errors,
                  f"{path.parent.name}: long frame mismatch at row {zero_index}", checks)
            check(row["timestamp_ms"] == str(record["timestamp_ms"]), errors,
                  f"{path.parent.name}: long timestamp mismatch at row {zero_index}", checks)
            check(row["angle_name"] == name, errors,
                  f"{path.parent.name}: long angle order mismatch at row {zero_index}", checks)
            for field in ("joint", "feature", "plane", "unit"):
                check(row[field] == definition[field], errors,
                      f"{path.parent.name}: {name} {field} mismatch", checks)
            check(row["source"] == "vision", errors, f"{path.parent.name}: {name} source is not vision", checks)
            valid_flag = row["valid"] == "1"
            outlier_flag = row["outlier"] == "1"
            check(row["valid"] in {"0", "1"}, errors, f"{path.parent.name}: invalid valid flag", checks)
            check(row["outlier"] in {"0", "1"}, errors, f"{path.parent.name}: invalid outlier flag", checks)
            value = float(row["angle"])
            raw_value = float(row["angle_raw"])
            conf_value = float(row["confidence"])
            check(valid_flag == np.isfinite(value), errors,
                  f"{path.parent.name}: {name} valid flag mismatch at frame {record['frame_id']}", checks)
            check(not (outlier_flag and valid_flag), errors,
                  f"{path.parent.name}: {name} outlier is marked valid", checks)
            if np.isfinite(conf_value):
                check(0.0 <= conf_value <= 1.0, errors,
                      f"{path.parent.name}: {name} confidence outside [0,1]", checks)
            for candidate, label in ((value, "angle"), (raw_value, "angle_raw")):
                if np.isfinite(candidate):
                    if definition["unit"] == "deg":
                        check(0.0 <= candidate <= 180.0, errors,
                              f"{path.parent.name}: {name} {label} outside [0,180]", checks)
                    else:
                        check(candidate >= 0.0, errors,
                              f"{path.parent.name}: {name} {label} is negative", checks)
            reason = row["invalid_reason"]
            check(valid_flag or bool(reason), errors,
                  f"{path.parent.name}: {name} invalid without reason at frame {record['frame_id']}", checks)
            check(not valid_flag or not reason, errors,
                  f"{path.parent.name}: {name} valid row has reason", checks)
            if outlier_flag:
                check(reason.startswith("angle_jump_gt_"), errors,
                      f"{path.parent.name}: {name} outlier missing jump reason", checks)
            if reason:
                reason_counts[name][reason] += 1
            values[name][record_index] = value
            confidence[name][record_index] = conf_value
            outliers[name][record_index] = outlier_flag
            reasons[name][record_index] = reason
    expected_rows = count * width
    check(row_count == expected_rows, errors,
          f"{path.parent.name}: expected {expected_rows} long rows, got {row_count}", checks)
    stats = {
        "rows": row_count,
        "reason_counts": {name: dict(counts) for name, counts in reason_counts.items()},
    }
    return values, confidence, outliers, reasons, stats


def check_meta(
    meta: dict[str, Any],
    names: list[str],
    values: dict[str, np.ndarray],
    outliers: dict[str, np.ndarray],
    reasons: dict[str, list[str]],
    records: list[dict[str, Any]],
    errors: list[str],
    checks: list[int],
) -> None:
    count = len(records)
    check(meta.get("task") == "task1-2", errors, "meta task mismatch", checks)
    check(meta.get("frame_count") == count, errors, "meta frame_count mismatch", checks)
    check(meta.get("angle_count") == len(names), errors, "meta angle_count mismatch", checks)
    check(meta.get("visibility_threshold") == 0.5, errors, "unexpected visibility threshold", checks)
    check(meta.get("smoothing", {}).get("method") == "savitzky_golay", errors, "unexpected smoothing method", checks)
    check(meta.get("smoothing", {}).get("window") == 7, errors, "unexpected smoothing window", checks)
    check(meta.get("smoothing", {}).get("polyorder") == 2, errors, "unexpected smoothing polyorder", checks)
    check(meta.get("outlier_rule", {}).get("threshold_per_frame") == 30.0,
          errors, "unexpected outlier threshold", checks)
    quality = meta.get("quality", [])
    check(len(quality) == len(names), errors, "meta quality row count mismatch", checks)
    quality_by_name = {row["name"]: row for row in quality}
    check(set(quality_by_name) == set(names), errors, "meta quality names mismatch", checks)
    no_pose_frames = sum(not record.get("detected", False) for record in records)
    for name in names:
        row = quality_by_name.get(name)
        if row is None:
            continue
        valid = np.isfinite(values[name])
        nan_count = int((~valid).sum())
        reason_counts = Counter(reason for reason in reasons[name] if reason)
        check(row.get("valid_frames") == int(valid.sum()), errors, f"{name}: valid_frames mismatch", checks)
        check(row.get("nan_frames") == nan_count, errors, f"{name}: nan_frames mismatch", checks)
        check(abs(row.get("nan_ratio", -1.0) - nan_count / count) < 1e-15,
              errors, f"{name}: nan_ratio mismatch", checks)
        check(row.get("outlier_count") == int(outliers[name].sum()), errors,
              f"{name}: outlier_count mismatch", checks)
        check(row.get("low_visibility_count") == reason_counts.get("low_visibility", 0), errors,
              f"{name}: low_visibility_count mismatch", checks)
        check(row.get("reason_counts") == dict(reason_counts), errors,
              f"{name}: reason_counts mismatch", checks)
        check(reason_counts.get("no_pose", 0) == no_pose_frames, errors,
              f"{name}: no_pose count mismatch", checks)


def check_body_frame(
    path: Path,
    records: list[dict[str, Any]],
    errors: list[str],
    checks: list[int],
) -> dict[str, float]:
    body = json.loads(path.read_text())
    check(len(body) == len(records), errors, "body-frame frame count mismatch", checks)
    max_orthogonality_error = 0.0
    max_determinant_error = 0.0
    max_roundtrip_error = 0.0
    body_frames = 0
    previous_timestamp = -1
    for index, (record, entry) in enumerate(zip(records, body)):
        check(entry.get("frame_id") == record.get("frame_id"), errors,
              f"body-frame frame id mismatch at {index}", checks)
        timestamp = int(entry.get("timestamp_ms", -1))
        check(timestamp == int(record.get("timestamp_ms", -2)), errors,
              f"body-frame timestamp mismatch at {index}", checks)
        check(timestamp > previous_timestamp, errors,
              f"body-frame timestamp not increasing at {index}", checks)
        previous_timestamp = timestamp
        detected = bool(record.get("detected"))
        check(entry.get("detected") == detected, errors,
              f"body-frame detected flag mismatch at {index}", checks)
        points_payload = entry.get("landmarks_body_frame")
        if not detected:
            check(points_payload is None, errors, f"body-frame points present without pose at {index}", checks)
            continue
        if points_payload is None:
            errors.append(f"body-frame points missing at detected frame {index}")
            continue
        body_points = np.asarray(points_payload, dtype=np.float64)
        check(body_points.shape == (33, 3), errors, f"body-frame shape is not 33x3 at {index}", checks)
        check(np.isfinite(body_points).all(), errors, f"body-frame has non-finite values at {index}", checks)
        raw_points = np.asarray(record["landmarks_world"], dtype=np.float64)
        frame = make_body_axes(raw_points)
        if frame is None:
            errors.append(f"source landmarks cannot construct body frame at {index}")
            continue
        axes, hip_mid = frame
        orthogonality = axes @ axes.T
        orthogonality_error = float(np.max(np.abs(orthogonality - np.eye(3))))
        determinant_error = abs(float(np.linalg.det(axes)) - 1.0)
        expected = np.round((raw_points - hip_mid) @ axes.T, 6)
        roundtrip_error = float(np.max(np.abs(body_points - expected)))
        max_orthogonality_error = max(max_orthogonality_error, orthogonality_error)
        max_determinant_error = max(max_determinant_error, determinant_error)
        max_roundtrip_error = max(max_roundtrip_error, roundtrip_error)
        check(orthogonality_error < 1e-8, errors, f"body axes not orthonormal at {index}", checks)
        check(determinant_error < 1e-8, errors, f"body axes are not right-handed at {index}", checks)
        check(roundtrip_error < 1.1e-6, errors, f"body-frame transform mismatch at {index}", checks)
        body_frames += 1
    check(body_frames == sum(bool(record.get("detected")) for record in records), errors,
          "body-frame available count differs from detected count", checks)
    return {
        "body_frames": body_frames,
        "max_orthogonality_error": max_orthogonality_error,
        "max_determinant_error": max_determinant_error,
        "max_roundtrip_error": max_roundtrip_error,
    }


def check_plots(plots_dir: Path, warnings: list[str], errors: list[str], checks: list[int]) -> int:
    index_path = plots_dir / "index.json"
    if not index_path.exists():
        warnings.append("plot index is missing; run plot_angles.py")
        return 0
    payload = json.loads(index_path.read_text())
    plots = payload.get("plots", [])
    check(len(plots) > 0, errors, "plot index contains no plots", checks)
    for item in plots:
        plot_path = Path(item.get("path", ""))
        check(plot_path.exists() and plot_path.stat().st_size > 0, errors,
              f"plot file missing or empty: {plot_path}", checks)
        check(2 <= len(item.get("angles", [])) <= 3, errors,
              f"plot should contain 2-3 angle curves: {plot_path.name}", checks)
    return len(plots)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    checks = [0]
    names, vocabulary = load_vocabulary(args.angles_dir / "angle_vocabulary.json", errors, checks)
    summary = json.loads((args.angles_dir / "summary.json").read_text())
    metas = summary.get("videos", [])
    check(summary.get("task") == "task1-2", errors, "summary task mismatch", checks)
    check(summary.get("angle_count") == len(names), errors, "summary angle_count mismatch", checks)
    check(summary.get("totals", {}).get("videos") == len(metas), errors,
          "summary video total mismatch", checks)
    check(summary.get("totals", {}).get("frames") == sum(meta["frame_count"] for meta in metas), errors,
          "summary frame total mismatch", checks)
    check(summary.get("totals", {}).get("angle_rows") == sum(meta["frame_count"] for meta in metas) * len(names),
          errors, "summary angle row total mismatch", checks)

    video_reports: dict[str, dict[str, Any]] = {}
    total_rows = 0
    total_outliers = 0
    total_low_visibility = 0
    for meta in metas:
        video_id = meta["video_id"]
        video_dir = args.angles_dir / video_id
        records = json.loads((video_dir / "landmarks.json").read_text())
        check(len(records) == meta["frame_count"], errors, f"{video_id}: source frame count mismatch", checks)
        previous_timestamp = -1
        for index, record in enumerate(records):
            check(record.get("frame_id") == index, errors, f"{video_id}: source frame id mismatch at {index}", checks)
            timestamp = int(record["timestamp_ms"])
            check(timestamp > previous_timestamp, errors,
                  f"{video_id}: source timestamp not increasing at {index}", checks)
            previous_timestamp = timestamp
        values, _, outliers, reasons, long_stats = read_long(
            video_dir / "angles.csv", names, vocabulary, records, errors, checks
        )
        wide_values = read_wide(video_dir / "angles_wide.csv", names, records, errors, checks)
        for name in names:
            check(np.allclose(values[name], wide_values[name], equal_nan=True, atol=1e-6),
                  errors, f"{video_id}: {name} differs between long and wide tables", checks)
        check_meta(meta, names, values, outliers, reasons, records, errors, checks)
        body_stats = check_body_frame(video_dir / "landmarks_body_frame.json", records, errors, checks)
        nan_ratios = [float(np.mean(~np.isfinite(values[name]))) for name in names]
        outlier_total = sum(int(outliers[name].sum()) for name in names)
        low_visibility_total = sum(
            long_stats["reason_counts"][name].get("low_visibility", 0) for name in names
        )
        total_rows += long_stats["rows"]
        total_outliers += outlier_total
        total_low_visibility += low_visibility_total
        video_reports[video_id] = {
            "frame_count": len(records),
            "angle_count": len(names),
            "long_rows": long_stats["rows"],
            "wide_columns": 2 + len(names),
            "mean_nan_ratio": float(np.mean(nan_ratios)),
            "max_nan_ratio": float(np.max(nan_ratios)),
            "max_nan_angle": names[int(np.argmax(nan_ratios))],
            "outlier_count": outlier_total,
            "low_visibility_count": low_visibility_total,
            **body_stats,
        }

    quality_rows = list(csv.DictReader((args.angles_dir / "quality_report.csv").open(newline="")))
    expected_quality_rows = len(metas) * len(names)
    check(len(quality_rows) == expected_quality_rows, errors,
          f"quality_report.csv expected {expected_quality_rows} rows, got {len(quality_rows)}", checks)
    meta_by_video = {meta["video_id"]: meta for meta in metas}
    for row in quality_rows:
        video_id = row["video_id"]
        name = row["angle_name"]
        meta_quality = {item["name"]: item for item in meta_by_video.get(video_id, {}).get("quality", [])}
        source = meta_quality.get(name)
        check(source is not None, errors, f"quality_report has unknown row {video_id}/{name}", checks)
        if source is None:
            continue
        for key in ("valid_frames", "nan_frames", "outlier_count", "low_visibility_count"):
            check(int(row[key]) == int(source[key]), errors,
                  f"quality_report mismatch for {video_id}/{name}/{key}", checks)
        check(abs(float(row["nan_ratio"]) - float(source["nan_ratio"])) < 1e-15, errors,
              f"quality_report nan_ratio mismatch for {video_id}/{name}", checks)
        check(row["unit"] == source["unit"], errors,
              f"quality_report unit mismatch for {video_id}/{name}", checks)

    plot_count = check_plots(args.angles_dir / "plots", warnings, errors, checks)
    return {
        "task": "task1-2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "angles_dir": str(args.angles_dir.resolve()),
        "status": "pass" if not errors else "fail",
        "checks_executed": checks[0],
        "summary": {
            "videos": len(metas),
            "frames": sum(meta["frame_count"] for meta in metas),
            "angles": len(names),
            "long_rows": total_rows,
            "outliers": total_outliers,
            "low_visibility_rows": total_low_visibility,
            "plots": plot_count,
        },
        "errors": errors,
        "warnings": warnings,
        "videos": video_reports,
    }


def main() -> int:
    args = parse_args()
    args.angles_dir = args.angles_dir.resolve()
    output = (args.output or (args.angles_dir / "audit.json")).resolve()
    report = audit(args)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "checks_executed": report["checks_executed"],
        "summary": report["summary"],
        "warnings": report["warnings"],
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    if report["errors"]:
        for message in report["errors"][:50]:
            print(f"ERROR: {message}")
        if len(report["errors"]) > 50:
            print(f"ERROR: ... {len(report['errors']) - 50} more")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
