#!/usr/bin/env python3
"""批量跑 testdata 视频：上传 -> 等待任务 -> 发布到 Web 样本库 -> API 校验。

用法：
    python tests/run_testdata_web_batch.py
    python tests/run_testdata_web_batch.py --only squat_front1_none squat_side4_none
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
JOBS_DIR = ROOT / "webdata" / "jobs"
MANIFEST = RESULTS / "testdata" / "testdata_clips.json"
STATE_PATH = RESULTS / "testdata" / "batch_state.json"
SUMMARY_PATH = RESULTS / "testdata" / "batch_summary.json"
RENDER_TMP_ROOT = RESULTS / "_reports_tmp"
REPORT_SCRIPT = ROOT / "scripts" / "render_clip_report.py"

BASE_URL = "http://127.0.0.1:8010"
API = f"{BASE_URL}/api/v1"

ANALYSIS_DEST = RESULTS / "clip_analysis_v1"
FEEDBACK_DEST = RESULTS / "feedback_v1"
REPORT_DEST = RESULTS / "reports_v1"

SAMPLE_FILES = ("analysis.json", "clip.json", "exercise.json", "quality.json", "reps.csv", "phases.csv")

REQUIRED_JOB_FILES = (
    ("analysis.json", "work/analysis/{job_id}/analysis.json"),
    ("clip.json", "work/analysis/{job_id}/clip.json"),
    ("exercise.json", "work/analysis/{job_id}/exercise.json"),
    ("quality.json", "work/analysis/{job_id}/quality.json"),
    ("reps.csv", "work/analysis/{job_id}/reps.csv"),
    ("phases.csv", "work/analysis/{job_id}/phases.csv"),
    ("issues.json", "work/feedback/{job_id}/issues.json"),
    ("features.csv", "work/feedback/{job_id}/features.csv"),
    ("report.html", "artifacts/{job_id}/report.html"),
    ("media", "artifacts/{job_id}/media/{job_id}_clip.webm"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_client() -> httpx.Client:
    return httpx.Client(base_url=API, timeout=httpx.Timeout(180.0, read=180.0))


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def load_clips(manifest_path: Path, only: list[str]) -> list[dict[str, Any]]:
    payload = read_json(manifest_path)
    clips = payload["clips"]
    if only:
        unknown = sorted(set(only) - {clip["clip_id"] for clip in clips})
        if unknown:
            raise ValueError("manifest 中不存在: " + ", ".join(unknown))
        clips = [clip for clip in clips if clip["clip_id"] in set(only)]
    return clips


def load_state(state_path: Path) -> dict[str, Any]:
    if state_path.is_file():
        state = read_json(state_path)
        state.setdefault("clips", {})
        return state
    return {"schema_version": "testdata-web-batch-state-v1", "clips": {}}


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    write_json(state_path, state)


def upload_job(client: httpx.Client, clip: dict[str, Any]) -> str:
    path = Path(clip["video_path"])
    if not path.is_file():
        raise FileNotFoundError(f"视频不存在: {path}")
    form = {
        "exercise": clip["exercise"],
        "view": clip["view"],
        "rotation": clip.get("rotation") or "none",
    }
    with path.open("rb") as handle:
        response = client.post(
            "/jobs/upload",
            files={"file": (path.name, handle, "video/mp4")},
            data=form,
            timeout=httpx.Timeout(300.0, read=300.0),
        )
    response.raise_for_status()
    return response.json()["job_id"]


def wait_job(client: httpx.Client, job_id: str, poll_s: float, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_stage = None
    while time.monotonic() < deadline:
        response = client.get(f"/jobs/{job_id}")
        response.raise_for_status()
        record = response.json()
        marker = f"{record['status']}/{record['stage']}"
        if marker != last_stage:
            last_stage = marker
            log(f"  job {job_id}: {marker}")
        if record["status"] in {"succeeded", "failed", "cancelled"}:
            return record
        time.sleep(poll_s)
    raise TimeoutError(f"任务 {job_id} 等待超时（>{timeout_s}s）")


def patch_json(path: Path, updates: dict[str, Any], prefix_pairs: tuple[tuple[str, str], ...] = ()) -> None:
    payload = read_json(path)

    def patch_value(value: Any) -> Any:
        if isinstance(value, str):
            for old, new in prefix_pairs:
                value = value.replace(old, new)
            return value
        if isinstance(value, dict):
            return {key: patch_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [patch_value(item) for item in value]
        return value

    payload = patch_value(payload)
    payload.update(updates)
    write_json(path, payload)


def publish_job(client: httpx.Client, job_id: str, clip: dict[str, Any]) -> dict[str, Any]:
    sample_id = clip["clip_id"]
    job_dir = JOBS_DIR / job_id
    analysis_src = job_dir / "work" / "analysis" / job_id
    feedback_src = job_dir / "work" / "feedback" / job_id
    artifacts_src = job_dir / "artifacts" / job_id

    missing = [
        str(template.format(job_id=job_id))
        for name, template in REQUIRED_JOB_FILES
        if not (job_dir / template.format(job_id=job_id)).is_file()
    ]
    if missing:
        raise FileNotFoundError("任务产物缺失: " + ", ".join(missing))

    dest_analysis = ANALYSIS_DEST / sample_id
    dest_feedback = FEEDBACK_DEST / sample_id
    dest_report = REPORT_DEST / sample_id
    for dest in (dest_analysis, dest_feedback, dest_report):
        if dest.exists():
            shutil.rmtree(dest)

    shutil.copytree(analysis_src, dest_analysis)
    shutil.copytree(feedback_src, dest_feedback)

    # analysis.json：样本 ID / split / 内部输出路径全部改回样本库
    patch_json(
        dest_analysis / "analysis.json",
        {"clip_id": sample_id, "split": "testdata"},
        ((str(analysis_src.resolve()), str(dest_analysis.resolve())),),
    )
    # clip.json：直接以生成 manifest 的条目为准，保证样本可溯源到 rawdata 视频
    clip_entry = json.loads(json.dumps(clip))
    clip_entry["processed_trim"] = dict(clip_entry["trim"])
    write_json(dest_analysis / "clip.json", clip_entry)
    missing_sample = [name for name in SAMPLE_FILES if not (dest_analysis / name).is_file()]
    if missing_sample:
        raise FileNotFoundError("样本必需文件缺失: " + ", ".join(missing_sample))

    patch_json(dest_feedback / "issues.json", {"clip_id": sample_id})

    # 用样本 ID 在临时目录重渲染报告：媒体名、源视频相对链接都与最终目录同深度
    tmp_dir = RENDER_TMP_ROOT / sample_id
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    (tmp_dir / "annotated_frames").mkdir(parents=True)
    frames_src = artifacts_src / "annotated_frames"
    if frames_src.is_dir():
        for frame in frames_src.glob("*.png"):
            shutil.copy2(frame, tmp_dir / "annotated_frames" / frame.name)
    (tmp_dir / "media").mkdir()
    shutil.copy2(
        artifacts_src / "media" / f"{job_id}_clip.webm",
        tmp_dir / "media" / f"{sample_id}_clip.webm",
    )

    subprocess.run(
        [
            sys.executable,
            str(REPORT_SCRIPT),
            "--analysis-dir", str(ANALYSIS_DEST),
            "--feedback-dir", str(FEEDBACK_DEST),
            "--output-dir", str(RENDER_TMP_ROOT),
            "--clip-id", sample_id,
            "--split", "all",
            "--skip-frames",
            "--skip-video",
        ],
        cwd=str(ROOT),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )

    dest_report.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tmp_dir / "report.html", dest_report / "report.html")
    shutil.copytree(tmp_dir / "annotated_frames", dest_report / "annotated_frames")
    shutil.copytree(tmp_dir / "media", dest_report / "media")
    shutil.rmtree(RENDER_TMP_ROOT, ignore_errors=True)

    analysis = read_json(dest_analysis / "analysis.json")
    issues = read_json(dest_feedback / "issues.json")
    issue_summary = issues.get("summary") or {}
    return {
        "sample_id": sample_id,
        "view": analysis.get("view"),
        "rep_count": int(analysis.get("rep_count") or 0),
        "rejected_rep_count": int(analysis.get("rejected_rep_count") or 0),
        "quality_status": analysis.get("quality_status"),
        "active_issue_count": int(issue_summary.get("active_issue_count") or 0),
        "suppressed_issue_count": int(issue_summary.get("suppressed_issue_count") or 0),
        "report_dir": str(dest_report),
    }


def verify_sample(client: httpx.Client, sample_id: str) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    detail_response = client.get(f"/samples/{sample_id}")
    checks["detail_200"] = detail_response.status_code == 200
    detail: dict[str, Any] = detail_response.json() if checks["detail_200"] else {}
    sample = detail.get("sample") or {}
    video = detail.get("video") or {}
    checks["has_media"] = bool(sample.get("has_media")) and bool(video.get("url"))
    checks["has_report"] = bool(sample.get("has_report")) and bool(detail.get("report_url"))

    if video.get("url"):
        media = client.get(f"{BASE_URL}{video['url']}", headers={"Range": "bytes=0-1023"})
        checks["media_range"] = media.status_code in {200, 206} and len(media.content) > 0
    else:
        checks["media_range"] = False

    if detail.get("report_url"):
        report = client.get(f"{BASE_URL}{detail['report_url']}")
        checks["report_200"] = report.status_code == 200 and "<html" in report.text.lower()
    else:
        checks["report_200"] = False

    frame_url = next((issue.get("frame_url") for issue in detail.get("issues", []) if issue.get("frame_url")), None)
    if frame_url:
        frame = client.get(f"{BASE_URL}{frame_url}")
        checks["frame_200"] = frame.status_code == 200
    else:
        checks["frame_200"] = True

    listed = client.get("/samples")
    checks["listed"] = listed.status_code == 200 and sample_id in {item["sample_id"] for item in listed.json()}
    checks["all_passed"] = all(checks.values())
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", default=None, help="只跑指定 clip_id")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--poll", type=float, default=10.0)
    parser.add_argument("--job-timeout", type=float, default=2400.0)
    args = parser.parse_args()

    clips = load_clips(args.manifest, args.only)
    state = load_state(args.state)
    log(f"待处理 {len(clips)} 个视频")

    with make_client() as client:
        health = client.get(f"{BASE_URL}/api/v1/health")
        health.raise_for_status()
        log(f"后端健康检查通过: {health.json()}")

        for index, clip in enumerate(clips, start=1):
            sample_id = clip["clip_id"]
            entry = state["clips"].get(sample_id, {})
            log(f"({index}/{len(clips)}) {sample_id} view={clip['view']}")

            if entry.get("status") == "published" and (REPORT_DEST / sample_id / "report.html").is_file():
                log("  已发布，跳过")
                continue

            try:
                job_id = entry.get("job_id")
                if job_id:
                    record = client.get(f"/jobs/{job_id}").json()
                    log(f"  复用任务 {job_id}: {record['status']}/{record['stage']}")
                    if record["status"] not in {"succeeded", "failed", "cancelled"}:
                        record = wait_job(client, job_id, args.poll, args.job_timeout)
                else:
                    job_id = upload_job(client, clip)
                    entry = {"clip_id": sample_id, "job_id": job_id, "status": "queued", "updated_at": utc_now()}
                    state["clips"][sample_id] = entry
                    save_state(args.state, state)
                    log(f"  已上传，job_id={job_id}")
                    record = wait_job(client, job_id, args.poll, args.job_timeout)

                if record["status"] != "succeeded":
                    raise RuntimeError(
                        f"任务失败: {record.get('error_code')} {record.get('error_message')}"
                    )

                stats = publish_job(client, job_id, clip)
                checks = verify_sample(client, sample_id)
                entry.update(
                    {
                        "status": "published",
                        "job_id": job_id,
                        "updated_at": utc_now(),
                        "error": None,
                        "stats": stats,
                        "checks": checks,
                    }
                )
                state["clips"][sample_id] = entry
                save_state(args.state, state)
                if not checks["all_passed"]:
                    log(f"  发布完成但校验未全通过: {checks}")
                else:
                    log(
                        f"  发布完成: reps={stats['rep_count']} quality={stats['quality_status']} "
                        f"issues={stats['active_issue_count']} 校验全部通过"
                    )
            except Exception as exc:  # noqa: BLE001 - 单个样本失败不阻断批量
                entry.update({"status": "failed", "updated_at": utc_now(), "error": str(exc)})
                entry.setdefault("clip_id", sample_id)
                state["clips"][sample_id] = entry
                save_state(args.state, state)
                log(f"  失败: {exc}")

    published = [clip for clip in clips if state["clips"].get(clip["clip_id"], {}).get("status") == "published"]
    failures = [clip for clip in clips if state["clips"].get(clip["clip_id"], {}).get("status") != "published"]
    summary = {
        "schema_version": "testdata-web-batch-summary-v1",
        "generated_at": utc_now(),
        "clip_count": len(clips),
        "published_count": len(published),
        "failed_count": len(failures),
        "clips": state["clips"],
    }
    write_json(args.summary, summary)
    log(f"完成：发布 {len(published)}/{len(clips)}，失败 {len(failures)}")
    for clip in failures:
        log(f"  未发布: {clip['clip_id']} -> {state['clips'].get(clip['clip_id'], {}).get('error')}")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
