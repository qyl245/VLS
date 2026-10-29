"""SQLite 任务表：状态事实来源（status.json 仅作人工镜像）。"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from ..core import config
from ..schemas.jobs import JobRecord, JobStage

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  source_kind TEXT NOT NULL,
  sample_id TEXT,
  status TEXT NOT NULL,
  stage TEXT NOT NULL,
  progress REAL NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT,
  error_code TEXT,
  error_message TEXT,
  pipeline_version TEXT NOT NULL DEFAULT 'v0',
  stages_json TEXT NOT NULL DEFAULT '[]',
  request_json TEXT NOT NULL DEFAULT '{}',
  complete_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    config.ensure_webdata()
    conn = sqlite3.connect(config.DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def _row_to_record(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        job_id=row["job_id"],
        source_kind=row["source_kind"],
        sample_id=row["sample_id"],
        status=row["status"],
        stage=row["stage"],
        progress=row["progress"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        error_code=row["error_code"],
        error_message=row["error_message"],
        pipeline_version=row["pipeline_version"],
        stages=[JobStage(**item) for item in json.loads(row["stages_json"] or "[]")],
    )


def insert_job(record: JobRecord, request: dict[str, Any]) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, source_kind, sample_id, status, stage, progress, created_at,"
            " started_at, finished_at, error_code, error_message, pipeline_version, stages_json, request_json)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                record.job_id,
                record.source_kind,
                record.sample_id,
                record.status,
                record.stage,
                record.progress,
                record.created_at,
                record.started_at,
                record.finished_at,
                record.error_code,
                record.error_message,
                record.pipeline_version,
                json.dumps([s.model_dump() for s in record.stages], ensure_ascii=False),
                json.dumps(request, ensure_ascii=False),
            ),
        )


def get_job(job_id: str) -> JobRecord | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return _row_to_record(row) if row else None


def get_request(job_id: str) -> dict[str, Any]:
    with connect() as conn:
        row = conn.execute("SELECT request_json FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return json.loads(row["request_json"]) if row else {}


def list_jobs(limit: int = 50) -> list[JobRecord]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [_row_to_record(row) for row in rows]


def find_active_for_sample(sample_id: str) -> JobRecord | None:
    """同一冻结样本存在未完成任务时复用它，避免重复提交。"""
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE sample_id = ? AND status IN ('queued','validating','running')"
            " ORDER BY created_at DESC LIMIT 1",
            (sample_id,),
        ).fetchone()
    return _row_to_record(row) if row else None


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    if "stages" in fields:
        fields["stages_json"] = json.dumps([s if isinstance(s, dict) else s.model_dump() for s in fields.pop("stages")], ensure_ascii=False)
    if "complete" in fields:
        fields["complete_json"] = json.dumps(fields.pop("complete"), ensure_ascii=False)
    columns = ", ".join(f"{key} = ?" for key in fields)
    with connect() as conn:
        conn.execute(f"UPDATE jobs SET {columns} WHERE job_id = ?", (*fields.values(), job_id))


def claim_next_queued() -> JobRecord | None:
    """原子地把最早的 queued 任务置为 validating。"""
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        conn.execute(
            "UPDATE jobs SET status = 'validating', stage = 'validate_upload', started_at = ? WHERE job_id = ?",
            (utcnow(), row["job_id"]),
        )
        conn.execute("COMMIT")
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)).fetchone()
    return _row_to_record(row)


def recover_stale() -> tuple[list[str], list[str]]:
    """启动时恢复：queued 重新排队，running/validating 标记为失败（子进程已不可恢复）。"""
    requeued: list[str] = []
    failed: list[str] = []
    with connect() as conn:
        rows = conn.execute("SELECT job_id, status FROM jobs WHERE status IN ('queued','running','validating')").fetchall()
        for row in rows:
            if row["status"] == "queued":
                requeued.append(row["job_id"])
            else:
                failed.append(row["job_id"])
                conn.execute(
                    "UPDATE jobs SET status = 'failed', stage = 'interrupted', finished_at = ?,"
                    " error_code = 'stale_after_restart', error_message = '服务重启，执行中的任务已中断'"
                    " WHERE job_id = ?",
                    (utcnow(), row["job_id"]),
                )
    return requeued, failed
