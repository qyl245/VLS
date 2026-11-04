"""单机任务执行器：SQLite 为事实来源 + 单 worker 线程串行执行。"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

from ..core import config
from ..core.errors import AppError
from ..schemas.jobs import JobRecord, JobStage
from ..services.pipeline import stages
from ..storage import db

_STAGE_ORDER = stages.STAGE_NAMES


class JobManager:
    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._cancel: set[str] = set()
        self._lock = threading.Lock()

    # ---------- 生命周期 ----------

    def on_startup(self) -> None:
        config.ensure_webdata()
        db.init_db()
        requeued, _failed = db.recover_stale()
        for job_id in requeued:
            self._queue.put(job_id)
        self._thread = threading.Thread(target=self._loop, name="vls-job-worker", daemon=True)
        self._thread.start()

    def on_shutdown(self) -> None:
        self._queue.put("")
        if self._thread is not None:
            self._thread.join(timeout=5)

    # ---------- 对外接口 ----------

    def enqueue(self, job_id: str) -> None:
        self._queue.put(job_id)

    def cancel(self, job_id: str) -> bool:
        record = db.get_job(job_id)
        if record is None:
            return False
        if record.status == "queued":
            db.update_job(job_id, status="cancelled", stage="cancelled", finished_at=db.utcnow())
            self._mirror(job_id)
            return True
        if record.status in {"validating", "running"}:
            with self._lock:
                self._cancel.add(job_id)
            return True
        return False

    # ---------- 内部 ----------

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            if not job_id:
                self._queue.task_done()
                return
            try:
                self._run(job_id)
            except Exception:  # noqa: BLE001 - worker 不允许因单任务异常退出
                pass
            finally:
                self._queue.task_done()

    def _run(self, job_id: str) -> None:
        record = db.get_job(job_id)
        if record is None or record.status in {"cancelled", "succeeded"}:
            return
        request = db.get_request(job_id)
        stages_done: list[JobStage] = list(record.stages)
        last_mark = time.perf_counter()

        def on_stage(name: str, detail: dict[str, Any]) -> None:
            nonlocal last_mark
            now = time.perf_counter()
            stages_done.append(
                JobStage(
                    name=name,
                    status="ok",
                    finished_at=db.utcnow(),
                    duration_ms=int((now - last_mark) * 1000),
                )
            )
            last_mark = now
            status = "validating" if name == "validate_upload" else "running"
            index = _STAGE_ORDER.index(name) if name in _STAGE_ORDER else 0
            progress = round(min(max(index / (len(_STAGE_ORDER) - 1), 0.0), 1.0), 3)
            db.update_job(job_id, status=status, stage=name, progress=progress, stages=stages_done)
            self._mirror(job_id)
            with self._lock:
                cancelled = job_id in self._cancel
            if cancelled:
                raise AppError("job_cancelled", "任务已取消", status_code=409)

        db.update_job(job_id, status="validating", stage="validate_upload", started_at=db.utcnow())
        self._mirror(job_id)
        try:
            result = stages.run_stages(job_id, request, on_stage)
            with self._lock:
                self._cancel.discard(job_id)
            db.update_job(
                job_id,
                status="succeeded",
                stage="done",
                progress=1.0,
                finished_at=db.utcnow(),
                error_code=None,
                error_message=None,
                complete=result.get("summary") or {},
            )
        except AppError as exc:
            target = "cancelled" if exc.code == "job_cancelled" else "failed"
            db.update_job(
                job_id,
                status=target,
                stage=exc.code,
                finished_at=db.utcnow(),
                error_code=exc.code,
                error_message=exc.message,
            )
        except Exception as exc:  # noqa: BLE001
            db.update_job(
                job_id,
                status="failed",
                stage="internal_error",
                finished_at=db.utcnow(),
                error_code="internal_error",
                error_message=str(exc),
            )
        finally:
            with self._lock:
                self._cancel.discard(job_id)
            self._mirror(job_id)

    def _mirror(self, job_id: str) -> None:
        """写 status.json 镜像，仅供人工查看，不作为事实来源。"""
        record = db.get_job(job_id)
        if record is None:
            return
        job_dir = config.JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        path = Path(job_dir) / "status.json"
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")


manager = JobManager()
