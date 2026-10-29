"""Web 层 API 回归：样本读取、时间基准、路径防护、Range 流、任务状态机。

所有写操作落在临时 VLS_WEBDATA_ROOT，不触碰真实 webdata/ 与 results/。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web" / "backend"))

_TMP_ROOT = tempfile.mkdtemp(prefix="vls-webdata-")
os.environ["VLS_WEBDATA_ROOT"] = _TMP_ROOT

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.storage import db  # noqa: E402

SAMPLE_ID = "0001_pushup_set01"
FRAME_NAME = "pushup_asymmetry_01_frame_2984.png"
FAILED_VIDEO_BYTES = b"not a real video file"


class WebApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)

    # ---------- 基础 ----------

    def test_01_health(self) -> None:
        payload = self.client.get("/api/v1/health").json()
        self.assertEqual(payload["status"], "ok")

    def test_02_samples_index(self) -> None:
        samples = self.client.get("/api/v1/samples").json()
        ids = [item["sample_id"] for item in samples]
        self.assertIn(SAMPLE_ID, ids)
        self.assertGreaterEqual(len(samples), 13)
        item = next(sample for sample in samples if sample["sample_id"] == SAMPLE_ID)
        for key in ("exercise", "view", "rep_count", "quality_status", "has_media"):
            self.assertIn(key, item)
        self.assertTrue(item["has_media"])

    def test_03_sample_detail_shape(self) -> None:
        detail = self.client.get(f"/api/v1/samples/{SAMPLE_ID}").json()
        for key in ("sample", "video", "timeline", "issues", "quality", "contract", "confidence", "report_url"):
            self.assertIn(key, detail)
        timebase = detail["video"]["timebase"]
        self.assertEqual(timebase["video_time_origin"], "clip_start")
        self.assertEqual(timebase["fps"], 30.0)
        self.assertGreater(timebase["duration_seconds"], 0)
        self.assertGreater(len(detail["timeline"]["reps"]), 0)

    def test_04_timebase_conversion_matches_source_frames(self) -> None:
        """下发的问题锚点时间必须等于 源时间 - trim.start_frame/fps。"""
        analysis = json.loads((ROOT / "results" / "clip_analysis_v1" / SAMPLE_ID / "analysis.json").read_text(encoding="utf-8"))
        issues_raw = json.loads((ROOT / "results" / "feedback_v1" / SAMPLE_ID / "issues.json").read_text(encoding="utf-8"))
        fps = float(analysis["fps"])
        start_frame = int(analysis["trim"]["start_frame"])
        detail = self.client.get(f"/api/v1/samples/{SAMPLE_ID}").json()

        expected = {
            item["issue_id"]: round(float(item["anchor_time_s"]) - start_frame / fps, 6)
            for item in issues_raw["issues"]
            if item.get("anchor_time_s") is not None
        }
        for issue in detail["issues"]:
            if issue["anchor_s"] is None:
                continue
            self.assertAlmostEqual(issue["anchor_s"], expected[issue["issue_id"]], places=5)

        duration = detail["video"]["timebase"]["duration_seconds"]
        for rep in detail["timeline"]["reps"]:
            self.assertGreaterEqual(rep["start_s"], -0.001)
            self.assertLessEqual(rep["end_s"], duration + 0.001)
            self.assertLessEqual(rep["start_s"], rep["bottom_s"])
        for interval in detail["timeline"]["evidence_intervals"]:
            self.assertLessEqual(interval["start_s"], interval["end_s"])

    def test_05_evidence_frame_url_points_to_real_file(self) -> None:
        detail = self.client.get(f"/api/v1/samples/{SAMPLE_ID}").json()
        issues_with_frame = [issue for issue in detail["issues"] if issue["frame_url"]]
        self.assertGreater(len(issues_with_frame), 0)
        response = self.client.get(issues_with_frame[0]["frame_url"])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        for issue in detail["issues"]:
            if issue["suppressed"]:
                self.assertIsNone(issue["frame_url"])

    # ---------- 路径防护 ----------

    def test_06_unknown_sample_returns_404(self) -> None:
        response = self.client.get("/api/v1/samples/does_not_exist")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "sample_not_found")

    def test_07_path_traversal_is_rejected(self) -> None:
        for bad in ("../../etc", "..%2F..%2Fetc", "0001_pushup_set01%2F..%2F..", "%2e%2e%2fwebdata"):
            response = self.client.get(f"/api/v1/samples/{bad}")
            self.assertEqual(response.status_code, 404, msg=bad)

    def test_08_frame_name_traversal_is_rejected(self) -> None:
        for bad in ("..%2Freport.html", "..%2F..%2Fanalysis.json", "report.html", "%2e%2e%2f%2e%2e%2fapp.db"):
            response = self.client.get(f"/api/v1/samples/{SAMPLE_ID}/frames/{bad}")
            self.assertEqual(response.status_code, 404, msg=bad)

    def test_09_no_absolute_paths_in_payload(self) -> None:
        body = self.client.get(f"/api/v1/samples/{SAMPLE_ID}").text
        self.assertNotIn("D:\\", body)
        self.assertNotIn("/mnt/storage", body)
        self.assertNotIn("results/", body)

    # ---------- 媒体 ----------

    def test_10_media_supports_range(self) -> None:
        response = self.client.get(f"/api/v1/samples/{SAMPLE_ID}/media", headers={"Range": "bytes=0-1023"})
        self.assertEqual(response.status_code, 206)
        self.assertEqual(len(response.content), 1024)
        self.assertIn("bytes 0-1023/", response.headers["content-range"])
        self.assertEqual(response.headers["content-type"], "video/webm")

    def test_11_media_full_request_is_200(self) -> None:
        response = self.client.get(f"/api/v1/samples/{SAMPLE_ID}/media")
        self.assertEqual(response.status_code, 200)
        self.assertGreater(int(response.headers["content-length"]), 1024)

    def test_12_reports_index(self) -> None:
        payload = self.client.get("/api/v1/reports").json()
        ids = [item["report_id"] for item in payload["reports"]]
        self.assertIn(SAMPLE_ID, ids)
        for item in payload["reports"]:
            self.assertTrue(item["url"].startswith("/reports/"))
            self.assertNotIn("D:\\", item["url"])

    def test_13_report_html_is_served(self) -> None:
        response = self.client.get(f"/reports/{SAMPLE_ID}/report.html")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])

    def test_14_no_disk_traversal_through_static_mount(self) -> None:
        # 编码后的穿越路径必须被静态目录拒绝
        for path in ("/reports/..%2F..%2Fapp.db", "/reports/%2e%2e%2f%2e%2e%2fwebdata%2fapp.db"):
            response = self.client.get(path)
            self.assertIn(response.status_code, (404, 400), msg=path)
        # SPA 兜底可以返回 index.html，但绝不能泄漏真实文件内容
        for path in ("/webdata/app.db", "/../app.db"):
            response = self.client.get(path)
            self.assertNotIn(b"SQLite format 3", response.content, msg=path)
            self.assertIn("text/html", response.headers["content-type"], msg=path)

    # ---------- 任务 ----------

    def test_15_create_job_requires_sample_source(self) -> None:
        response = self.client.post("/api/v1/jobs", json={"source_kind": "upload", "sample_id": None})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "invalid_source")

    def test_16_upload_rejects_unsupported_suffix(self) -> None:
        response = self.client.post(
            "/api/v1/jobs/upload",
            files={"file": ("clip.txt", b"hello", "text/plain")},
            data={"exercise": "pushup", "view": "front", "rotation": "none"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "unsupported_file_type")

    def test_17_upload_rejects_invalid_exercise(self) -> None:
        response = self.client.post(
            "/api/v1/jobs/upload",
            files={"file": ("clip.mp4", FAILED_VIDEO_BYTES, "video/mp4")},
            data={"exercise": "burpee", "view": "front", "rotation": "none"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "invalid_exercise")

    def test_18_upload_job_reaches_failed_state(self) -> None:
        response = self.client.post(
            "/api/v1/jobs/upload",
            files={"file": ("broken.mp4", FAILED_VIDEO_BYTES, "video/mp4")},
            data={"exercise": "pushup", "view": "front", "rotation": "none"},
        )
        self.assertEqual(response.status_code, 202)
        job_id = response.json()["job_id"]
        record = self._wait_for_terminal_state(job_id)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error_code"], "invalid_video")
        self.assertIsNotNone(record["finished_at"])

    def test_19_unknown_job_returns_404(self) -> None:
        response = self.client.get("/api/v1/jobs/deadbeef0000")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "job_not_found")

    def test_20_result_of_unfinished_job_is_409(self) -> None:
        job_id = uuid.uuid4().hex[:12]
        record = self._make_record(job_id, status="queued")
        db.insert_job(record, {})
        response = self.client.get(f"/api/v1/jobs/{job_id}/result")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "job_not_ready")

    def test_21_cancel_queued_job(self) -> None:
        job_id = uuid.uuid4().hex[:12]
        record = self._make_record(job_id, status="queued")
        db.insert_job(record, {})
        response = self.client.post(f"/api/v1/jobs/{job_id}/cancel")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "cancelled")

    def test_22_cancel_finished_job_is_rejected(self) -> None:
        job_id = uuid.uuid4().hex[:12]
        record = self._make_record(job_id, status="succeeded")
        db.insert_job(record, {})
        response = self.client.post(f"/api/v1/jobs/{job_id}/cancel")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "job_cancel_rejected")

    def test_23_stale_running_job_is_recovered_as_failed(self) -> None:
        job_id = uuid.uuid4().hex[:12]
        record = self._make_record(job_id, status="running")
        db.insert_job(record, {})
        requeued, failed = db.recover_stale()
        self.assertIn(job_id, failed)
        after = db.get_job(job_id)
        self.assertIsNotNone(after)
        self.assertEqual(after.status, "failed")
        self.assertEqual(after.error_code, "stale_after_restart")

    def test_24_job_dedup_for_same_sample(self) -> None:
        first = self.client.post("/api/v1/jobs", json={"source_kind": "sample", "sample_id": SAMPLE_ID})
        if first.status_code != 202:
            self.skipTest("源视频不可用，跳过去重用例")
        job_id = first.json()["job_id"]
        second = self.client.post("/api/v1/jobs", json={"source_kind": "sample", "sample_id": SAMPLE_ID})
        self.assertEqual(second.status_code, 202)
        active_status = db.get_job(job_id)
        if active_status is not None and active_status.status in {"queued", "validating", "running"}:
            self.assertEqual(second.json()["job_id"], job_id)
        else:
            self.assertNotEqual(second.json()["job_id"], job_id)

    # ---------- 辅助 ----------

    def _wait_for_terminal_state(self, job_id: str, timeout_s: float = 20.0) -> dict:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            record = self.client.get(f"/api/v1/jobs/{job_id}").json()
            if record["status"] in {"succeeded", "failed", "cancelled"}:
                return record
            time.sleep(0.5)
        self.fail(f"job {job_id} 未在 {timeout_s}s 内结束")

    @staticmethod
    def _make_record(job_id: str, status: str):
        from app.schemas.jobs import JobRecord

        return JobRecord(
            job_id=job_id,
            source_kind="upload",
            sample_id=None,
            status=status,  # type: ignore[arg-type]
            stage=status,
            progress=0.0,
            created_at=db.utcnow(),
        )


if __name__ == "__main__":
    unittest.main()
