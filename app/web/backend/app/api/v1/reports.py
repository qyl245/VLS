"""报告浏览：只暴露 reports_v1 白名单目录下的既有 HTML 报告。"""

from __future__ import annotations

from fastapi import APIRouter

from ...core import config
from ...core.errors import not_found
from ...storage import artifacts

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("")
def list_reports() -> dict:
    items = []
    root = config.REPORTS_DIR
    if root.is_dir():
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            html = entry / "report.html"
            if not html.is_file():
                continue
            media = entry / "media" / f"{entry.name}_clip.webm"
            items.append(
                {
                    "report_id": entry.name,
                    "url": f"/reports/{entry.name}/report.html",
                    "has_media": media.is_file(),
                    "size_bytes": html.stat().st_size,
                }
            )
    return {"reports": items}


@router.get("/{report_id}")
def get_report(report_id: str) -> dict:
    report_dir = artifacts.sample_report_dir(report_id)
    if report_dir is None or not (report_dir / "report.html").is_file():
        raise not_found("report_not_found", f"report not found: {report_id}")
    return {
        "report_id": report_id,
        "url": f"/reports/{report_id}/report.html",
        "title": f"{report_id} 分析报告",
    }
