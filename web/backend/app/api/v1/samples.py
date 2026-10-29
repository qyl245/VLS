"""样本库 API：列表、详情、视频 Range 流、证据帧。"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

from ...schemas.samples import SampleDetail, SampleSummary
from ...services import registry
from ...storage import artifacts

router = APIRouter(prefix="/samples", tags=["samples"])


@router.get("", response_model=list[SampleSummary])
def list_samples() -> list[SampleSummary]:
    return registry.list_summaries()


@router.get("/{sample_id}", response_model=SampleDetail)
def get_sample(sample_id: str) -> SampleDetail:
    return registry.detail(sample_id)


@router.get("/{sample_id}/media")
def get_sample_media(sample_id: str) -> FileResponse:
    path = artifacts.sample_media_path(sample_id)
    return FileResponse(path, media_type="video/webm", filename=path.name)


@router.get("/{sample_id}/frames/{frame_name}")
def get_sample_frame(sample_id: str, frame_name: str) -> FileResponse:
    path = artifacts.sample_frame_path(sample_id, frame_name)
    return FileResponse(path, media_type="image/png", filename=path.name)


@router.get("/{sample_id}/artifacts")
def list_sample_artifacts(sample_id: str) -> dict:
    """产物清单：只给文件名、类型和大小，不暴露服务器路径。"""
    analysis_dir = artifacts.sample_analysis_dir(sample_id)
    report_dir = artifacts.sample_report_dir(sample_id)
    items = []
    for name in sorted(p.name for p in analysis_dir.iterdir() if p.is_file()):
        items.append({"name": name, "kind": "analysis", "size_bytes": (analysis_dir / name).stat().st_size})
    if report_dir is not None:
        for sub in ("media", "annotated_frames"):
            directory = report_dir / sub
            if not directory.is_dir():
                continue
            for path in sorted(directory.iterdir()):
                if path.is_file():
                    items.append({"name": f"{sub}/{path.name}", "kind": sub, "size_bytes": path.stat().st_size})
    return {"sample_id": sample_id, "artifacts": items}
