"""FastAPI 入口：REST + 静态产物托管 + 前端 dist。"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.v1 import jobs, reports, samples
from .core import config, errors
from .workers.manager import manager


@asynccontextmanager
async def lifespan(_: FastAPI):
    config.ensure_webdata()
    manager.on_startup()
    try:
        yield
    finally:
        manager.on_shutdown()


app = FastAPI(title="VLS Motion Review", version="0.1.0", lifespan=lifespan)
errors.install(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
def health() -> dict:
    return {"status": "ok", "readonly_roots": [str(p) for p in config.READONLY_ROOTS]}


app.include_router(samples.router, prefix="/api/v1")
app.include_router(jobs.router, prefix="/api/v1")
app.include_router(reports.router, prefix="/api/v1")

# 既有报告：只托管 reports_v1 这一白名单目录（HTML 内相对引用 media/ 与 annotated_frames/）
if config.REPORTS_DIR.is_dir():
    app.mount("/reports", StaticFiles(directory=str(config.REPORTS_DIR), html=True), name="reports")

# 日常部署：FastAPI 托管前端构建产物（SPA，未命中路径回落 index.html）
if config.FRONTEND_DIST.is_dir():
    dist = config.FRONTEND_DIST
    assets_dir = dist / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/", include_in_schema=False)
    def _spa_index() -> FileResponse:
        return FileResponse(dist / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    def _spa_fallback(full_path: str) -> FileResponse:
        if full_path.startswith(("api/", "reports/")):
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = (dist / full_path).resolve()
        try:
            candidate.relative_to(dist.resolve())
        except ValueError:
            raise HTTPException(status_code=404, detail="Not Found") from None
        if candidate.is_file() and candidate.suffix in {".js", ".css", ".png", ".svg", ".ico", ".woff2", ".map"}:
            return FileResponse(candidate)
        return FileResponse(dist / "index.html")
