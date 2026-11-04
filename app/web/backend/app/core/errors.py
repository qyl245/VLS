"""统一错误响应：``{"error": {"code", "message"}}``。"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import ROOT, WEBDATA_ROOT


class AppError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400, **extra: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.extra = extra


def not_found(code: str, message: str) -> AppError:
    return AppError(code, message, status_code=404)


def bad_request(code: str, message: str, **extra: Any) -> AppError:
    return AppError(code, message, status_code=400, **extra)


def payload(exc: AppError) -> dict[str, Any]:
    body: dict[str, Any] = {"code": exc.code, "message": exc.message}
    body.update(exc.extra)
    return {"error": body}


def install(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=payload(exc))

    @app.exception_handler(ValueError)
    async def _handle_value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"error": {"code": "invalid_argument", "message": str(exc)}},
        )


__all__ = ["AppError", "bad_request", "install", "not_found", "payload", "ROOT", "WEBDATA_ROOT"]
