"""FastAPI 应用入口。

T04 起接入 API 服务层路由（视频/动作/播放/校准），统一挂载到 ``/api`` 前缀，
统一响应格式 ``{code, message, data}``；``code != 0`` 表示错误。
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.api.routes_actions import router as actions_router
from app.api.routes_calibration import router as calibration_router
from app.api.routes_playback import router as playback_router
from app.api.routes_video import router as video_router
from app.config import (
    API_CODE_OK,
    API_MESSAGE_OK,
    CORS_ORIGINS,
    ensure_directories,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """应用生命周期：启动时确保数据目录存在。"""
    ensure_directories()
    yield


app = FastAPI(
    title="舞蹈动作拆解 API",
    description="K-pop 舞蹈视频动作拆解后端服务",
    version="0.1.0",
    lifespan=lifespan,
)

# 允许前端 Vite dev server 跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 压缩响应（全量关键点等大 JSON 载荷）
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ---------------------------------------------------------------------------
# 统一错误响应：把异常转换为 {code, message, data}
# ---------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    """业务错误（4xx/5xx）→ 统一格式，code 取 HTTP 状态码。"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.status_code, "message": str(exc.detail), "data": {}},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    """请求体校验失败（422）→ 统一格式。"""
    return JSONResponse(
        status_code=422,
        content={"code": 422, "message": "请求参数校验失败", "data": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """未捕获异常（500）→ 统一格式，避免泄露内部细节。"""
    return JSONResponse(
        status_code=500,
        content={"code": 500, "message": f"服务器内部错误：{type(exc).__name__}", "data": {}},
    )


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    """健康检查接口。"""
    return {"code": API_CODE_OK, "message": API_MESSAGE_OK, "data": {}}


# ---------------------------------------------------------------------------
# 业务路由（统一挂载 /api 前缀）
# ---------------------------------------------------------------------------
app.include_router(video_router)
app.include_router(actions_router)
app.include_router(playback_router)
app.include_router(calibration_router)
