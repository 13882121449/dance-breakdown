"""视频路由：上传、启动处理、状态/进度查询、元数据读取、视频文件下发。

统一响应格式 ``{code, message, data}``；错误通过 HTTPException 抛出，
由 main.py 注册的异常处理器统一转换。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.api.response import ok
from app.services.video_service import VideoService, get_video_service

router = APIRouter(prefix="/api/video", tags=["video"])

# 视频扩展名 → MIME 类型（用于 FileResponse 正确下发）
_MEDIA_TYPES: dict[str, str] = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".m4v": "video/x-m4v",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
}


@router.post("/upload", summary="上传视频")
async def upload_video(
    file: UploadFile = File(...),
    service: VideoService = Depends(get_video_service),
) -> dict:
    """上传视频文件，返回 ``video_id``（后续用它启动处理）。"""
    try:
        content = await file.read()
        data = service.upload(file.filename or "video.mp4", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ok(data)


@router.post("/{video_id}/process", summary="启动离线处理管线")
def start_process(
    video_id: str,
    service: VideoService = Depends(get_video_service),
) -> dict:
    """后台启动处理管线，立即返回处理中状态（前端轮询 status/progress）。"""
    try:
        data = service.start_process(video_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ok(data)


@router.get("/{video_id}/status", summary="查询处理状态")
def get_status(
    video_id: str,
    service: VideoService = Depends(get_video_service),
) -> dict:
    """返回 ``{video_id, status, progress, error}``。"""
    try:
        data = service.get_status(video_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ok(data)


@router.get("/{video_id}/progress", summary="查询处理进度")
def get_progress(
    video_id: str,
    service: VideoService = Depends(get_video_service),
) -> dict:
    """返回 ``{stage, percent, message}``。"""
    try:
        data = service.get_progress(video_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ok(data)


@router.get("/{video_id}/metadata", summary="读取动作元数据")
def get_metadata(
    video_id: str,
    service: VideoService = Depends(get_video_service),
) -> dict:
    """返回完整 :class:`ActionMetadata`。"""
    try:
        metadata = service.get_metadata(video_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ok(metadata.model_dump())


@router.get("/{video_id}/file", summary="下发视频文件")
def get_video_file(
    video_id: str,
    service: VideoService = Depends(get_video_service),
) -> FileResponse:
    """返回原始视频文件（供前端 <video> 播放，帧号时间轴对齐基准）。"""
    try:
        path = service.get_video_path(video_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    media_type = _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media_type, filename=path.name)
