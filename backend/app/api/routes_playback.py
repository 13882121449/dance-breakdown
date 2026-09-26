"""播放数据路由：下发元数据 + 全量动画帧 + 视频地址 + 单帧关键点。

前端同步播放（architecture.md 6.2）以整数帧号为唯一主键：
- ``GET /api/playback/{video_id}`` 一次性下发元数据与全量动画帧；
- ``GET /api/playback/{video_id}/keypoints/{frame}`` 供校准 UI 按帧读取关键点。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.response import ok
from app.services.video_service import VideoService, get_video_service

router = APIRouter(prefix="/api/playback", tags=["playback"])


@router.get("/{video_id}", summary="获取播放数据（元数据+动画）")
def get_playback(
    video_id: str,
    service: VideoService = Depends(get_video_service),
) -> dict:
    """返回元数据、全量动画帧、视频地址与关键点摘要。"""
    try:
        data = service.get_playback(video_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ok(data)


@router.get("/{video_id}/keypoints", summary="获取全量关键点")
def get_all_keypoints(
    video_id: str,
    service: VideoService = Depends(get_video_service),
) -> dict:
    """返回全量 ``[T, 33, 4]`` 关键点（供前端播放时内存取帧）。"""
    try:
        data = service.get_all_keypoints(video_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ok(data)


@router.get("/{video_id}/keypoints/{frame}", summary="获取单帧关键点")
def get_frame_keypoints(
    video_id: str,
    frame: int,
    service: VideoService = Depends(get_video_service),
) -> dict:
    """返回指定帧的 ``[33, 4]`` 关键点（供校准 UI 修正关键点）。"""
    try:
        data = service.get_frame_keypoints(video_id, frame)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IndexError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ok(data)
