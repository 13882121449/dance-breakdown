"""动作注解路由：动作列表读取、单动作读取、注解增改。

注解通过 ``action_id`` 关联动作段；人工修改后 ``annotation_source`` 置为 manual。
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.response import ok
from app.services.annotation_service import AnnotationService, get_annotation_service

router = APIRouter(prefix="/api/video", tags=["actions"])


class AnnotationUpdate(BaseModel):
    """注解更新请求体：只更新提供的字段。"""

    annotations: dict[str, Any] = Field(default_factory=dict)
    name: Optional[str] = None


@router.get("/{video_id}/actions", summary="获取动作列表")
def get_actions(
    video_id: str,
    service: AnnotationService = Depends(get_annotation_service),
) -> dict:
    """返回动作段列表（含 17 维注解）。"""
    try:
        actions = service.get_actions(video_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ok([seg.model_dump() for seg in actions])


@router.get("/{video_id}/actions/{action_id}", summary="获取单个动作")
def get_action(
    video_id: str,
    action_id: str,
    service: AnnotationService = Depends(get_annotation_service),
) -> dict:
    """返回单个动作段。"""
    try:
        seg = service.get_action(video_id, action_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ok(seg.model_dump())


@router.patch("/{video_id}/actions/{action_id}", summary="更新动作注解")
def update_annotation(
    video_id: str,
    action_id: str,
    body: AnnotationUpdate,
    service: AnnotationService = Depends(get_annotation_service),
) -> dict:
    """部分更新动作注解（并可选重命名），返回更新后的动作段。"""
    try:
        seg = service.update_annotation(
            video_id=video_id,
            action_id=action_id,
            annotations=body.annotations,
            name=body.name,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ok(seg.model_dump())
