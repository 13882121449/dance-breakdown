"""人工校准路由：提交/读取校准数据并回写元数据。

校准提交（architecture.md 6.3）触发 CalibrationService.apply_calibration：
切分点调整覆盖动作段边界，关键点修正触发受影响动作段局部重定向。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.response import ok
from app.models.schemas import CalibrationData
from app.services.calibration_service import CalibrationService, get_calibration_service

router = APIRouter(prefix="/api/video", tags=["calibration"])


@router.get("/{video_id}/calibration", summary="读取校准数据")
def get_calibration(
    video_id: str,
    service: CalibrationService = Depends(get_calibration_service),
) -> dict:
    """返回当前校准数据（含 segment_adjustments / keypoint_corrections）。"""
    try:
        data = service.get_calibration(video_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ok(data)


@router.post("/{video_id}/calibration", summary="提交人工校准")
def apply_calibration(
    video_id: str,
    calibration: CalibrationData,
    service: CalibrationService = Depends(get_calibration_service),
) -> dict:
    """合并校准数据并回写，返回更新后的完整元数据。"""
    try:
        metadata = service.apply_calibration(video_id, calibration)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ok(metadata.model_dump())
