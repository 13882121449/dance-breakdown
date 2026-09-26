"""服务层共享的文件存储读写工具。

统一时间轴铁律：所有产物（关键点/动画/元数据）都以 ``video_id`` 命名，
帧号 1:1 保序。本模块集中管理落盘路径与 JSON/npy 读写，避免各服务重复实现。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.models.schemas import ActionMetadata, AnimationClip, KeypointSequence

# 关键点文件名后缀（与 pipeline.pipeline.PipelineOrchestrator 落盘约定一致）
KPS_RAW_SUFFIX: str = "_kps.npy"
KPS_SMOOTHED_SUFFIX: str = "_kps_smoothed.npy"
ANIMATION_SUFFIX: str = "_anim.json"


def metadata_file(metadata_dir: Any, video_id: str) -> Path:
    """返回元数据文件路径 ``metadata/{video_id}.json``。"""
    return Path(metadata_dir) / f"{video_id}.json"


def load_metadata(metadata_dir: Any, video_id: str) -> ActionMetadata:
    """读取并反序列化元数据，不存在或非法时抛异常。"""
    path = metadata_file(metadata_dir, video_id)
    if not path.exists():
        raise FileNotFoundError(f"元数据不存在：{path}")
    return ActionMetadata.model_validate_json(path.read_text(encoding="utf-8"))


def save_metadata(metadata_dir: Any, metadata: ActionMetadata) -> Path:
    """把元数据写回 ``metadata/{video_id}.json``。"""
    path = metadata_file(metadata_dir, metadata.video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(metadata.to_json(), encoding="utf-8")
    return path


def animation_file(animations_dir: Any, video_id: str) -> Path:
    """返回动画文件路径 ``animations/{video_id}_anim.json``。"""
    return Path(animations_dir) / f"{video_id}{ANIMATION_SUFFIX}"


def load_animation(animations_dir: Any, video_id: str) -> AnimationClip:
    """读取全量动画帧。"""
    path = animation_file(animations_dir, video_id)
    if not path.exists():
        raise FileNotFoundError(f"动画文件不存在：{path}")
    return AnimationClip.model_validate_json(path.read_text(encoding="utf-8"))


def save_animation(animations_dir: Any, video_id: str, clip: AnimationClip) -> Path:
    """写回全量动画帧。"""
    path = animation_file(animations_dir, video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(clip.model_dump_json(indent=2), encoding="utf-8")
    return path


def keypoints_file(
    keypoints_dir: Any, video_id: str, smoothed: bool = True
) -> Path:
    """返回关键点文件路径（默认平滑后关键点）。"""
    suffix = KPS_SMOOTHED_SUFFIX if smoothed else KPS_RAW_SUFFIX
    return Path(keypoints_dir) / f"{video_id}{suffix}"


def load_keypoints(
    keypoints_dir: Any, video_id: str, smoothed: bool = True
) -> KeypointSequence:
    """读取关键点序列（``[T, 33, 4]``）。"""
    path = keypoints_file(keypoints_dir, video_id, smoothed)
    if not path.exists():
        raise FileNotFoundError(f"关键点文件不存在：{path}")
    return KeypointSequence.load(path, video_id=video_id)


def save_keypoints(
    keypoints_dir: Any, kps: KeypointSequence, smoothed: bool = True
) -> Path:
    """保存关键点序列（npy）。"""
    path = keypoints_file(keypoints_dir, kps.video_id, smoothed)
    path.parent.mkdir(parents=True, exist_ok=True)
    kps.save(path)
    return path


def frame_keypoints_to_list(frame: np.ndarray) -> list[list[float]]:
    """把单帧 ``[33, 4]`` 关键点转为 JSON 友好的嵌套列表。"""
    return [[float(v) for v in row] for row in frame]


def keypoints_to_list(data: np.ndarray) -> list[list[list[float]]]:
    """把全量 ``[T, 33, 4]`` 关键点转为 JSON 友好的嵌套列表（供批量下发）。"""
    return [[[float(v) for v in row] for row in frame] for frame in data]
