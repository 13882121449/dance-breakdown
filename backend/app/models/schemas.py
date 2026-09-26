"""Pydantic 数据模型：视频元信息、关键点序列、动作段、动作元数据、校准数据等。

本模块是整条离线处理管线（T02/T03）与后续服务层（T04）共用的数据契约，
严格遵循 architecture.md 中的统一时间轴模型与 17 维注解 schema。
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# 17 维注解字段（PRD 11 维基础 + 6 维 K-pop 专属）
# ---------------------------------------------------------------------------
# P0 基础 4 维
P0_BASE_FIELDS: tuple[str, ...] = (
    "force_point",
    "body_posture",
    "rhythm",
    "common_mistakes",
)
# P0 K-pop 专属 4 维
P0_KPOP_FIELDS: tuple[str, ...] = (
    "killing_point",
    "sharpness",
    "power_groove",
    "beat_accent",
)
# P1 其余 9 维
P1_FIELDS: tuple[str, ...] = (
    "weight_transfer",
    "joint_angle",
    "breathing",
    "music_beat",
    "transition",
    "safety",
    "difficulty",
    "facial_expression",
    "wave_isolation",
)
# 全部 17 维，顺序与 architecture.md 2.4 一致
ALL_ANNOTATION_FIELDS: tuple[str, ...] = (
    P0_BASE_FIELDS + P0_KPOP_FIELDS + P1_FIELDS
)

# MediaPipe Pose 关键点数量（33 landmarks）
MEDIAPIPE_NUM_KEYPOINTS: int = 33
# 每帧每关键点 4 个值：x（像素）、y（像素）、z（world 相对深度）、vis（可见度）
KEYPOINT_DIM: int = 4


class Annotations(BaseModel):
    """动作注解（17 维结构化字段）。

    P0 共 8 维：发力点/身体姿态/节奏/常见错误 + killing_point/sharpness/
    power_groove/beat_accent；P1 为其余 9 维。模板自动生成时按需填充。
    """

    model_config = ConfigDict(extra="allow")

    # P0 基础维度
    force_point: str = ""
    body_posture: str = ""
    rhythm: str = ""
    common_mistakes: str = ""
    # P0 K-pop 专属维度
    killing_point: str = ""
    sharpness: str = ""
    power_groove: str = ""
    beat_accent: str = ""
    # P1 维度
    weight_transfer: str = ""
    joint_angle: str = ""
    breathing: str = ""
    music_beat: str = ""
    transition: str = ""
    safety: str = ""
    difficulty: str = ""
    facial_expression: str = ""
    wave_isolation: str = ""


class VideoMeta(BaseModel):
    """视频元信息（帧抽取阶段产出）。"""

    video_id: str
    fps: int = 30
    duration_sec: float = 0.0
    total_frames: int = 0
    status: str = "frames_extracted"


class KeypointSequence(BaseModel):
    """关键点序列。

    数据形状 ``[T, 33, 4]``，索引即帧号（统一时间轴铁律）：
    ``data[t]`` 为第 t 帧的 33 个关键点，每点 ``(x, y, z, vis)``。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    video_id: str
    total_frames: int = 0
    num_keypoints: int = MEDIAPIPE_NUM_KEYPOINTS
    data: np.ndarray = Field(default_factory=lambda: np.empty((0, 0, 0)))

    def get_frame(self, t: int) -> np.ndarray:
        """返回第 t 帧的关键点（shape ``[33, 4]``）。"""
        if t < 0 or t >= self.total_frames:
            raise IndexError(f"帧号 {t} 越界（total_frames={self.total_frames}）")
        return self.data[t]

    def save(self, path: Any) -> None:
        """以 .npy 保存关键点数组。"""
        np.save(str(path), self.data)

    @classmethod
    def load(cls, path: Any, video_id: str = "") -> "KeypointSequence":
        """从 .npy 加载关键点序列。"""
        data = np.load(str(path))
        if data.ndim != 3 or data.shape[1] != MEDIAPIPE_NUM_KEYPOINTS:
            raise ValueError(f"关键点数据形状非法：{data.shape}，期望 [T, 33, 4]")
        return cls(
            video_id=video_id,
            total_frames=int(data.shape[0]),
            num_keypoints=int(data.shape[1]),
            data=data,
        )


class AnimationRef(BaseModel):
    """动作段对应的 3D 动画 clip 引用（按帧范围切片，不复制数据）。"""

    clip_name: str = ""
    start_frame: int = 0
    end_frame: int = 0


class ActionSegment(BaseModel):
    """动作段：通过 start_frame/end_frame 同时指向视频片段与 3D 动画帧。"""

    action_id: str
    name: str = ""
    start_frame: int = 0
    end_frame: int = 0
    start_time_sec: float = 0.0
    end_time_sec: float = 0.0
    annotations: Annotations = Field(default_factory=Annotations)
    annotation_source: str = "template"  # "template"=模板自动 / "manual"=人工
    keypoints_ref: Optional[str] = None
    animation_ref: Optional[AnimationRef] = None


class SegmentAdjustment(BaseModel):
    """人工切分点调整（覆盖对应 action 的 start/end_frame）。"""

    action_id: str
    start_frame: int
    end_frame: int


class KeypointCorrection(BaseModel):
    """关键帧关键点人工修正（触发局部重定向）。"""

    action_id: str
    frame: int
    joint: str
    value: list[float]


class CalibrationData(BaseModel):
    """人工校准数据（写入 metadata.json 的 calibration 字段）。"""

    status: str = "pending"  # "pending"=待校准 / "calibrated"=已校准
    segment_adjustments: list[SegmentAdjustment] = Field(default_factory=list)
    keypoint_corrections: list[KeypointCorrection] = Field(default_factory=list)
    calibrated_at: Optional[str] = None
    calibrated_by: str = "local_user"


class AnimationClip(BaseModel):
    """3D 骨骼动画帧（方向向量法重定向的全量输出）。

    ``joint_rotations`` 为 ``joint -> [[w,x,y,z] x T]``，
    ``root_positions`` 为 ``[[x,y,z] x T]``，三者帧号 1:1。
    """

    clip_name: str = ""
    start_frame: int = 0
    end_frame: int = 0
    joints: list[str] = Field(default_factory=list)
    joint_rotations: dict[str, list[list[float]]] = Field(default_factory=dict)
    root_positions: list[list[float]] = Field(default_factory=list)


class ActionMetadata(BaseModel):
    """动作元数据（管线最终产物，落盘为 metadata.json）。"""

    video_id: str
    fps: int = 30
    duration_sec: float = 0.0
    total_frames: int = 0
    model: dict[str, Any] = Field(default_factory=dict)
    actions: list[ActionSegment] = Field(default_factory=list)
    calibration: CalibrationData = Field(default_factory=CalibrationData)
    keypoints_file: str = ""
    animation_file: str = ""

    def to_json(self) -> str:
        """序列化为 JSON 字符串（写入 metadata.json）。"""
        return self.model_dump_json(indent=2)
