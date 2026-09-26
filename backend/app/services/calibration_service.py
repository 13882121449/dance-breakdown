"""人工校准回写服务：合并切分点调整与关键帧关键点修正，局部重定向并回写。

校准流程（architecture.md 6.3）：
1. 读 metadata.json；
2. ``segment_adjustments`` 覆盖对应 action 的 start/end_frame；
3. ``keypoint_corrections`` 更新关键点后，仅对受影响动作段做局部重定向；
4. 回写 metadata.json（status=calibrated）。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.config import ANIMATIONS_DIR, KEYPOINTS_DIR, METADATA_DIR
from app.models.schemas import (
    ActionMetadata,
    AnimationClip,
    CalibrationData,
    KeypointSequence,
    SegmentAdjustment,
)
from app.pipeline.retargeter import MODEL_DEFINITION, Retargeter
from app.services import storage

# 直接映射到单一 landmark 的模型关节 → MediaPipe landmark 索引
_DIRECT_JOINT_TO_LANDMARK: dict[str, int] = {
    joint: int(idx) for idx, joint in MODEL_DEFINITION["retarget_map"].items()
}


class CalibrationService:
    """人工校准回写服务。"""

    def __init__(
        self,
        metadata_dir: Any = METADATA_DIR,
        keypoints_dir: Any = KEYPOINTS_DIR,
        animations_dir: Any = ANIMATIONS_DIR,
    ) -> None:
        self.metadata_dir: Path = Path(metadata_dir)
        self.keypoints_dir: Path = Path(keypoints_dir)
        self.animations_dir: Path = Path(animations_dir)

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------
    def get_calibration(self, video_id: str) -> dict[str, Any]:
        """返回当前校准数据。"""
        metadata = storage.load_metadata(self.metadata_dir, video_id)
        return metadata.calibration.model_dump()

    def apply_calibration(
        self, video_id: str, calibration: CalibrationData
    ) -> ActionMetadata:
        """合并校准数据并回写元数据，返回更新后的 :class:`ActionMetadata`。"""
        metadata = storage.load_metadata(self.metadata_dir, video_id)

        # 1) 切分点调整：覆盖动作段边界（动画/关键点不动，仅改引用范围）
        for adj in calibration.segment_adjustments:
            self._apply_segment_adjustment(metadata, adj)

        # 2) 关键点修正：更新关键点 + 受影响动作段局部重定向
        if calibration.keypoint_corrections:
            kps = storage.load_keypoints(
                self.keypoints_dir, video_id, smoothed=True
            )
            clip = self._load_or_retarget_full(video_id, kps)
            affected_ranges = self._apply_keypoint_corrections(
                metadata, kps, calibration.keypoint_corrections
            )

            for start, end in affected_ranges:
                self._local_retarget(kps, clip, start, end)

            storage.save_keypoints(self.keypoints_dir, kps, smoothed=True)
            storage.save_animation(self.animations_dir, video_id, clip)

        # 3) 回写校准元数据
        calibration.status = "calibrated"
        calibration.calibrated_at = datetime.now(timezone.utc).isoformat()
        metadata.calibration = calibration
        storage.save_metadata(self.metadata_dir, metadata)
        return metadata

    # ------------------------------------------------------------------
    # 切分点调整
    # ------------------------------------------------------------------
    def _apply_segment_adjustment(
        self, metadata: ActionMetadata, adj: SegmentAdjustment
    ) -> None:
        seg = self._find_segment(metadata, adj.action_id)
        if adj.start_frame < 0 or adj.end_frame <= adj.start_frame:
            raise ValueError(
                f"非法切分点：start_frame={adj.start_frame}, end_frame={adj.end_frame}"
            )
        fps = metadata.fps or 30
        seg.start_frame = adj.start_frame
        seg.end_frame = adj.end_frame
        seg.start_time_sec = round(adj.start_frame / fps, 3)
        seg.end_time_sec = round(adj.end_frame / fps, 3)
        if seg.animation_ref is not None:
            seg.animation_ref.start_frame = adj.start_frame
            seg.animation_ref.end_frame = adj.end_frame

    # ------------------------------------------------------------------
    # 关键点修正
    # ------------------------------------------------------------------
    def _apply_keypoint_corrections(
        self,
        metadata: ActionMetadata,
        kps: KeypointSequence,
        corrections: list[Any],
    ) -> list[tuple[int, int]]:
        """把关键点修正写入关键点数组，返回受影响动作段的帧范围列表。"""
        affected: list[tuple[int, int]] = []
        for corr in corrections:
            seg = self._find_segment(metadata, corr.action_id)
            landmark = self._resolve_landmark(corr.joint)

            frame = int(corr.frame)
            if frame < 0 or frame >= kps.total_frames:
                raise ValueError(
                    f"关键点修正帧号越界：{frame}（total_frames={kps.total_frames}）"
                )

            value = list(corr.value)
            if len(value) < 2:
                raise ValueError(f"关键点修正值至少需 [x, y]，收到 {value}")
            kps.data[frame, landmark, 0] = float(value[0])
            kps.data[frame, landmark, 1] = float(value[1])
            if len(value) >= 3:
                kps.data[frame, landmark, 2] = float(value[2])

            affected.append((seg.start_frame, seg.end_frame))
        return affected

    def _local_retarget(
        self, kps: KeypointSequence, clip: AnimationClip, start: int, end: int
    ) -> None:
        """对 ``[start, end]``（含端点）做局部重定向，并拼回全量动画。"""
        start = max(0, int(start))
        end = min(kps.total_frames - 1, int(end))
        if start > end:
            return

        subset = KeypointSequence(
            video_id=kps.video_id,
            total_frames=end - start + 1,
            num_keypoints=kps.num_keypoints,
            data=kps.data[start : end + 1],
        )
        sub_clip = Retargeter(MODEL_DEFINITION).retarget(subset)

        for joint, rotations in clip.joint_rotations.items():
            if joint not in sub_clip.joint_rotations:
                continue
            for local, g in enumerate(range(start, end + 1)):
                if g < len(rotations):
                    rotations[g] = sub_clip.joint_rotations[joint][local]

        for local, g in enumerate(range(start, end + 1)):
            if g < len(clip.root_positions):
                clip.root_positions[g] = sub_clip.root_positions[local]

    def _load_or_retarget_full(
        self, video_id: str, kps: KeypointSequence
    ) -> AnimationClip:
        """读取全量动画；缺失时由关键点全量重定向生成。"""
        try:
            return storage.load_animation(self.animations_dir, video_id)
        except FileNotFoundError:
            clip = Retargeter(MODEL_DEFINITION).retarget(kps)
            storage.save_animation(self.animations_dir, video_id, clip)
            return clip

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _find_segment(metadata: ActionMetadata, action_id: str) -> Any:
        for seg in metadata.actions:
            if seg.action_id == action_id:
                return seg
        raise FileNotFoundError(f"动作段不存在：{action_id}")

    @staticmethod
    def _resolve_landmark(joint: str) -> int:
        if joint not in _DIRECT_JOINT_TO_LANDMARK:
            raise ValueError(
                f"关节 {joint} 无单一 landmark 映射（派生关节 Hips/Spine/Chest/Neck "
                f"不支持单帧直接修正），可修正关节：{sorted(_DIRECT_JOINT_TO_LANDMARK)}"
            )
        return _DIRECT_JOINT_TO_LANDMARK[joint]


# 进程内单例工厂
_calibration_service: Optional[CalibrationService] = None


def get_calibration_service() -> CalibrationService:
    """返回进程内单例 CalibrationService。"""
    global _calibration_service
    if _calibration_service is None:
        _calibration_service = CalibrationService()
    return _calibration_service
