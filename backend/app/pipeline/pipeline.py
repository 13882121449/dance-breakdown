"""管线编排器：串起 帧抽取 → 人像分割 → 姿态估计 → 平滑 → (切分 ∥ 重定向) → 元数据。

统一时间轴铁律：整条管线以整数帧号为唯一主键，各阶段产物帧号 1:1 保序。
长任务通过 :meth:`get_progress` 暴露阶段进度（供 T04 轮询）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from app.config import (
    ANIMATIONS_DIR,
    FRAMES_DIR,
    FRAMES_MASKED_DIR,
    KEYPOINTS_DIR,
    METADATA_DIR,
    TARGET_FPS,
)
from app.models.schemas import ActionMetadata, KeypointSequence, VideoMeta
from app.pipeline.frame_extractor import FrameExtractor
from app.pipeline.metadata_builder import MetadataBuilder
from app.pipeline.person_segmenter import PersonSegmenter
from app.pipeline.pose_estimator import PoseEstimator
from app.pipeline.retargeter import Retargeter
from app.pipeline.segmenter import ActionSegmenter
from app.pipeline.smoother import SequenceSmoother

ProgressCallback = Callable[[str, str, int, str], None]


class PipelineOrchestrator:
    """离线处理管线编排器。"""

    # 管线阶段顺序（stage, 进度百分比, 描述）
    _STAGES: tuple[tuple[str, int, str], ...] = (
        ("extract", 5, "抽取视频帧"),
        ("segment_person", 20, "人像分割去背景"),
        ("estimate", 40, "姿态估计"),
        ("smooth", 55, "时序平滑"),
        ("segment_actions", 70, "动作切分"),
        ("retarget", 85, "骨骼重定向"),
        ("metadata", 95, "生成元数据"),
        ("done", 100, "完成"),
    )

    def __init__(
        self,
        target_fps: int = TARGET_FPS,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        self.target_fps: int = target_fps
        self._callback: Optional[ProgressCallback] = progress_callback
        self._progress: dict[str, dict[str, Any]] = {}

    def process(
        self,
        video_path: Any,
        video_id: Optional[str] = None,
        model_def: Optional[dict[str, Any]] = None,
    ) -> ActionMetadata:
        """执行完整离线管线，返回 :class:`ActionMetadata`。

        Raises:
            ImportError: mediapipe 未安装时在分割/姿态阶段抛出。
            FileNotFoundError: mediapipe 模型文件缺失时抛出（见 README 下载说明）。
        """
        video_path = Path(video_path)
        video_id = video_id or video_path.stem

        # 1) 帧抽取
        self._set_progress(video_id, "extract", 5, "抽取视频帧")
        frames_dir = FRAMES_DIR / video_id
        video_meta = FrameExtractor(self.target_fps).extract(
            video_path, frames_dir, video_id, self.target_fps
        )

        # 2) 人像分割
        self._set_progress(video_id, "segment_person", 20, "人像分割去背景")
        masked_dir = FRAMES_MASKED_DIR / video_id
        person_segmenter = PersonSegmenter()
        try:
            person_segmenter.segment(frames_dir, masked_dir)
        finally:
            person_segmenter.close()

        # 3) 姿态估计
        self._set_progress(video_id, "estimate", 40, "姿态估计")
        pose_estimator = PoseEstimator()
        try:
            kps = pose_estimator.estimate(masked_dir, video_meta)
        finally:
            pose_estimator.close()

        # 4) 时序平滑
        self._set_progress(video_id, "smooth", 55, "时序平滑")
        kps_smoothed = SequenceSmoother().smooth(kps)

        # 5) 动作切分（节拍为主，能量峰兜底）
        self._set_progress(video_id, "segment_actions", 70, "动作切分")
        action_segmenter = ActionSegmenter()
        beats: list[int] = []
        onsets: list[int] = []
        try:
            beats = action_segmenter.detect_beats(video_path, video_meta.fps)
            onsets = action_segmenter.detect_onsets(video_path, video_meta.fps)
        except Exception:
            # 音频提取失败（无 ffmpeg/audioread）时退化为纯能量峰切分
            beats = []
            onsets = []
        segments = action_segmenter.segment(
            kps_smoothed, video_meta.fps, beats, onsets
        )

        # 6) 骨骼重定向
        self._set_progress(video_id, "retarget", 85, "骨骼重定向")
        clip = Retargeter(model_def).retarget(kps_smoothed)

        # 7) 落盘关键点/动画/元数据
        self._set_progress(video_id, "metadata", 95, "生成元数据")
        self._persist(video_id, kps, kps_smoothed, clip)

        metadata = MetadataBuilder(model_def).build(
            video_id=video_id,
            video_meta=video_meta,
            segments=segments,
            clip=clip,
            kps_path=f"keypoints/{video_id}_kps_smoothed.npy",
            anim_path=f"animations/{video_id}_anim.json",
        )
        MetadataBuilder(model_def).save(metadata, METADATA_DIR / f"{video_id}.json")

        self._set_progress(video_id, "done", 100, "完成")
        return metadata

    def get_progress(self, video_id: str) -> dict[str, Any]:
        """返回指定视频的处理进度。"""
        return self._progress.get(
            video_id, {"stage": "idle", "percent": 0, "message": "未开始"}
        )

    # ------------------------------------------------------------------
    def _persist(
        self,
        video_id: str,
        kps_raw: KeypointSequence,
        kps_smoothed: KeypointSequence,
        clip: Any,
    ) -> None:
        KEYPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        ANIMATIONS_DIR.mkdir(parents=True, exist_ok=True)
        METADATA_DIR.mkdir(parents=True, exist_ok=True)

        kps_raw.save(KEYPOINTS_DIR / f"{video_id}_kps.npy")
        kps_smoothed.save(KEYPOINTS_DIR / f"{video_id}_kps_smoothed.npy")
        (ANIMATIONS_DIR / f"{video_id}_anim.json").write_text(
            clip.model_dump_json(indent=2), encoding="utf-8"
        )

    def _set_progress(self, video_id: str, stage: str, percent: int, message: str) -> None:
        self._progress[video_id] = {
            "stage": stage,
            "percent": percent,
            "message": message,
        }
        if self._callback is not None:
            self._callback(video_id, stage, percent, message)
