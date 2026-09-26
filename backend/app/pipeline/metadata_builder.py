"""元数据生成：组装 ActionMetadata 并落盘 metadata.json。

动作段的 17 维注解中，节拍/重音/副歌等「可自动检测」维度由 segmenter 填充，
本模块用规则模板补齐其余维度（仅覆盖空字段，不覆盖自动检测结果），
``annotation_source`` 保持 ``"template"``。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from app.models.schemas import (
    ALL_ANNOTATION_FIELDS,
    ActionMetadata,
    ActionSegment,
    AnimationClip,
    AnimationRef,
    CalibrationData,
    VideoMeta,
)
from app.pipeline.retargeter import MODEL_DEFINITION

# 模板注解：17 维中不可自动检测的维度给出通用规则模板。
# 可自动检测维度（rhythm/killing_point/beat_accent）也提供兜底文案，
# 仅在字段为空时填充（不覆盖 segmenter 的自动检测结果）。
_TEMPLATE_ANNOTATIONS: dict[str, str] = {
    "force_point": "核心稳定，发力集中在躯干与主要动作肢体",
    "body_posture": "躯干保持直立，肩膀放松下沉",
    "rhythm": "按音乐节拍完成，保持律动",
    "common_mistakes": "注意动作幅度与节奏卡点，避免抢拍",
    "weight_transfer": "重心保持在两脚之间，随动作自然转移",
    "joint_angle": "肘/膝关节保持自然弯曲，避免锁死",
    "breathing": "发力呼气，还原吸气",
    "music_beat": "动作落点对齐音乐重拍",
    "transition": "与前后动作自然衔接，不停顿",
    "safety": "动作幅度循序渐进，充分热身避免拉伤",
    "difficulty": "初级",
    "killing_point": "注意动作的传播感与记忆点呈现",
    "sharpness": "动作利落锐利（angular），末拍定格",
    "power_groove": "发力干脆，保持 hip-hop groove 下沉",
    "beat_accent": "动作落点对齐音乐强拍重音",
    "facial_expression": "重拍时眼神专注，表情自然",
    "wave_isolation": "注意 wave 与 isolation 的局部控制",
}


class MetadataBuilder:
    """动作元数据组装器。"""

    def __init__(self, model_def: Optional[dict[str, Any]] = None) -> None:
        self.model_def: dict[str, Any] = dict(model_def or MODEL_DEFINITION)

    def build(
        self,
        video_id: str,
        video_meta: VideoMeta,
        segments: list[ActionSegment],
        clip: Optional[AnimationClip] = None,
        kps_path: str = "",
        anim_path: str = "",
    ) -> ActionMetadata:
        """组装 :class:`ActionMetadata`（补齐模板注解）。"""
        for seg in segments:
            self._fill_template_annotations(seg)
            if seg.animation_ref is None and clip is not None:
                seg.animation_ref = AnimationRef(
                    clip_name=clip.clip_name,
                    start_frame=seg.start_frame,
                    end_frame=seg.end_frame,
                )

        return ActionMetadata(
            video_id=video_id,
            fps=video_meta.fps,
            duration_sec=video_meta.duration_sec,
            total_frames=video_meta.total_frames,
            model=self._model_summary(),
            actions=segments,
            calibration=CalibrationData(),
            keypoints_file=kps_path,
            animation_file=anim_path,
        )

    def save(self, metadata: ActionMetadata, out_path: Any) -> Path:
        """把元数据写入 ``out_path``（metadata.json）。"""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(metadata.to_json(), encoding="utf-8")
        return out_path

    def _fill_template_annotations(self, seg: ActionSegment) -> None:
        """补齐空注解字段（不覆盖 segmenter 的自动检测结果）。"""
        for field in ALL_ANNOTATION_FIELDS:
            current = getattr(seg.annotations, field, "")
            if not current and field in _TEMPLATE_ANNOTATIONS:
                setattr(seg.annotations, field, _TEMPLATE_ANNOTATIONS[field])

    def _model_summary(self) -> dict[str, Any]:
        """元数据中的 model 字段（关节 + 重定向映射）。"""
        return {
            "name": self.model_def.get("name", ""),
            "file": self.model_def.get("file", ""),
            "joints": list(self.model_def.get("joints", [])),
            "retarget_map": dict(self.model_def.get("retarget_map", {})),
        }
