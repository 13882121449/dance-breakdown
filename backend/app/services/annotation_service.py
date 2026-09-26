"""注解服务：读取动作列表、人工增删改动作注解并回写元数据。

注解通过 ``action_id`` 关联动作段；人工修改后 ``annotation_source`` 置为
``"manual"``，与模板自动生成（``"template"``）区分。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from app.config import METADATA_DIR
from app.models.schemas import ActionMetadata, ActionSegment
from app.services import storage


class AnnotationService:
    """动作注解读写服务。"""

    def __init__(self, metadata_dir: Any = METADATA_DIR) -> None:
        self.metadata_dir: Path = Path(metadata_dir)

    def get_actions(self, video_id: str) -> list[ActionSegment]:
        """返回指定视频的动作段列表（含 17 维注解）。"""
        metadata = self.get_metadata(video_id)
        return list(metadata.actions)

    def get_action(self, video_id: str, action_id: str) -> ActionSegment:
        """返回单个动作段。"""
        for seg in self.get_actions(video_id):
            if seg.action_id == action_id:
                return seg
        raise FileNotFoundError(f"动作段不存在：{action_id}")

    def get_metadata(self, video_id: str) -> ActionMetadata:
        """读取元数据。"""
        return storage.load_metadata(self.metadata_dir, video_id)

    def update_annotation(
        self,
        video_id: str,
        action_id: str,
        annotations: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> ActionSegment:
        """更新动作注解（部分字段），并把来源标记为 ``"manual"``。

        Args:
            video_id: 视频标识。
            action_id: 动作段标识。
            annotations: 待更新的注解字段（只更新提供的键）。
            name: 可选的动作名称（重命名）。

        Returns:
            更新后的 :class:`ActionSegment`。
        """
        metadata = self.get_metadata(video_id)
        seg = self._find_segment(metadata, action_id)

        if name is not None and name.strip():
            seg.name = name.strip()

        if annotations:
            for key, value in annotations.items():
                if hasattr(seg.annotations, key):
                    setattr(seg.annotations, key, value)
                else:
                    raise ValueError(f"未知注解字段：{key}")

        seg.annotation_source = "manual"
        storage.save_metadata(self.metadata_dir, metadata)
        return seg

    @staticmethod
    def _find_segment(
        metadata: ActionMetadata, action_id: str
    ) -> ActionSegment:
        for seg in metadata.actions:
            if seg.action_id == action_id:
                return seg
        raise FileNotFoundError(f"动作段不存在：{action_id}")


# 进程内单例工厂
_annotation_service: Optional[AnnotationService] = None


def get_annotation_service() -> AnnotationService:
    """返回进程内单例 AnnotationService。"""
    global _annotation_service
    if _annotation_service is None:
        _annotation_service = AnnotationService()
    return _annotation_service
