"""AnnotationService 单元测试：动作读取与注解增改回写。"""

import tempfile
import unittest
from pathlib import Path

from app.models.schemas import ActionMetadata, ActionSegment, Annotations, VideoMeta
from app.services import storage
from app.services.annotation_service import AnnotationService


def _make_metadata(video_id: str = "dance_001") -> ActionMetadata:
    seg1 = ActionSegment(
        action_id="act_001",
        name="动作 1",
        start_frame=0,
        end_frame=30,
        annotations=Annotations(difficulty="初级"),
    )
    seg2 = ActionSegment(
        action_id="act_002",
        name="动作 2",
        start_frame=30,
        end_frame=60,
    )
    return ActionMetadata(
        video_id=video_id,
        fps=30,
        total_frames=60,
        actions=[seg1, seg2],
    )


class TestAnnotationService(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.metadata_dir = Path(self._tmp.name) / "metadata"
        self.service = AnnotationService(metadata_dir=self.metadata_dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _seed(self, video_id: str = "dance_001") -> ActionMetadata:
        metadata = _make_metadata(video_id)
        storage.save_metadata(self.metadata_dir, metadata)
        return metadata

    def test_get_actions(self) -> None:
        self._seed()
        actions = self.service.get_actions("dance_001")
        self.assertEqual([a.action_id for a in actions], ["act_001", "act_002"])

    def test_get_action(self) -> None:
        self._seed()
        seg = self.service.get_action("dance_001", "act_001")
        self.assertEqual(seg.name, "动作 1")

    def test_get_action_missing(self) -> None:
        self._seed()
        with self.assertRaises(FileNotFoundError):
            self.service.get_action("dance_001", "act_999")

    def test_update_annotation(self) -> None:
        self._seed()
        seg = self.service.update_annotation(
            video_id="dance_001",
            action_id="act_001",
            annotations={"difficulty": "中级", "safety": "注意肩部"},
        )
        self.assertEqual(seg.annotations.difficulty, "中级")
        self.assertEqual(seg.annotations.safety, "注意肩部")
        self.assertEqual(seg.annotation_source, "manual")

        # 已持久化
        reloaded = storage.load_metadata(self.metadata_dir, "dance_001")
        self.assertEqual(
            reloaded.actions[0].annotations.difficulty, "中级"
        )
        self.assertEqual(reloaded.actions[0].annotation_source, "manual")

    def test_update_annotation_rename(self) -> None:
        self._seed()
        seg = self.service.update_annotation(
            video_id="dance_001", action_id="act_001", name="右手画圆"
        )
        self.assertEqual(seg.name, "右手画圆")

    def test_update_unknown_field_raises(self) -> None:
        self._seed()
        with self.assertRaises(ValueError):
            self.service.update_annotation(
                video_id="dance_001",
                action_id="act_001",
                annotations={"not_a_field": "x"},
            )


if __name__ == "__main__":
    unittest.main()
