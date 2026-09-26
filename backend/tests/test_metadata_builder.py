"""MetadataBuilder 单元测试（17 维注解补齐 + JSON 往返）。"""

import json
import tempfile
import unittest
from pathlib import Path

from app.models.schemas import (
    ALL_ANNOTATION_FIELDS,
    ActionMetadata,
    ActionSegment,
    AnimationClip,
    VideoMeta,
)
from app.pipeline.metadata_builder import MetadataBuilder
from app.pipeline.retargeter import MODEL_DEFINITION


class TestMetadataBuilder(unittest.TestCase):
    def _make_segments(self, fps: int) -> list[ActionSegment]:
        s1 = ActionSegment(
            action_id="act_001",
            name="动作 1",
            start_frame=0,
            end_frame=30,
            start_time_sec=0.0,
            end_time_sec=1.0,
        )
        s2 = ActionSegment(
            action_id="act_002",
            name="动作 2",
            start_frame=30,
            end_frame=60,
            start_time_sec=1.0,
            end_time_sec=2.0,
        )
        return [s1, s2]

    def test_build_metadata_fields(self) -> None:
        video_meta = VideoMeta(video_id="dance_001", fps=30, duration_sec=2.0, total_frames=60)
        clip = AnimationClip(
            clip_name="dance_001_anim", start_frame=0, end_frame=59,
            joints=MODEL_DEFINITION["joints"],
            joint_rotations={j: [[1, 0, 0, 0]] for j in MODEL_DEFINITION["joints"]},
            root_positions=[[0, 0, 0]],
        )
        builder = MetadataBuilder()
        metadata = builder.build(
            video_id="dance_001",
            video_meta=video_meta,
            segments=self._make_segments(30),
            clip=clip,
            kps_path="keypoints/dance_001_kps_smoothed.npy",
            anim_path="animations/dance_001_anim.json",
        )

        self.assertEqual(metadata.video_id, "dance_001")
        self.assertEqual(metadata.fps, 30)
        self.assertEqual(metadata.total_frames, 60)
        self.assertEqual(len(metadata.actions), 2)
        self.assertEqual(metadata.calibration.status, "pending")
        self.assertEqual(len(metadata.model["joints"]), 17)
        self.assertEqual(metadata.keypoints_file, "keypoints/dance_001_kps_smoothed.npy")

    def test_annotations_all_17_dimensions_filled(self) -> None:
        builder = MetadataBuilder()
        metadata = builder.build(
            video_id="dance_001",
            video_meta=VideoMeta(video_id="dance_001", fps=30, total_frames=60),
            segments=self._make_segments(30),
        )
        for seg in metadata.actions:
            self.assertEqual(seg.annotation_source, "template")
            for field in ALL_ANNOTATION_FIELDS:
                self.assertNotEqual(
                    getattr(seg.annotations, field),
                    "",
                    f"注解维度 {field} 未补齐",
                )

    def test_json_round_trip(self) -> None:
        builder = MetadataBuilder()
        metadata = builder.build(
            video_id="dance_001",
            video_meta=VideoMeta(video_id="dance_001", fps=30, total_frames=60),
            segments=self._make_segments(30),
        )
        raw = metadata.to_json()
        # 可被重新解析，字段一致
        parsed = ActionMetadata.model_validate_json(raw)
        self.assertEqual(parsed.video_id, metadata.video_id)
        self.assertEqual(len(parsed.actions), len(metadata.actions))

    def test_save_writes_file(self) -> None:
        builder = MetadataBuilder()
        metadata = builder.build(
            video_id="dance_001",
            video_meta=VideoMeta(video_id="dance_001", fps=30, total_frames=60),
            segments=self._make_segments(30),
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = builder.save(metadata, Path(tmp) / "metadata.json")
            self.assertTrue(out.exists())
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["video_id"], "dance_001")


if __name__ == "__main__":
    unittest.main()
