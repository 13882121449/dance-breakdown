"""视频处理服务：上传、启动离线管线、状态/进度查询、元数据读取。

异步处理采用 ``threading.Thread``（daemon）承载 :class:`PipelineOrchestrator`，
处理状态保存在进程内注册表 ``_tasks`` 中（MVP 单机内存态，重启即失效，
与 architecture.md「游客本地、纯本地运行」定位一致）。
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from app.config import (
    ALLOWED_VIDEO_EXTENSIONS,
    ANIMATIONS_DIR,
    KEYPOINTS_DIR,
    MAX_UPLOAD_SIZE_BYTES,
    METADATA_DIR,
    UPLOAD_DIR,
)
from app.models.schemas import ActionMetadata, AnimationClip, KeypointSequence
from app.pipeline.pipeline import PipelineOrchestrator
from app.services import storage

# 任务状态常量
STATUS_UPLOADED: str = "uploaded"
STATUS_PROCESSING: str = "processing"
STATUS_COMPLETED: str = "completed"
STATUS_FAILED: str = "failed"


class VideoService:
    """视频上传与离线处理服务。"""

    def __init__(
        self,
        orchestrator: Optional[PipelineOrchestrator] = None,
        upload_dir: Any = UPLOAD_DIR,
        metadata_dir: Any = METADATA_DIR,
        keypoints_dir: Any = KEYPOINTS_DIR,
        animations_dir: Any = ANIMATIONS_DIR,
    ) -> None:
        self.upload_dir: Path = Path(upload_dir)
        self.metadata_dir: Path = Path(metadata_dir)
        self.keypoints_dir: Path = Path(keypoints_dir)
        self.animations_dir: Path = Path(animations_dir)
        self._orchestrator: PipelineOrchestrator = orchestrator or PipelineOrchestrator(
            progress_callback=self._on_progress
        )
        self._tasks: dict[str, dict[str, Any]] = {}
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # 上传
    # ------------------------------------------------------------------
    def upload(self, filename: str, content: bytes) -> dict[str, Any]:
        """保存上传的视频字节，返回 ``{video_id, filename, status, size_bytes}``。

        Raises:
            ValueError: 扩展名不在允许列表、或文件大小超限、或内容为空。
        """
        if not content:
            raise ValueError("上传内容为空")

        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_VIDEO_EXTENSIONS))
            raise ValueError(f"不支持的视频格式 {ext or '(无扩展名)'}，允许：{allowed}")

        if len(content) > MAX_UPLOAD_SIZE_BYTES:
            raise ValueError(
                f"文件过大：{len(content)} 字节，上限 {MAX_UPLOAD_SIZE_BYTES} 字节"
            )

        video_id = f"dance_{uuid.uuid4().hex[:12]}"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        dest = self.upload_dir / f"{video_id}{ext}"
        dest.write_bytes(content)

        with self._lock:
            self._tasks[video_id] = {
                "video_id": video_id,
                "status": STATUS_UPLOADED,
                "filename": filename,
                "size_bytes": len(content),
                "progress": {"stage": "uploaded", "percent": 0, "message": "已上传"},
                "error": None,
            }

        return {
            "video_id": video_id,
            "filename": filename,
            "status": STATUS_UPLOADED,
            "size_bytes": len(content),
        }

    # ------------------------------------------------------------------
    # 处理
    # ------------------------------------------------------------------
    def start_process(self, video_id: str) -> dict[str, Any]:
        """后台启动离线管线，立即返回 ``{video_id, status}``（幂等）。

        若已在处理中则直接返回当前状态，不重复启动。
        """
        video_path = self.get_video_path(video_id)

        with self._lock:
            task = self._tasks.get(video_id)
            if task is not None and task["status"] == STATUS_PROCESSING:
                return {"video_id": video_id, "status": STATUS_PROCESSING}
            self._tasks[video_id] = {
                "video_id": video_id,
                "status": STATUS_PROCESSING,
                "filename": (task or {}).get("filename", video_path.name),
                "size_bytes": (task or {}).get("size_bytes", 0),
                "progress": {"stage": "queued", "percent": 0, "message": "排队中"},
                "error": None,
            }

        thread = threading.Thread(
            target=self._run_pipeline,
            args=(video_id, video_path),
            name=f"pipeline-{video_id}",
            daemon=True,
        )
        thread.start()
        return {"video_id": video_id, "status": STATUS_PROCESSING}

    def get_status(self, video_id: str) -> dict[str, Any]:
        """返回 ``{video_id, status, progress, error}``。

        若内存态无记录但元数据已存在，视为 ``completed``（进程重启后的兜底）。
        """
        with self._lock:
            task = self._tasks.get(video_id)

        if task is not None:
            return {
                "video_id": video_id,
                "status": task["status"],
                "progress": task["progress"],
                "error": task["error"],
            }

        # 兜底：进程重启后从磁盘判定
        if storage.metadata_file(self.metadata_dir, video_id).exists():
            return {
                "video_id": video_id,
                "status": STATUS_COMPLETED,
                "progress": {"stage": "done", "percent": 100, "message": "完成"},
                "error": None,
            }
        if self._video_file_exists(video_id):
            return {
                "video_id": video_id,
                "status": STATUS_UPLOADED,
                "progress": {"stage": "uploaded", "percent": 0, "message": "已上传"},
                "error": None,
            }
        raise FileNotFoundError(f"视频不存在：{video_id}")

    def get_progress(self, video_id: str) -> dict[str, Any]:
        """返回处理进度（``{stage, percent, message}``）。"""
        return self.get_status(video_id)["progress"]

    def get_metadata(self, video_id: str) -> ActionMetadata:
        """读取已生成的元数据（处理完成后）。"""
        return storage.load_metadata(self.metadata_dir, video_id)

    def get_video_path(self, video_id: str) -> Path:
        """查找上传的视频文件路径（按允许扩展名搜索）。"""
        for ext in ALLOWED_VIDEO_EXTENSIONS:
            candidate = self.upload_dir / f"{video_id}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"视频文件不存在：{video_id}")

    def get_video_url(self, video_id: str) -> str:
        """返回前端可直接访问的视频 URL（相对路径）。"""
        return f"/api/video/{video_id}/file"

    # ------------------------------------------------------------------
    # 播放数据下发
    # ------------------------------------------------------------------
    def get_playback(self, video_id: str) -> dict[str, Any]:
        """返回播放所需的完整数据（元数据 + 动画 + 视频地址 + 关键点摘要）。"""
        metadata = self.get_metadata(video_id)
        animation = self.get_animation(video_id)
        return {
            "video_id": metadata.video_id,
            "fps": metadata.fps,
            "total_frames": metadata.total_frames,
            "duration_sec": metadata.duration_sec,
            "video_url": self.get_video_url(video_id),
            "model": metadata.model,
            "actions": [seg.model_dump() for seg in metadata.actions],
            "calibration": metadata.calibration.model_dump(),
            "animation": animation.model_dump(),
            "keypoints": {
                "file": metadata.keypoints_file,
                "num_keypoints": 33,
                "total_frames": metadata.total_frames,
            },
        }

    def get_animation(self, video_id: str) -> AnimationClip:
        """读取全量动画帧。"""
        return storage.load_animation(self.animations_dir, video_id)

    def get_frame_keypoints(self, video_id: str, frame: int) -> dict[str, Any]:
        """返回指定帧的 ``[33, 4]`` 关键点（供校准 UI）。"""
        kps = self._load_keypoints(video_id)
        if frame < 0 or frame >= kps.total_frames:
            raise IndexError(
                f"帧号越界：{frame}（total_frames={kps.total_frames}）"
            )
        return {
            "video_id": video_id,
            "frame": frame,
            "keypoints": storage.frame_keypoints_to_list(kps.data[frame]),
        }

    def get_all_keypoints(self, video_id: str) -> dict[str, Any]:
        """返回全量 ``[T, 33, 4]`` 关键点。

        前端播放时一次性拉取到内存后按帧取用，避免逐帧请求导致骨架掉帧。
        """
        kps = self._load_keypoints(video_id)
        return {
            "video_id": video_id,
            "total_frames": kps.total_frames,
            "num_keypoints": kps.num_keypoints,
            "keypoints": storage.keypoints_to_list(kps.data),
        }

    def _load_keypoints(self, video_id: str) -> KeypointSequence:
        """读取平滑后关键点序列。"""
        return storage.load_keypoints(self.keypoints_dir, video_id, smoothed=True)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _run_pipeline(self, video_id: str, video_path: Path) -> None:
        try:
            metadata = self._orchestrator.process(video_path, video_id)
            with self._lock:
                self._tasks[video_id]["status"] = STATUS_COMPLETED
                self._tasks[video_id]["progress"] = {
                    "stage": "done",
                    "percent": 100,
                    "message": "完成",
                }
                self._tasks[video_id]["metadata"] = metadata
                self._tasks[video_id]["error"] = None
        except Exception as exc:  # noqa: BLE001 - 记录失败原因供轮询返回
            with self._lock:
                self._tasks[video_id]["status"] = STATUS_FAILED
                self._tasks[video_id]["progress"] = {
                    "stage": "failed",
                    "percent": 0,
                    "message": "处理失败",
                }
                self._tasks[video_id]["error"] = f"{type(exc).__name__}: {exc}"

    def _on_progress(
        self, video_id: str, stage: str, percent: int, message: str
    ) -> None:
        """管线进度回调：更新内存态进度。"""
        with self._lock:
            task = self._tasks.get(video_id)
            if task is not None:
                task["progress"] = {
                    "stage": stage,
                    "percent": percent,
                    "message": message,
                }

    def _video_file_exists(self, video_id: str) -> bool:
        return any(
            (self.upload_dir / f"{video_id}{ext}").exists()
            for ext in ALLOWED_VIDEO_EXTENSIONS
        )


# 进程内单例工厂（路由通过 FastAPI 依赖注入获取，测试可覆盖）
_video_service: Optional[VideoService] = None


def get_video_service() -> VideoService:
    """返回进程内单例 VideoService。"""
    global _video_service
    if _video_service is None:
        _video_service = VideoService()
    return _video_service
