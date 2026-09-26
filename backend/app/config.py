"""后端应用配置：路径、上传限制、统一帧率、CORS 等。

所有数据目录均位于项目根目录下的 ``data/`` 中，通过 :func:`ensure_directories`
在应用启动时创建，避免运行期出现目录不存在的问题。
"""

from pathlib import Path

# 项目根目录（backend/app/config.py -> backend/app -> backend -> 项目根）
BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# 数据目录（运行时生成，git 忽略）
# ---------------------------------------------------------------------------
DATA_DIR: Path = BASE_DIR / "data"
UPLOAD_DIR: Path = DATA_DIR / "uploads"
FRAMES_DIR: Path = DATA_DIR / "frames"
FRAMES_MASKED_DIR: Path = DATA_DIR / "frames_masked"
KEYPOINTS_DIR: Path = DATA_DIR / "keypoints"
ANIMATIONS_DIR: Path = DATA_DIR / "animations"
METADATA_DIR: Path = DATA_DIR / "metadata"

# ---------------------------------------------------------------------------
# 上传限制
# ---------------------------------------------------------------------------
# 单文件最大上传大小（字节），默认 500 MB
MAX_UPLOAD_SIZE_BYTES: int = 500 * 1024 * 1024
# 允许的视频扩展名（不含点，统一小写）
ALLOWED_VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".mov", ".m4v", ".avi", ".webm"}
)

# ---------------------------------------------------------------------------
# 处理参数
# ---------------------------------------------------------------------------
# 全系统统一帧率：处理前将视频重采样到该 fps（帧号为唯一时间轴主键）
TARGET_FPS: int = 30

# ---------------------------------------------------------------------------
# mediapipe Tasks 模型（新版 Tasks API 需显式提供 .task/.tflite 模型文件）
# ---------------------------------------------------------------------------
MODELS_DIR: Path = BASE_DIR / "backend" / "models"
# 姿态估计模型：lite 版（体积更小、加载更快；full/heavy 可按需替换）
POSE_LANDMARKER_MODEL: Path = MODELS_DIR / "pose_landmarker_lite.task"
SELFIE_SEGMENTATION_MODEL: Path = MODELS_DIR / "selfie_segmenter.tflite"

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
# 允许跨域的前端来源（Vite dev server）
CORS_ORIGINS: list[str] = ["http://localhost:5173"]

# ---------------------------------------------------------------------------
# API 统一响应格式
# ---------------------------------------------------------------------------
# 成功时的业务码
API_CODE_OK: int = 0
API_MESSAGE_OK: str = "ok"


def ensure_directories() -> None:
    """确保所有运行时数据目录存在（幂等）。"""
    for directory in (
        DATA_DIR,
        UPLOAD_DIR,
        FRAMES_DIR,
        FRAMES_MASKED_DIR,
        KEYPOINTS_DIR,
        ANIMATIONS_DIR,
        METADATA_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
