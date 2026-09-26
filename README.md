# 舞蹈动作拆解软件

上传 K-pop 舞蹈视频 → 自动识别拆解动作 → 3D 人物逐步重现 → 附动作注解。
舞种定位：K-pop 男团编舞。

## 技术栈

- **后端**：Python + FastAPI（OpenCV / MediaPipe / librosa）
- **前端**：Vite + React + TypeScript + MUI + Tailwind CSS + Zustand + Three.js
- **数据**：本地文件系统 + JSON 元数据

## 目录结构

```
dance-breakdown/
├── backend/          # FastAPI 后端
├── frontend/         # Vite + React 前端
├── data/             # 运行时数据（uploads/frames/keypoints/... 等）
├── docs/             # 架构图
├── PRD.md            # 产品需求
└── architecture.md   # 系统架构设计
```

## 启动后端

```bash
cd backend
python -m venv .venv
./.venv/Scripts/python -m pip install -r requirements.txt
./.venv/Scripts/python -m uvicorn app.main:app --reload
```

> 依赖较大时可用清华镜像：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`

健康检查：<http://localhost:8000/health>，返回 `{"code":0,"message":"ok","data":{}}`。

### mediapipe 模型文件（姿态估计/人像分割需要）

mediapipe 1.0+ 使用 Tasks API，需手动下载模型文件到 `backend/models/`（提供一键脚本，含重试与完整性校验）：

```bash
cd backend
./.venv/Scripts/python scripts/download_models.py all
# 或分别下载：pose_lite / pose_full / selfie
./.venv/Scripts/python scripts/download_models.py pose_lite
```

脚本会依次尝试官方源与 ghproxy 镜像，并校验文件完整性。手动下载也可：

```bash
cd backend/models
# 人像分割（selfie_segmenter，单文件 TFLite）
curl -L -o selfie_segmenter.tflite "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
# 姿态估计（pose_landmarker_lite，.task 为 zip 容器）
curl -L -o pose_landmarker_lite.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
```

> 注意：`.task` 文件是 zip 容器，下载后可用 `python -c "import zipfile; print(zipfile.ZipFile('pose_landmarker_lite.task').namelist())"` 校验是否完整（应能列出若干 .tflite 条目）。

## 启动前端

```bash
cd frontend
npm install
npm run dev
```

浏览器访问 <http://localhost:5173>。

## 说明

已完成 **T01-T05 全部功能**：
- T01 基础设施：FastAPI 后端骨架 + Vite/React 前端骨架 + data 目录
- T02/T03 后端管线：帧抽取、人像分割、姿态估计、时序平滑、动作切分（节拍为主+能量兜底）、
  骨骼重定向（方向向量法四元数）、元数据生成（17 维注解）、管线编排
- T04 API 服务层：上传/处理/状态轮询/元数据/动作/注解/播放/校准 11 个端点
- T05 前端：视频上传、视频播放、动作列表、3D 骨架重现、17 维注解编辑、时间轴

后端单测（67 个，全绿）：
`cd backend && PYTHONPATH=. ./.venv/Scripts/python -m unittest discover -s tests -v`

端到端冒烟：`cd backend && ./.venv/Scripts/python scripts/e2e_smoke.py`
