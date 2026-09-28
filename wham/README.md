# WHAM 视频级 SMPL 重建 —— 最小验证（WSL2 + RTX 4060）

本目录提供在**用户本机（Windows + RTX 4060 Laptop 8GB）**跑通 WHAM（CVPR 2024 视频级 SMPL 重建）最小验证的一键脚本与文档。

> 目标：用「效果优先」的方式，把舞蹈视频重建为逐帧 SMPL 参数（姿态/体型/顶点 mesh），再接入现有软件光栅化管线渲染「真人模型复刻视频」。

---

## 0. 结论先行

| 项 | 结论 |
|----|------|
| 推理引擎 | WHAM（`yohanshin/WHAM`），`--estimate_local_only`（跳过 SLAM） |
| 运行环境 | WSL2 Ubuntu（**本机已是 26.04 LTS**）+ conda python 3.10 |
| PyTorch | **2.1.0 + cu118**（RTX 4060 = Ada sm_89，必须 ≥2.0+cu117；官方 1.11+cu113 会 `no kernel image`） |
| 核心产出 | `wham_output.pkl`（逐帧 `verts/pose/betas/trans`），供现有软件光栅化渲染 SMPL mesh |
| 最大风险 | ViTPose 依赖 mmcv 在 torch2.1 上难编译 → 提供 **Plan B（YOLOv8-pose 替换 ViTPose）** |

---

## 1. 目录与文件

```
wham/
├── setup_wsl2.sh                 # 环境一键搭建（conda+torch+WHAM+mmcv+SMPL+checkpoint）
├── run_demo.sh                   # 推理：视频 -> wham_output.pkl（+ 可选可视化）
├── patch_detector_yolopose.py    # Plan B：YOLOv8-pose 替换 ViTPose（绕开 mmcv 编译）
└── README.md                     # 本文档
```

---

## 2. 路径规划铁律（务必遵守，勿写回 C 盘）

用户已把 Ubuntu 的 vhdx 迁到 `D:\WSL\Ubuntu\ext4.vhdx`，因此 `/home/asus/` 物理上已经在 D 盘。

| 内容 | Linux 路径 | 说明 |
|------|-----------|------|
| conda | `/home/asus/miniconda3/` | vhdx 内（物理 D 盘），conda 需要 Linux 文件系统 |
| WHAM 代码仓库 | `/home/asus/wham/WHAM/` | 同上，含 mmcv 源码编译 |
| **checkpoints（约 4GB）** | `/mnt/d/wham-data/checkpoints/` | D 盘直接可见，不占 vhdx |
| **SMPL 模型** | `/mnt/d/wham-data/smpl/`（下载）→ 复制到仓库 | 同上 |
| 输入视频 | `/mnt/d/wham-data/videos/` | 同上 |
| 输出结果 | `/mnt/d/wham-data/output/` | 同上 |

脚本内部已自动把 WHAM 仓库里的 `checkpoints/`、`dataset/body_models/` 软链到 `/mnt/d/wham-data/`，大文件不落 vhdx。

> Windows 路径 ↔ WSL2 路径换算：`D:\wham-data\videos\dance.mp4` ↔ `/mnt/d/wham-data/videos/dance.mp4`。

---

## 3. 从零到跑通

### 第 1 步：Windows 侧（PowerShell，管理员）—— 确认 WSL2 + GPU 驱动

```powershell
wsl --update                       # WSL2 最新
wsl --set-default-version 2
# 确认 GPU 驱动 >= 528（WSL2 内 GPU 直通无需在 Linux 里装驱动）
```

> 本机 Ubuntu 已装好且 vhdx 已迁 D 盘，无需重新安装。若需重装：`wsl --install -d Ubuntu-22.04`（本机版本 26.04 亦可，脚本不依赖具体 Ubuntu 版本，只用 conda 的 python）。

### 第 2 步：WSL2 内跑一键搭建

```bash
cd /mnt/d/wham-data 2>/dev/null || mkdir -p /mnt/d/wham-data
# 把本项目 wham/ 目录拷到 WSL2 可见位置（或直接在 /mnt/d 下执行）
bash /mnt/d/<项目路径>/wham/setup_wsl2.sh
```

**预期输出（每一步一个 GATE）：**

```
[WHAM-SETUP] 步骤 0/9：前置检查        -> [GATE OK] GPU 直通 + D 盘可写
[WHAM-SETUP] 步骤 1/9：安装 Miniconda  -> [GATE OK] Miniconda 安装完成
[WHAM-SETUP] 步骤 2/9：创建 conda 环境 -> [GATE OK] python=3.10
[WHAM-SETUP] 步骤 3/9：安装 PyTorch    -> sm_89 support = OK        ← 关键校验
[WHAM-SETUP] 步骤 4/9：clone WHAM      -> ViTPose 子模块已拉取
[WHAM-SETUP] 步骤 5/9：安装依赖        -> sm_89 support = OK
[WHAM-SETUP] 步骤 6/9：mmcv/mmpose     -> [GATE OK] 或 [WARN] 引导 Plan B
[WHAM-SETUP] 步骤 8/9：SMPL 模型       -> [GATE OK] neutral/female/male
[WHAM-SETUP] 步骤 9/9：checkpoints     -> [GATE OK] 全部就绪（约 4GB）
```

> 幂等：任何一步失败后，修复问题直接**重跑整个脚本**，已完成的步骤会跳过，已下载的 checkpoint 会续传跳过。

### 第 3 步：注册并下载 SMPL 模型（唯一需要人工 + 账号的步骤）

SMPL 是 MPI 的研究用途模型，必须注册：

1. **SMPL 主模型**：打开 <https://smpl.is.tue.mpg.de/> → 注册（邮箱验证）→ 登录 → Downloads → 下载 **SMPL Python 版**（`SMPL_python_v.1.0.0.zip`），解压得到：
   - `basicModel_f_lbs_10_207_0_v1.0.0.pkl`
   - `basicmodel_m_lbs_10_207_0_v1.0.0.pkl`
2. **SMPLify（neutral 模型来源）**：打开 <https://smplify.is.tue.mpg.de/> → 同样注册 → 下载 `mpips_smplify_public_v2.zip`，解压后在其 `smplify_public/code/models/` 下得到：
   - `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`
3. 把以上 **3 个 pkl** 拷到 `/mnt/d/wham-data/smpl/`，然后重跑 `setup_wsl2.sh`（脚本会自动复制并重命名为 WHAM 期望的 `SMPL_NEUTRAL/FEMALE/MALE.pkl`）。

> ⚠️ SMPL 为 MPI 授权的研究用途模型，**不可外传/商用**。本项目第一版纯本地使用，符合研究用途；如后续商业化需换 Meshcapade 商业许可。

### 第 4 步：放视频并推理

```bash
# 把舞蹈视频放到 /mnt/d/wham-data/videos/（建议先用 10~30 秒短片做最小验证）
bash /mnt/d/<项目路径>/wham/run_demo.sh /mnt/d/wham-data/videos/dance.mp4
# 附加可视化（需 pyrender+OSMesa，失败不影响 pkl）：
bash /mnt/d/<项目路径>/wham/run_demo.sh /mnt/d/wham-data/videos/dance.mp4 --visualize
```

**预期输出：**

```
[WHAM-RUN] 校验环境
[WHAM-RUN] 写入 8GB 友好配置（FLIP_EVAL=0, SEQLEN=16, BATCH_SIZE=1）
[WHAM-RUN] Stage 1/2：推理并保存 wham_output.pkl
  Preprocess: 2D detection and SLAM ...   (逐帧进度条)
  Feature extraction ...                  (逐帧进度条)
  ... WHAM 推理 ...
[OK] SMPL 结果已保存: /mnt/d/wham-data/output/dance/wham_output.pkl
```

`wham_output.pkl` 内容（`save_pkl` 产出，逐帧对齐、不重采样）：

| 字段 | 形状/含义 |
|------|-----------|
| `verts` | `[T, 6890, 3]` 相机系 SMPL 顶点（即渲染用的 mesh） |
| `pose` | `[T, 72]` 相机系 SMPL 姿态（root orient + 23 body 关节，axis-angle） |
| `pose_world` / `trans_world` | 全局系（local_only 模式下与相机系相同） |
| `betas` | `[1, 10]` 体型参数 |
| `frame_ids` | `[T]` 帧号（与视频帧严格 1:1） |

---

## 4. 8GB 显存调优

`run_demo.sh` 已默认写「8GB 友好」配置。可用环境变量覆盖：

| 环境变量 | 默认 | 说明 |
|----------|------|------|
| `WHAM_FLIP_EVAL` | `0` | 设为 `1` 开测试时翻转增强（效果更好，显存翻倍，8GB 慎开） |
| `WHAM_SEQLEN` | `16` | 时序上下文窗口；越大动作越连贯，越吃显存 |
| `WHAM_OUTPUT_DIR` | `/mnt/d/wham-data/output` | 输出目录 |

**显存不足时的降级顺序**：`FLIP_EVAL=0` → `SEQLEN 16→8` → 视频抽帧（30fps→15fps，跑完对 SMPL 参数插值回 30fps）→ 缩短片段。

---

## 5. 常见报错排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `no kernel image is available for execution on the device` | torch 版本 <2.0 或 cu113 | 确认 `torch.__version__==2.1.0+cu118` 且 `get_arch_list()` 含 `sm_89`/`+PTX` |
| `torch.cuda.is_available() == False` | WSL2 GPU 直通未生效 | Windows 侧 `wsl --update` + 驱动 ≥528，WSL2 内 `nvidia-smi` 应可见 RTX 4060 |
| mmcv 编译失败（`error: command ... nvcc failed`） | torch2.1 与 mmcv 1.x 不兼容 | 走 Plan B（见下），或装 CUDA toolkit 11.8 后重试源码编译 |
| `No module named 'mmpose'` / `mmcv` | Plan A 依赖未装成 | 走 Plan B（`patch_detector_yolopose.py`） |
| `ImportError: No module named 'pyrender'`（仅 `--visualize`） | 可视化依赖缺 | `pip install pyrender trimesh` + `sudo apt-get install -y libosmesa6-dev`；或不用 `--visualize`（pkl 已产出） |
| `chumpy` 安装失败 | chumpy 与 numpy≥1.24/python≥3.12 不兼容 | 已用 conda python 3.10 + numpy 1.22.3，勿升 numpy |
| `gcc` 版本过高报错 | Ubuntu 26.04 自带 gcc 太新，老 C++ 扩展编不过 | conda 装旧 gcc：`conda install -c conda-forge gxx_linux-64=9.5` |
| Google Drive 下载失败/限流 | gdown 被限流 | 稍后重跑 `setup_wsl2.sh`（已下载的文件会跳过续传） |
| OOM（`CUDA out of memory`） | 视频过长 / FLIP_EVAL 开启 | 按第 4 节降级顺序处理 |

---

## 6. Plan B：YOLOv8-pose 替换 ViTPose（绕开 mmcv）

WHAM 官方用 **ViTPose（依赖 mmcv/mmpose/mmdet）** 做 2D 关键点，而 mmcv 1.x 在 torch2.1+cu118 上无预编译 wheel、源码编译成功率低。**Plan B** 用 ultralytics 的 **YOLOv8-pose**（自带 COCO 17 关键点，自动下载权重）替换 ViTPose，**完全不需要 mmcv/mmpose/mmdet**：

```bash
cd /home/asus/wham/WHAM
python /mnt/d/<项目路径>/wham/patch_detector_yolopose.py
# 之后照常推理（无需 mmcv / vitpose checkpoint）
bash /mnt/d/<项目路径>/wham/run_demo.sh /mnt/d/wham-data/videos/dance.mp4
```

- 输出结构与原版 `DetectionModel` 完全一致（`frame_id/bbox/keypoints`），下游 `FeatureExtractor` / `demo.py` 零改动。
- 仅支持**单人**（与舞蹈场景一致）。
- 效果上官方推荐 ViTPose，Plan B 是「先跑通」的降级方案，重建质量仍显著优于 MediaPipe Pose。

---

## 7. Top 3 风险点（预判）

1. **mmcv/ViTPose 编译失败（高风险）**：torch2.1 与 mmcv 1.x 生态不兼容，源码编译依赖 CUDA toolkit 11.8 + 兼容 gcc。→ 已内置 Plan B（YOLOv8-pose），可无缝降级，不阻塞最小验证。

2. **8GB 显存边界（中风险）**：WHAM（ViT-B 编码 + 时序 motion decoder）+ 2D 检测同进程逼近 8GB，长视频一次性前向易 OOM。→ 默认 `FLIP_EVAL=0` + `SEQLEN=16` + `BATCH_SIZE=1` + `PYTORCH_CUDA_ALLOC_CONF`；先用短片验证，再决定是否抽帧。

3. **单视频耗时（中风险）**：ViT 逐帧推理慢，粗估数分钟视频跑 30~60 分钟量级。→ `--estimate_local_only` 已跳过 SLAM；先跑 10~30 秒短片做耗时基准，再决定 fps 抽帧策略（30→15fps + 插值）。

---

## 8. 与现有管线的对接（下一步）

WHAM 产出 `wham_output.pkl` 的 `verts [T,6890,3]` 后，按架构方案直接接入现有软件光栅化 `Renderer`（`render_vertices(verts, faces)`），SMPL 的 `faces` 固定为 `[13776,3]`（从 SMPL pkl 读一次缓存），无需 OpenGL/pyrender，即可渲染出「真人 SMPL 模型复刻视频」。这一对接在 WHAM 最小验证跑通后由后端管线实现。
