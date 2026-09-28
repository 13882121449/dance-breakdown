#!/usr/bin/env bash
# =============================================================================
# WHAM（4D-Humans 视频级 SMPL 重建）WSL2 一键环境搭建脚本
# -----------------------------------------------------------------------------
# 目标机器 : Windows + RTX 4060 Laptop (Ada Lovelace, sm_89, 8GB) + WSL2 Ubuntu
# 执行位置 : WSL2 Ubuntu 终端内（用户名 asus，vhdx 已迁移到 D 盘）
#
# 路径铁律（务必遵守，勿改回 C 盘）:
#   - conda / WHAM 代码仓库 / mmcv 源码编译  -> /home/asus/（vhdx 内，物理在 D 盘）
#     原因：conda 和 mmcv 需要 Linux 原生文件系统（符号链接/权限），/mnt/d 不支持
#   - checkpoint(约4GB) / SMPL 模型 / 输入视频 / 输出 -> /mnt/d/wham-data/（D 盘直接可见）
#
# 幂等性 : 可重复执行，已完成的步骤自动跳过（GATE 标记）。
# 用法   :
#   bash setup_wsl2.sh                        # 完整安装（含 checkpoint 下载）
#   WHAM_SKIP_CKPT=1 bash setup_wsl2.sh       # 跳过 4GB checkpoint 下载
#   SMPL_DIR=/mnt/d/wham-data/smpl bash setup_wsl2.sh   # 指定 SMPL pkl 目录
# =============================================================================
set -euo pipefail

# 脚本所在目录（用于引用同目录的 run_demo.sh / patch_detector_yolopose.py）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# 可配置项（如需修改，用环境变量覆盖，勿直接改脚本里的默认值）
# ---------------------------------------------------------------------------
PY_VER="${WHAM_PY_VER:-3.10}"          # WHAM 官方 python=3.9，3.10 兼容 numpy1.22+torch2.1
ENV_NAME="${WHAM_ENV_NAME:-wham}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
WHAM_DIR="${WHAM_DIR:-$HOME/wham/WHAM}"
DATA_ROOT="${WHAM_DATA_ROOT:-/mnt/d/wham-data}"
CKPT_DIR="$DATA_ROOT/checkpoints"
SMPL_DIR="${SMPL_DIR:-$DATA_ROOT/smpl}"
VIDEO_DIR="$DATA_ROOT/videos"
OUTPUT_DIR="$DATA_ROOT/output"
CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
WHAM_REPO="https://github.com/yohanshin/WHAM.git"
PYTORCH_INDEX="https://mirrors.aliyun.com/pytorch-wheels/cu118/"
TORCH_VERSION="2.1.0+cu118"
TORCHVISION_VERSION="0.16.0+cu118"
CUDA_ARCH="8.9"                        # RTX 4060 = Ada = sm_89

# 临时目录指向根分区（WSL2 的 /tmp 是 tmpfs 仅 3.8G，torch 2.3GB 下载会撑爆）
TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "$TMPDIR"
export TMPDIR

# ---------------------------------------------------------------------------
# 日志 / GATE 工具
# ---------------------------------------------------------------------------
C_RESET='\033[0m'; C_CYAN='\033[1;36m'; C_GREEN='\033[1;32m'
C_YELLOW='\033[1;33m'; C_RED='\033[1;31m'
log()  { echo -e "${C_CYAN}[WHAM-SETUP]${C_RESET} $*"; }
info() { echo -e "           $*"; }
gate() { echo -e "${C_GREEN}[GATE OK]${C_RESET}   $*"; }
warn() { echo -e "${C_YELLOW}[WARN]${C_RESET}     $*"; }
fail() { echo -e "${C_RED}[GATE FAIL]${C_RESET} $*"; }

die() { echo -e "${C_RED}[FATAL]${C_RESET} $*" >&2; exit 1; }

# 全局变量：记录 mmcv 是否安装成功（供 Plan B 提示用）
MMCV_OK=0

# ---------------------------------------------------------------------------
# 0. 前置检查（GATE 0）
# ---------------------------------------------------------------------------
preflight() {
  log "步骤 0/9：前置检查"
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi 不可用：请先在 Windows 侧装好 NVIDIA 驱动(>=528) 并确认 WSL2 GPU 直通。"
  info "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"

  [ -d /mnt/d ] || die "未检测到 /mnt/d：请确认 D 盘已挂载到 WSL2（/etc/wsl.conf 或 Windows 资源管理器）。"
  info "D 盘挂载: /mnt/d 存在"

  local free_gb
  free_gb=$(df -BG /mnt/d 2>/dev/null | awk 'NR==2{print $4}' | tr -d 'G')
  info "D 盘剩余空间: ${free_gb}GB（建议 >= 20GB）"

  mkdir -p "$DATA_ROOT" "$CKPT_DIR" "$SMPL_DIR" "$VIDEO_DIR" "$OUTPUT_DIR"
  gate "前置检查通过（GPU 直通 + D 盘可写）"
}

# ---------------------------------------------------------------------------
# 1. Miniconda
# ---------------------------------------------------------------------------
install_conda() {
  log "步骤 1/9：安装 Miniconda -> $CONDA_DIR"
  if [ -x "$CONDA_DIR/bin/conda" ]; then
    gate "conda 已存在，跳过（$CONDA_DIR/bin/conda）"
    return 0
  fi
  local installer="/tmp/miniconda_installer.sh"
  info "下载 Miniconda 安装器 ..."
  wget -q "$CONDA_URL" -O "$installer" || curl -fsSL "$CONDA_URL" -o "$installer"
  bash "$installer" -b -p "$CONDA_DIR" -u
  rm -f "$installer"
  "$CONDA_DIR/bin/conda" init bash >/dev/null 2>&1 || true
  gate "Miniconda 安装完成"
}

# 返回一个已激活 env 的 conda 命令前缀（无需交互式 shell）
conda_run() {
  "$CONDA_DIR/bin/conda" run -n "$ENV_NAME" --no-capture-output "$@"
}

# ---------------------------------------------------------------------------
# 2. 创建 conda 环境
# ---------------------------------------------------------------------------
create_env() {
  log "步骤 2/9：创建 conda 环境 '$ENV_NAME'（python=$PY_VER）"
  if "$CONDA_DIR/bin/conda" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    gate "环境 '$ENV_NAME' 已存在，跳过"
    return 0
  fi
  "$CONDA_DIR/bin/conda" create -y -n "$ENV_NAME" "python=$PY_VER" pip
  gate "conda 环境创建完成（python=$PY_VER）"
}

# ---------------------------------------------------------------------------
# 3. PyTorch 2.1.0 + cu118（Ada sm_89 关键步骤，不可用官方 1.11+cu113）
# ---------------------------------------------------------------------------
install_pytorch() {
  log "步骤 3/9：安装 PyTorch $TORCH_VERSION + cu118（RTX 4060 = sm_89 必需 >=2.0+cu117）"
  conda_run python -c "import torch" 2>/dev/null && {
    gate "PyTorch 已安装，跳过"
    return 0
  }
  conda_run pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
    -f "$PYTORCH_INDEX"
  conda_run pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
    -f "$PYTORCH_INDEX"  # 二次确认（cu118 索引偶发装成 CPU 版）
  verify_torch
}

verify_torch() {
  log "校验 PyTorch sm_89 kernel ..."
  conda_run python - <<'PY'
import torch
print("cuda_available   =", torch.cuda.is_available())
print("device_name      =", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("torch_version    =", torch.__version__)
if torch.cuda.is_available():
    archs = torch.cuda.get_arch_list()
    print("arch_list        =", archs)
    ok = any(("sm_89" in a) or ("sm_90" in a) or ("+PTX" in a) or ("compute_89" in a) for a in archs)
    if not ok:
        raise SystemExit("arch_list 不含 sm_89/+PTX，PyTorch 版本可能过低")
print("sm_89 support    = OK")
PY
  gate "PyTorch sm_89 kernel 校验通过"
}

# ---------------------------------------------------------------------------
# 4. clone WHAM（含 ViTPose 子模块；DPVO 在 --estimate_local_only 下不需要）
# ---------------------------------------------------------------------------
clone_wham() {
  log "步骤 4/9：clone WHAM 仓库 -> $WHAM_DIR"
  if [ -d "$WHAM_DIR/.git" ]; then
    gate "WHAM 仓库已存在，跳过 clone"
  else
    mkdir -p "$(dirname "$WHAM_DIR")"
    # 只拉 WHAM + ViTPose 子模块，跳过 DPVO（SLAM，本地坐标模式不需要）
    git clone "$WHAM_REPO" "$WHAM_DIR"
    ( cd "$WHAM_DIR" && git submodule update --init --recursive third-party/ViTPose )
    gate "WHAM 仓库 clone 完成（ViTPose 子模块已拉取）"
  fi
  # 立即建立 checkpoints/dataset -> /mnt/d 软链，保证后续 SMPL 放置、辅助数据解压都落 D 盘
  link_heavy_dirs
}

# ---------------------------------------------------------------------------
# 5. 安装 WHAM 依赖（过滤掉会覆盖 torch/numpy/mmcv 的 pin，装完重钉 torch）
# ---------------------------------------------------------------------------
install_wham_deps() {
  log "步骤 5/9：安装 WHAM 依赖（跳过 torch/mmcv 的旧 pin）"
  local req="$WHAM_DIR/requirements.txt"
  [ -f "$req" ] || die "未找到 $req，WHAM 仓库可能不完整"

  # 过滤：torch/torchvision 已装 cu118；mmcv==1.3.9 与 torch2.1 不兼容（第 6 步单独处理）
  local filtered="/tmp/wham_requirements_filtered.txt"
  grep -viE '^(torch|torchvision|mmcv)|chumpy' "$req" > "$filtered"
  info "安装条目（过滤后）: $(wc -l < "$filtered") 行"

  # chumpy 需要 numpy<1.24 且 python<3.12，此处用官方 numpy==1.22.3
  conda_run pip install "numpy==1.22.3"
  conda_run pip install -r "$filtered" || warn "部分依赖安装有 warning，继续（关键依赖将单独校验）"
  # 强制重钉 torch，防止 requirements 或依赖把 torch 拉回 CPU/旧版
  conda_run pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" -f "$PYTORCH_INDEX"
  rm -f "$filtered"
  verify_torch
  gate "WHAM 依赖安装完成"
}

# ---------------------------------------------------------------------------
# 6. ViTPose 的 mmcv/mmpose（Plan A：源码编译；失败则引导 Plan B）
# ---------------------------------------------------------------------------
install_mmcv() {
  log "步骤 6/9：安装 mmcv-full + mmdet + mmpose（ViTPose 2D 关键点依赖）"

  # 默认走 Plan B（YOLOv8-pose）：chumpy/mmpose 是 2015/2021 老包，py3.10+torch2.1 下构建失败
  if [ "${WHAM_USE_PLAN_B:-1}" = "1" ]; then
    warn "走 Plan B（YOLOv8-pose），跳过 mmcv/mmpose/ViTPose 老依赖链"
    MMCV_OK=0
    conda_run pip install ultralytics 2>/dev/null && gate "ultralytics（Plan B 依赖）安装完成" || warn "ultralytics 安装失败（稍后手动装）"
    return 0
  fi

  # 若已能 import mmpose，直接通过
  if conda_run python -c "import mmcv, mmpose" 2>/dev/null; then
    MMCV_OK=1; gate "mmcv/mmpose 已可用，跳过"; return 0
  fi

  # 6a. 预编译 wheel 快速尝试（torch2.1 一般无 1.x wheel，失败即转源码）
  if conda_run pip install "mmcv-full==1.7.2" -f "https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html" 2>/dev/null; then
    MMCV_OK=1
  else
    warn "无 torch2.1+cu118 的 mmcv-full 1.x 预编译 wheel，转源码编译"
    # 6b. 源码编译 mmcv-full（需要 nvcc + CUDA toolkit）
    if ! command -v nvcc >/dev/null 2>&1; then
      warn "未检测到 nvcc。mmcv 源码编译需要 CUDA toolkit 11.8。"
      warn "可执行（需 sudo，约 2.5GB，装在 /home 之外不影响 D 盘规划）:"
      info "  1) wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run"
      info "  2) sudo sh cuda_11.8.0_520.61.05_linux.run --toolkit --silent --override"
      info "  3) 重新执行本脚本"
      warn "或直接跳过 mmcv，走 Plan B（YOLOv8-pose 替换 ViTPose，见 README / patch_detector_yolopose.py）"
      MMCV_OK=0
      return 0
    fi
    info "nvcc: $(nvcc --version | tail -1)"
    if conda_run env MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="$CUDA_ARCH" \
        pip install "mmcv-full==1.7.2" --no-cache-dir 2>/dev/null; then
      MMCV_OK=1
    else
      warn "mmcv-full 源码编译失败（常见于 torch2.1 API 不兼容 / gcc 版本）"
      warn "请走 Plan B：bash patch_detector_yolopose.py，用 YOLOv8-pose 替换 ViTPose，无需 mmcv。"
      MMCV_OK=0
      return 0
    fi
  fi

  if [ "$MMCV_OK" = "1" ]; then
    conda_run pip install "mmdet==2.28.2" "mmpose==0.29.0" "xtcocotools"
    gate "mmcv-full/mmdet/mmpose 安装完成"
  fi
}

# ---------------------------------------------------------------------------
# 7. 安装 ViTPose（WHAM 的 third-party 子模块）
# ---------------------------------------------------------------------------
install_vitpose() {
  if [ "$MMCV_OK" != "1" ]; then
    warn "跳过 ViTPose 安装（mmcv 不可用）。若走 Plan A，请先解决 mmcv 再执行:"
    info "  cd $WHAM_DIR && pip install -v -e third-party/ViTPose"
    return 0
  fi
  log "步骤 7/9：安装 ViTPose（pip install -e third-party/ViTPose）"
  ( cd "$WHAM_DIR" && conda_run pip install -e third-party/ViTPose )
  gate "ViTPose 安装完成"
}

# ---------------------------------------------------------------------------
# 8. SMPL 模型（需用户注册下载，占位符 $SMPL_DIR，绝不硬编码）
# ---------------------------------------------------------------------------
install_smpl() {
  log "步骤 8/9：放置 SMPL 身体模型（需注册下载）"
  local dst="$WHAM_DIR/dataset/body_models/smpl"
  mkdir -p "$dst"

  # 用 $SMPL_DIR 下的原始 pkl，映射为 WHAM 期望的文件名
  declare -A MAPPING=(
    ["SMPL_NEUTRAL.pkl"]="basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
    ["SMPL_FEMALE.pkl"]="basicModel_f_lbs_10_207_0_v1.0.0.pkl"
    ["SMPL_MALE.pkl"]="basicModel_m_lbs_10_207_0_v1.0.0.pkl"
  )
  # male 模型官方 zip 里是 basicmodel_m_...（小写 m），仅 male 认两种写法
  local missing=0
  for dst_name in SMPL_NEUTRAL.pkl SMPL_FEMALE.pkl SMPL_MALE.pkl; do
    if [ -s "$dst/$dst_name" ]; then
      info "已存在: $dst_name"; continue
    fi
    local cands=("${MAPPING[$dst_name]}")
    [ "$dst_name" = "SMPL_MALE.pkl" ] && cands+=("basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
    local src_found=""
    for cand in "${cands[@]}"; do
      if [ -s "$SMPL_DIR/$cand" ]; then src_found="$SMPL_DIR/$cand"; break; fi
    done
    if [ -n "$src_found" ]; then
      cp "$src_found" "$dst/$dst_name"
      info "已复制: $src_found -> $dst_name"
    else
      missing=1
      fail "缺少 $dst_name（在 $SMPL_DIR 下未找到）"
    fi
  done

  if [ "$missing" = "1" ]; then
    warn "SMPL 模型不完整。请先注册并下载后，把 3 个 pkl 放到 $SMPL_DIR 再重跑本脚本："
    info "  SMPL   注册: https://smpl.is.tue.mpg.de/  (下载 SMPL_python_v.1.0.0.zip -> female/male)"
    info "  SMPLify注册: https://smplify.is.tue.mpg.de/ (下载 mpips_smplify_public_v2.zip -> neutral)"
    info "  原始文件名: basicModel_neutral_lbs_10_207_0_v1.0.0.pkl / basicModel_f_lbs_10_207_0_v1.0.0.pkl / basicModel_m_lbs_10_207_0_v1.0.0.pkl"
  else
    gate "SMPL 模型就绪（neutral/female/male）"
  fi
}

# ---------------------------------------------------------------------------
# 9. 下载 checkpoints（约 4GB -> /mnt/d，跳过 SLAM 专用 dpvo.pth）
# ---------------------------------------------------------------------------
install_checkpoints() {
  log "步骤 9/9：下载 checkpoints -> $CKPT_DIR（约 4GB，写 D 盘）"
  if [ "${WHAM_SKIP_CKPT:-0}" = "1" ]; then
    warn "WHAM_SKIP_CKPT=1，跳过 checkpoint 下载"; return 0
  fi

  # 先建立符号链接（checkpoints/dataset -> /mnt/d），使后续解压直接落 D 盘
  link_heavy_dirs

  # Google Drive 文件 ID（来自 WHAM 官方 fetch_demo_data.sh）
  declare -A FILES=(
    ["wham_vit_bedlam_w_3dpw.pth.tar"]="19qkI-a6xuwob9_RFNSPWf1yWErwVVlks"  # demo.yaml 默认权重
    ["wham_vit_w_3dpw.pth.tar"]="1i7kt9RlCCCNEW2aYaDWVr-G778JkLNcB"
    ["hmr2a.ckpt"]="1J6l8teyZrL0zFzHhzkC7efRhU0ZJ5G9Y"
    ["yolov8x.pt"]="1zJ0KP23tXD42D47cw1Gs7zE2BA_V_ERo"
    ["vitpose-h-multi-coco.pth"]="1xyF7F3I7lWtdq82xmEPVQ5zl4HaasBso"
    ["body_models.tar.gz"]="1pbmzRbWGgae6noDIyQOnohzaVnX_csUZ"       # SMPL 辅助数据(J_regressor等)
  )

  command -v gdown >/dev/null 2>&1 || conda_run pip install -U "gdown>=4.7"

  local missing=0
  for name in "${!FILES[@]}"; do
    local out="$CKPT_DIR/$name"
    if [ -s "$out" ]; then
      info "已存在: $name ($(du -h "$out" | cut -f1))"; continue
    fi
    info "下载 $name ..."
    if gdown --id "${FILES[$name]}" -O "$out" --quiet; then
      [ -s "$out" ] && info "完成: $name ($(du -h "$out" | cut -f1))" || { fail "$name 下载为空"; missing=1; }
    else
      fail "$name 下载失败（Google Drive 限流？可稍后重跑本脚本，已下载的会跳过）"
      missing=1
    fi
  done

  # 解压 SMPL 辅助数据（幂等：用哨兵文件标记，tar 顶层为 body_models/）
  if [ -s "$CKPT_DIR/body_models.tar.gz" ] && [ ! -f "$DATA_ROOT/dataset/.body_models_extracted" ]; then
    info "解压 body_models.tar.gz -> $WHAM_DIR/dataset/"
    mkdir -p "$WHAM_DIR/dataset"
    tar -xzf "$CKPT_DIR/body_models.tar.gz" -C "$WHAM_DIR/dataset/"
    touch "$DATA_ROOT/dataset/.body_models_extracted"
  fi

  if [ "$missing" = "1" ]; then
    warn "部分 checkpoint 缺失，可重跑本脚本续传（幂等）"
  else
    gate "checkpoints 全部就绪"
  fi
}

# 将 WHAM 仓库里的 checkpoints / dataset 指向 /mnt/d（大文件不落 vhdx）
# 注：WHAM 仓库根目录无 dataset/ 与 checkpoints/（由 fetch 脚本运行时创建），可整体软链
link_heavy_dirs() {
  log "建立大文件符号链接（checkpoints / dataset -> /mnt/d/wham-data/...）"

  mkdir -p "$CKPT_DIR" "$DATA_ROOT/dataset"

  # checkpoints
  if [ -e "$WHAM_DIR/checkpoints" ] && [ ! -L "$WHAM_DIR/checkpoints" ]; then
    warn "$WHAM_DIR/checkpoints 已是非链接目录，保留不动（若为空请手动删除后重跑）"
  elif [ ! -e "$WHAM_DIR/checkpoints" ]; then
    ln -s "$CKPT_DIR" "$WHAM_DIR/checkpoints"
    info "checkpoints -> $CKPT_DIR"
  fi

  # dataset（整体软链；body_models/smpl、辅助数据都在其中）
  if [ -e "$WHAM_DIR/dataset" ] && [ ! -L "$WHAM_DIR/dataset" ]; then
    warn "$WHAM_DIR/dataset 已是非链接目录，保留不动"
  elif [ ! -e "$WHAM_DIR/dataset" ]; then
    ln -s "$DATA_ROOT/dataset" "$WHAM_DIR/dataset"
    info "dataset -> $DATA_ROOT/dataset"
  fi
}

# ---------------------------------------------------------------------------
# 最终 GATE
# ---------------------------------------------------------------------------
final_gate() {
  log "最终校验"
  echo "--------------------------------------------------------------------------------"
  conda_run python - <<PY
import importlib, os
print(f"{'模块':<24}{'状态':<10}")
for m in ["torch", "numpy", "smplx", "ultralytics", "timm", "mmcv", "mmpose", "mmdet"]:
    try:
        importlib.import_module(m)
        print(f"{m:<24}OK")
    except Exception as e:
        print(f"{m:<24}MISSING ({type(e).__name__})")
import torch
print()
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU", torch.cuda.get_device_name(0), "| arch", torch.cuda.get_arch_list())
PY
  echo "--------------------------------------------------------------------------------"

  local smpl_ok=1 ckpt_ok=1
  for f in SMPL_NEUTRAL.pkl SMPL_FEMALE.pkl SMPL_MALE.pkl; do
    [ -s "$WHAM_DIR/dataset/body_models/smpl/$f" ] || { smpl_ok=0; fail "SMPL 缺失: $f"; }
  done
  for f in wham_vit_bedlam_w_3dpw.pth.tar hmr2a.ckpt yolov8x.pt; do
    [ -s "$WHAM_DIR/checkpoints/$f" ] || { ckpt_ok=0; fail "checkpoint 缺失: $f"; }
  done
  [ "$smpl_ok" = "1" ] && gate "SMPL 模型完整"
  [ "$ckpt_ok" = "1" ] && gate "checkpoints 完整"

  if [ "$MMCV_OK" = "1" ] && [ -s "$WHAM_DIR/checkpoints/vitpose-h-multi-coco.pth" ]; then
    gate "Plan A（ViTPose）就绪，可直接跑 demo"
  else
    warn "Plan A 未就绪（mmcv 或 ViTPose checkpoint 缺失）"
    info "  -> 推荐先走 Plan B（YOLOv8-pose，无需 mmcv/ViTPose）:"
    info "     cd $WHAM_DIR && python \"$SCRIPT_DIR/patch_detector_yolopose.py\""
  fi

  echo ""
  log "下一步：把舞蹈视频放到 $VIDEO_DIR，然后执行:"
  info "  bash \"$SCRIPT_DIR/run_demo.sh\" $VIDEO_DIR/xxx.mp4"
}

# ---------------------------------------------------------------------------
main() {
  preflight
  install_conda
  create_env
  install_pytorch
  clone_wham
  install_wham_deps
  install_mmcv
  install_vitpose
  install_smpl
  install_checkpoints
  final_gate
}

main "$@"
