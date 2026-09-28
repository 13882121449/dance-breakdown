#!/usr/bin/env bash
# =============================================================================
# WHAM 推理脚本：舞蹈视频 -> 逐帧 SMPL 结果（wham_output.pkl）+ 可视化 mp4
# -----------------------------------------------------------------------------
# 前置：已执行 setup_wsl2.sh 并完成 SMPL 注册下载 + checkpoint 下载。
# 输出：$OUTPUT_DIR/<视频名>/wham_output.pkl   (per-frame verts/pose/betas/trans)
#       $OUTPUT_DIR/<视频名>/output.mp4        (可视化，需 pyrender/OSMesa)
#
# 8GB 显存策略：默认 FLIP_EVAL=False（省一半显存）+ BATCH_SIZE=1；可用环境变量调优。
# 铁律：统一帧号，不重采样；建议先用 10~30 秒短片段做最小验证。
#
# 用法:
#   bash run_demo.sh /mnt/d/wham-data/videos/dance.mp4           # 基础（出 pkl）
#   bash run_demo.sh /mnt/d/wham-data/videos/dance.mp4 --visualize  # 附加可视化
#   WHAM_FLIP_EVAL=1 WHAM_SEQLEN=32 bash run_demo.sh video.mp4   # 效果优先(内存充足时)
# =============================================================================
set -euo pipefail

C_RESET='\033[0m'; C_CYAN='\033[1;36m'; C_GREEN='\033[1;32m'
C_YELLOW='\033[1;33m'; C_RED='\033[1;31m'
log()  { echo -e "${C_CYAN}[WHAM-RUN]${C_RESET} $*"; }
gate() { echo -e "${C_GREEN}[OK]${C_RESET}      $*"; }
warn() { echo -e "${C_YELLOW}[WARN]${C_RESET}    $*"; }
die()  { echo -e "${C_RED}[FATAL]${C_RESET} $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 参数与配置
# ---------------------------------------------------------------------------
VIDEO=""
VISUALIZE=0
for a in "$@"; do
  case "$a" in
    --visualize) VISUALIZE=1 ;;
    --no-visualize) VISUALIZE=0 ;;
    -*) die "未知参数: $a（仅支持 --visualize）" ;;
    *) VIDEO="$a" ;;
  esac
done
[ -n "$VIDEO" ] || die "用法: bash run_demo.sh <视频路径> [--visualize]"
[ -f "$VIDEO" ] || die "视频不存在: $VIDEO（Windows 的 D:\\x\\y.mp4 对应 WSL2 的 /mnt/d/x/y.mp4）"

CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="${WHAM_ENV_NAME:-wham}"
WHAM_DIR="${WHAM_DIR:-$HOME/wham/WHAM}"
OUTPUT_DIR="${WHAM_OUTPUT_DIR:-/mnt/d/wham-data/output}"

# 8GB 显存调优（效果优先可设 WHAM_FLIP_EVAL=1）
FLIP_EVAL="${WHAM_FLIP_EVAL:-0}"
SEQLEN="${WHAM_SEQLEN:-16}"          # 时序上下文窗口，越大越连贯但越吃显存
BATCH_SIZE=1

conda_run() { "$CONDA_DIR/bin/conda" run -n "$ENV_NAME" --no-capture-output "$@"; }

# ---------------------------------------------------------------------------
# 校验环境
# ---------------------------------------------------------------------------
log "校验环境"
[ -d "$WHAM_DIR" ] || die "WHAM 仓库不存在: $WHAM_DIR（先跑 setup_wsl2.sh）"
[ -s "$WHAM_DIR/checkpoints/wham_vit_bedlam_w_3dpw.pth.tar" ] || die "缺少默认权重 wham_vit_bedlam_w_3dpw.pth.tar"
conda_run python -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'" \
  || die "PyTorch/CUDA 校验失败"

# 防止 8GB 显存碎片化 OOM
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

# ---------------------------------------------------------------------------
# 生成 8GB 友好配置（备份原 demo.yaml，结束后恢复）
# ---------------------------------------------------------------------------
log "写入 8GB 友好配置（FLIP_EVAL=$FLIP_EVAL, SEQLEN=$SEQLEN, BATCH_SIZE=$BATCH_SIZE）"
YAML="$WHAM_DIR/configs/yamls/demo.yaml"
YAML_BAK="$WHAM_DIR/configs/yamls/demo.yaml.bak"
restore_yaml() { [ -f "$YAML_BAK" ] && mv -f "$YAML_BAK" "$YAML" || true; }
trap restore_yaml EXIT
[ -f "$YAML_BAK" ] || cp -f "$YAML" "$YAML_BAK"

cat > "$YAML" <<YML
LOGDIR: ''
DEVICE: 'cuda'
EXP_NAME: 'demo'
OUTPUT_DIR: 'experiments/'
NUM_WORKERS: 0
MODEL_CONFIG: 'configs/yamls/model_base.yaml'
FLIP_EVAL: $( [ "$FLIP_EVAL" = "1" ] && echo True || echo False )

TRAIN:
  STAGE: 'stage2'
  CHECKPOINT: 'checkpoints/wham_vit_bedlam_w_3dpw.pth.tar'
  BATCH_SIZE: $BATCH_SIZE

DATASET:
  SEQLEN: $SEQLEN

MODEL:
  BACKBONE: 'vit'
YML

# ---------------------------------------------------------------------------
# Stage 1：出 SMPL 结果 pkl（核心产物，不依赖 pyrender/OpenGL）
# ---------------------------------------------------------------------------
log "Stage 1/2：推理并保存 wham_output.pkl（estimate_local_only，跳过 SLAM）"
( cd "$WHAM_DIR" && conda_run python demo.py \
    --video "$VIDEO" \
    --estimate_local_only \
    --save_pkl \
    --output_pth "$OUTPUT_DIR" )
gate "SMPL 结果已保存: $OUTPUT_DIR/$(basename "$VIDEO" | sed 's/\.[^.]*$//')/wham_output.pkl"

# ---------------------------------------------------------------------------
# Stage 2（可选）：可视化 mp4（需 pyrender + OSMesa，失败不影响 pkl）
# ---------------------------------------------------------------------------
if [ "$VISUALIZE" = "1" ]; then
  log "Stage 2/2：可视化（复用缓存的 tracking 结果，无需重新预处理）"
  conda_run python -c "import pyrender" 2>/dev/null || \
    warn "未安装 pyrender，可视化将失败。安装: pip install pyrender trimesh && sudo apt-get install -y libosmesa6-dev libgl1-mesa-glx"
  ( cd "$WHAM_DIR" && conda_run python demo.py \
      --video "$VIDEO" \
      --estimate_local_only \
      --visualize \
      --output_pth "$OUTPUT_DIR" ) || warn "可视化失败（常见于无 OSMesa），wham_output.pkl 已正常产出，不影响后续管线"
fi

echo ""
gate "完成。产物位置:"
gate "  SMPL 逐帧结果: $OUTPUT_DIR/$(basename "$VIDEO" | sed 's/\.[^.]*$//')/wham_output.pkl"
[ "$VISUALIZE" = "1" ] && gate "  可视化视频   : $OUTPUT_DIR/$(basename "$VIDEO" | sed 's/\.[^.]*$//')/output.mp4"
