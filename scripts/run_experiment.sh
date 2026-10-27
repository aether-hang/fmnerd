#!/usr/bin/env bash
# author: aether
# Unified FM-NeRD launcher.
#
# Examples:
#   METHOD=TM bash scripts/run_experiment.sh
#   METHOD=DM DATASET=CIFAR100 IPC=1 bash scripts/run_experiment.sh
#   METHOD=TM DATASET=ImageNet SUBSET=nette bash scripts/run_experiment.sh
#   METHOD=TM DATASET=ImageNet SUBSET=woof EXTRA_ARGS="--vc_shared True"
#   DRY_RUN=1 METHOD=DC DATASET=CIFAR10 bash scripts/run_experiment.sh
#
# Everything is overridable through environment variables; the defaults below
# are the recommended starting points for IPC=1 and the listed datasets.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD="${METHOD:-TM}"
DATASET="${DATASET:-CIFAR10}"
SUBSET="${SUBSET:-nette}"
IPC="${IPC:-10}"
SEED="${SEED:-0}"
CUDA_ID="${CUDA_ID:-0}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-auto}"

case "$METHOD" in
  TM) DEFAULT_ITER=15000; DEFAULT_EVAL_IT=500;  DEFAULT_EPOCH=1000 ;;
  DM) DEFAULT_ITER=20000; DEFAULT_EVAL_IT=2000; DEFAULT_EPOCH=1000 ;;
  DC) DEFAULT_ITER=1000;  DEFAULT_EVAL_IT=500;  DEFAULT_EPOCH=300  ;;
  *) echo "METHOD must be TM, DM or DC (got: $METHOD)" >&2; exit 2 ;;
esac

# ---- subset / dataset defaults ------------------------------------------------
case "$SUBSET" in
  nette|imagenette) SUBSET_DIR="imagenette" ;;
  woof|imagewoof)   SUBSET_DIR="imagewoof" ;;
  *) SUBSET_DIR="$SUBSET" ;;
esac

DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
BUFFER_ROOT="${BUFFER_ROOT:-$ROOT/buffers}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs}"

case "$DATASET" in
  CIFAR10)
    RES="${RES:-32}"
    MODEL="${MODEL:-ConvNetD3}"
    DEFAULT_DATA="$DATA_ROOT"
    DEFAULT_MSZ=32
    DEFAULT_LAYER=16; DEFAULT_SHARED=1; DEFAULT_COND=3
    case "$IPC" in
      1) DEFAULT_MSZ=16; DEFAULT_LAYER=6; DEFAULT_SHARED=2; DEFAULT_COND=2 ;;
      2) DEFAULT_MSZ=32; DEFAULT_LAYER=8 ;;
      50) DEFAULT_MSZ=64 ;;
    esac
    DEFAULT_BATCH_SYN=$([ "$METHOD" = "TM" ] && echo 256 || echo 0)
    DEFAULT_BATCH_REAL=256
    DEFAULT_BATCH_TRAIN=256
    DEFAULT_STAGE1=2000
    ;;
  CIFAR100)
    RES="${RES:-32}"
    MODEL="${MODEL:-ConvNetD3}"
    DEFAULT_DATA="$DATA_ROOT"
    DEFAULT_MSZ=32
    DEFAULT_LAYER=16; DEFAULT_SHARED=1; DEFAULT_COND=3
    case "$IPC" in
      1) DEFAULT_MSZ=16; DEFAULT_LAYER=6; DEFAULT_SHARED=2; DEFAULT_COND=2 ;;
      2) DEFAULT_MSZ=32; DEFAULT_LAYER=8 ;;
      50) DEFAULT_MSZ=64 ;;
    esac
    # CIFAR-100 has 10x more classes than CIFAR-10; keep the MTT graph smaller.
    DEFAULT_BATCH_SYN=$([ "$METHOD" = "TM" ] && echo 128 || echo 0)
    DEFAULT_BATCH_REAL=256
    DEFAULT_BATCH_TRAIN=256
    DEFAULT_STAGE1=2000
    ;;
  ImageNet)
    RES="${RES:-128}"
    MODEL="${MODEL:-ConvNetD5}"
    DEFAULT_DATA="$DATA_ROOT/imagenet/$SUBSET_DIR"
    DEFAULT_MSZ=256
    DEFAULT_LAYER=16; DEFAULT_SHARED=1; DEFAULT_COND=3
    if [ "$IPC" -le 2 ]; then DEFAULT_MSZ=64; fi
    if [ "$IPC" -eq 1 ]; then DEFAULT_LAYER=20; DEFAULT_SHARED=2; DEFAULT_COND=2; fi
    if [ "$IPC" -ge 50 ]; then DEFAULT_MSZ=512; fi
    DEFAULT_BATCH_SYN=$([ "$METHOD" = "TM" ] && echo 102 || echo 8)
    DEFAULT_BATCH_REAL=64
    DEFAULT_BATCH_TRAIN=64
    DEFAULT_STAGE1=3000
    ;;
  *)
    echo "DATASET must be CIFAR10, CIFAR100 or ImageNet (got: $DATASET)" >&2
    exit 2
    ;;
esac

DATA_PATH="${DATA_PATH:-$DEFAULT_DATA}"
MSZ="${MSZ:-$DEFAULT_MSZ}"
LAYER_SIZE="${LAYER_SIZE:-$DEFAULT_LAYER}"
NUM_SHARED="${NUM_SHARED:-$DEFAULT_SHARED}"
NUM_COND="${NUM_COND:-$DEFAULT_COND}"
K="${K:-}"
BATCH_SYN="${BATCH_SYN:-$DEFAULT_BATCH_SYN}"
BATCH_REAL="${BATCH_REAL:-$DEFAULT_BATCH_REAL}"
BATCH_TRAIN="${BATCH_TRAIN:-$DEFAULT_BATCH_TRAIN}"
ITERATION="${ITERATION:-$DEFAULT_ITER}"
EVAL_IT="${EVAL_IT:-$DEFAULT_EVAL_IT}"
EPOCH_EVAL_TRAIN="${EPOCH_EVAL_TRAIN:-$DEFAULT_EPOCH}"
NUM_EVAL="${NUM_EVAL:-5}"
EVAL_MODE="${EVAL_MODE:-S}"
FLAG="${FLAG:-}"
DIPC="${DIPC:-0}"

STAGE1_STEPS="${STAGE1_STEPS:-$DEFAULT_STAGE1}"
STAGE1_LR="${STAGE1_LR:-5e-4}"
LR_CODE="${LR_CODE:-5e-4}"
if [ -n "${LR_SIREN:-}" ]; then
  :
elif [ "$METHOD" = "TM" ]; then
  LR_SIREN=1e-4
elif [ "$DATASET" = "ImageNet" ]; then
  LR_SIREN=5e-5
else
  LR_SIREN=1e-4
fi
LR_SIREN_LATE_SCALE="${LR_SIREN_LATE_SCALE:-0.1}"
LAMBDA_FREQ="${LAMBDA_FREQ:-0.1}"
LAMBDA_DIV="${LAMBDA_DIV:-1e-3}"
FREQ_WARMUP="${FREQ_WARMUP:-1500}"
USE_SPATIAL_BASIS="${USE_SPATIAL_BASIS:-1}"
BASIS_WIDTH="${BASIS_WIDTH:-8}"
STAGE1_TARGET="${STAGE1_TARGET:-spatial_avg}"
STAGE1_SAMPLE_SIZE="${STAGE1_SAMPLE_SIZE:-8}"
USE_ANCHOR_LOSS="${USE_ANCHOR_LOSS:-1}"
LAMBDA_ANCHOR_START="${LAMBDA_ANCHOR_START:-1e-2}"
LAMBDA_ANCHOR_TARGET="${LAMBDA_ANCHOR_TARGET:-1e-4}"
ANCHOR_DECAY_STEPS="${ANCHOR_DECAY_STEPS:-3000}"
CODE_INIT_SCALE="${CODE_INIT_SCALE:-1.0}"
MODULATION_CODE_SCALE="${MODULATION_CODE_SCALE:-}"
SIREN_INIT="${SIREN_INIT:-ddif}"
CODE_MODE="${CODE_MODE:-residual}"
CODE_STD_FLOOR="${CODE_STD_FLOOR:-1e-3}"

# Ablation switches; 1/true keeps the module, 0/false disables it.
to_bool() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y) echo "True" ;;
    0|false|no|n) echo "False" ;;
    *) echo "$1" ;;
  esac
}
USE_FREQ_LOSS="$(to_bool "${USE_FREQ_LOSS:-1}")"
USE_DIV_LOSS="$(to_bool "${USE_DIV_LOSS:-1}")"
USE_SPATIAL_BASIS="$(to_bool "$USE_SPATIAL_BASIS")"
USE_ANCHOR_LOSS="$(to_bool "$USE_ANCHOR_LOSS")"
STAGE1_ONLY="$(to_bool "${STAGE1_ONLY:-0}")"
VC_SHARED="$(to_bool "${VC_SHARED:-0}")"

# Resolve relative paths against the project root, not the method directory.
abspath() {
  case "$1" in
    /*) echo "$1" ;;
    *) echo "$ROOT/$1" ;;
  esac
}
# Resolve a relative interpreter path before cd-ing into a method folder.
case "$PYTHON" in
  */*) PYTHON="$(abspath "$PYTHON")" ;;
esac
DATA_PATH="$(abspath "$DATA_PATH")"
BUFFER_ROOT="$(abspath "$BUFFER_ROOT")"
LOG_ROOT="$(abspath "$LOG_ROOT")"
mkdir -p "$LOG_ROOT"

ARGS=(
  --dataset "$DATASET" --model "$MODEL" --ipc "$IPC" --res "$RES"
  --eval_mode "$EVAL_MODE" --device "$DEVICE"
  --data_path "$DATA_PATH" --save_path "$LOG_ROOT"
  --batch_real "$BATCH_REAL" --batch_train "$BATCH_TRAIN" --batch_syn "$BATCH_SYN"
  --Iteration "$ITERATION" --eval_it "$EVAL_IT"
  --epoch_eval_train "$EPOCH_EVAL_TRAIN" --num_eval "$NUM_EVAL"
  --seed "$SEED" --dipc "$DIPC"
  --msz_per_channel "$MSZ"
  --num_shared_layers "$NUM_SHARED" --num_cond_layers "$NUM_COND"
  --stage1_steps "$STAGE1_STEPS" --stage1_lr "$STAGE1_LR"
  --lr_code "$LR_CODE" --lr_siren "$LR_SIREN"
  --lr_siren_late_scale "$LR_SIREN_LATE_SCALE"
  --lambda_freq_target "$LAMBDA_FREQ" --freq_warmup "$FREQ_WARMUP"
  --lambda_div "$LAMBDA_DIV"
  --use_spatial_basis "$USE_SPATIAL_BASIS" --basis_width "$BASIS_WIDTH"
  --stage1_target "$STAGE1_TARGET" --stage1_sample_size "$STAGE1_SAMPLE_SIZE"
  --use_anchor_loss "$USE_ANCHOR_LOSS"
  --lambda_anchor_start "$LAMBDA_ANCHOR_START"
  --lambda_anchor_target "$LAMBDA_ANCHOR_TARGET"
  --anchor_decay_steps "$ANCHOR_DECAY_STEPS"
  --code_init_scale "$CODE_INIT_SCALE"
  --siren_init "$SIREN_INIT"
  --code_mode "$CODE_MODE" --code_std_floor "$CODE_STD_FLOOR"
  --use_freq_loss "$USE_FREQ_LOSS" --use_div_loss "$USE_DIV_LOSS"
  --stage1_only "$STAGE1_ONLY" --vc_shared "$VC_SHARED"
)

if [ -n "$MODULATION_CODE_SCALE" ]; then
  ARGS+=(--modulation_code_scale "$MODULATION_CODE_SCALE")
fi
if [ -n "${LAYER_SIZE:-}" ]; then
  ARGS+=(--layer_size "$LAYER_SIZE")
fi
if [ -n "$FLAG" ]; then
  ARGS+=(--FLAG "$FLAG")
fi
if [ "$DATASET" = "ImageNet" ]; then
  ARGS+=(--subset "$SUBSET_DIR")
fi
if [ -n "$K" ]; then
  ARGS+=(--k "$K")
fi
if [ "$METHOD" = "TM" ]; then
  ARGS+=(--buffer_path "$BUFFER_ROOT")
  [ -n "${SYN_STEPS:-}" ] && ARGS+=(--syn_steps "$SYN_STEPS")
  [ -n "${EXPERT_EPOCHS:-}" ] && ARGS+=(--expert_epochs "$EXPERT_EPOCHS")
  [ -n "${MAX_START_EPOCH:-}" ] && ARGS+=(--max_start_epoch "$MAX_START_EPOCH")
  [ -n "${LR_TEACHER:-}" ] && ARGS+=(--lr_teacher "$LR_TEACHER")
  [ -n "${LR_LR:-}" ] && ARGS+=(--lr_lr "$LR_LR")
else
  ARGS+=(--num_exp "${NUM_EXP:-1}" --lr_net "${LR_NET:-0.01}")
fi
[ "${FORCE_SAVE:-0}" = "1" ] && ARGS+=(--force_save)
[ "${DETECT_ANOMALY:-0}" = "1" ] && ARGS+=(--detect_anomaly True)

EXTRA_ARGS="${EXTRA_ARGS:-}"
K_EFFECTIVE=$(( 3 * MSZ ))
if [ -n "$K" ]; then K_EFFECTIVE="$K"; fi
DATASET_DISPLAY="$DATASET"
if [ "$DATASET" = "ImageNet" ]; then DATASET_DISPLAY="$DATASET/$SUBSET_DIR"; fi

echo "============================================================"
echo " FM-NeRD"
echo "   method      : $METHOD"
echo "   dataset     : $DATASET_DISPLAY"
echo "   model/res   : $MODEL / $RES"
echo "   ipc         : $IPC   msz/channel: $MSZ   k: $K_EFFECTIVE"
echo "   data        : $DATA_PATH"
if [ "$METHOD" = "TM" ]; then echo "   buffers     : $BUFFER_ROOT"; fi
echo "   logs        : $LOG_ROOT"
echo "   stage1      : $STAGE1_STEPS steps @ $STAGE1_LR"
echo "   iterations  : $ITERATION (eval every $EVAL_IT)"
echo "   batch_syn   : $BATCH_SYN"
echo "   aux flags   : freq=$USE_FREQ_LOSS div=$USE_DIV_LOSS stage1_only=$STAGE1_ONLY vc_shared=$VC_SHARED"
echo "============================================================"

CMD=("$PYTHON" "main_${METHOD}_FMNeRD.py" "${ARGS[@]}")
if [ -n "$EXTRA_ARGS" ]; then
  # shellcheck disable=SC2206
  CMD+=($EXTRA_ARGS)
fi
CMD+=("$@")

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf 'cd %q && CUDA_VISIBLE_DEVICES=%q ' "$ROOT/$METHOD" "$CUDA_ID"
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

cd "$ROOT/$METHOD"
CUDA_VISIBLE_DEVICES="$CUDA_ID" "${CMD[@]}"
