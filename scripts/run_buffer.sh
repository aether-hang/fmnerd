#!/usr/bin/env bash
# Build DDiF/MTT expert trajectories for TM.
#
# Usage:
#   bash scripts/run_buffer.sh                      # CIFAR-10
#   DATASET=CIFAR100 bash scripts/run_buffer.sh
#   DATASET=ImageNet SUBSET=nette bash scripts/run_buffer.sh
#   DATASET=ImageNet SUBSET=woof NUM_EXPERTS=100 TRAIN_EPOCHS=50 bash scripts/run_buffer.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${DATASET:-CIFAR10}"
SUBSET="${SUBSET:-nette}"
CUDA_ID="${CUDA_ID:-0}"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
BUFFER_ROOT="${BUFFER_ROOT:-$ROOT/buffers}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-50}"
NUM_EXPERTS="${NUM_EXPERTS:-100}"
SEED="${SEED:-0}"

case "$SUBSET" in
  nette|imagenette) SUBSET_DIR="imagenette" ;;
  woof|imagewoof)   SUBSET_DIR="imagewoof" ;;
  *) SUBSET_DIR="$SUBSET" ;;
esac

case "$DATASET" in
  CIFAR10|CIFAR100)
    RES="${RES:-32}"
    MODEL="${MODEL:-ConvNetD3}"
    DATA_PATH="${DATA_PATH:-$DATA_ROOT}"
    DEFAULT_BATCH=256
    ;;
  ImageNet)
    RES="${RES:-128}"
    MODEL="${MODEL:-ConvNetD5}"
    DATA_PATH="${DATA_PATH:-$DATA_ROOT/imagenet/$SUBSET_DIR}"
    DEFAULT_BATCH=64
    ;;
  *)
    echo "DATASET must be CIFAR10, CIFAR100 or ImageNet (got: $DATASET)" >&2
    exit 2
    ;;
esac

BATCH_TRAIN="${BATCH_TRAIN:-$DEFAULT_BATCH}"
BATCH_REAL="${BATCH_REAL:-$DEFAULT_BATCH}"
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
mkdir -p "$BUFFER_ROOT"

echo "============================================================"
echo " DDiF expert buffer"
echo "   dataset : $DATASET $([ "$DATASET" = ImageNet ] && echo "/$SUBSET_DIR")"
echo "   model   : $MODEL  res=$RES"
echo "   data    : $DATA_PATH"
echo "   buffer  : $BUFFER_ROOT"
echo "   experts : $NUM_EXPERTS x $TRAIN_EPOCHS epochs"
echo "============================================================"

CMD=(
  "$PYTHON" buffer.py
  --dataset "$DATASET" --subset "$SUBSET_DIR" --res "$RES"
  --model "$MODEL"
  --data_path "$DATA_PATH" --buffer_path "$BUFFER_ROOT"
  --train_epochs "$TRAIN_EPOCHS" --num_experts "$NUM_EXPERTS"
  --batch_train "$BATCH_TRAIN" --batch_real "$BATCH_REAL"
  --seed "$SEED"
)
if [ "${DRY_RUN:-0}" = "1" ]; then
  printf 'cd %q && CUDA_VISIBLE_DEVICES=%q ' "$ROOT/TM" "$CUDA_ID"
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi
cd "$ROOT/TM"
CUDA_VISIBLE_DEVICES="$CUDA_ID" "${CMD[@]}"
