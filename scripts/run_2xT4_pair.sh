#!/usr/bin/env bash
# Run TM on GPU 0 and DM on GPU 1 concurrently for one dataset (2xT4 16GB).
# DC can be run afterwards with RUN_DC=1 (or launch it manually on either GPU).
#
# Usage:
#   bash scripts/run_2xT4_pair.sh CIFAR10
#   bash scripts/run_2xT4_pair.sh CIFAR100
#   bash scripts/run_2xT4_pair.sh ImageNet nette
#   bash scripts/run_2xT4_pair.sh ImageNet woof
#   RUN_DC=1 bash scripts/run_2xT4_pair.sh CIFAR10
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-CIFAR10}"
SUBSET="${2:-nette}"

case "$DATASET" in
  CIFAR10)
    TM_SCRIPT="$ROOT/TM/scripts/run_TM_FMNeRD_CIFAR10.sh"
    DM_SCRIPT="$ROOT/DM/scripts/run_DM_FMNeRD_CIFAR10.sh"
    DC_SCRIPT="$ROOT/DC/scripts/run_DC_FMNeRD_CIFAR10.sh"
    ;;
  CIFAR100)
    TM_SCRIPT="$ROOT/TM/scripts/run_TM_FMNeRD_CIFAR100.sh"
    DM_SCRIPT="$ROOT/DM/scripts/run_DM_FMNeRD_CIFAR100.sh"
    DC_SCRIPT="$ROOT/DC/scripts/run_DC_FMNeRD_CIFAR100.sh"
    ;;
  ImageNet)
    case "$SUBSET" in
      nette|imagenette)
        TM_SCRIPT="$ROOT/TM/scripts/run_TM_FMNeRD_ImageNet_nette.sh"
        DM_SCRIPT="$ROOT/DM/scripts/run_DM_FMNeRD_ImageNet_nette.sh"
        DC_SCRIPT="$ROOT/DC/scripts/run_DC_FMNeRD_ImageNet_nette.sh"
        ;;
      woof|imagewoof)
        TM_SCRIPT="$ROOT/TM/scripts/run_TM_FMNeRD_ImageNet_woof.sh"
        DM_SCRIPT="$ROOT/DM/scripts/run_DM_FMNeRD_ImageNet_woof.sh"
        DC_SCRIPT="$ROOT/DC/scripts/run_DC_FMNeRD_ImageNet_woof.sh"
        ;;
      *) echo "SUBSET must be nette or woof" >&2; exit 2 ;;
    esac
    ;;
  *) echo "DATASET must be CIFAR10, CIFAR100 or ImageNet" >&2; exit 2 ;;
esac

echo "[2xT4] TM  -> GPU 0 : $TM_SCRIPT"
echo "[2xT4] DM  -> GPU 1 : $DM_SCRIPT"

CUDA_ID=0 bash "$TM_SCRIPT" &
TM_PID=$!
CUDA_ID=1 bash "$DM_SCRIPT" &
DM_PID=$!

status=0
wait "$TM_PID" || status=1
wait "$DM_PID" || status=1
if [ "$status" -ne 0 ]; then
  echo "[2xT4] at least one job failed; check logs/." >&2
  exit "$status"
fi

if [ "${RUN_DC:-0}" = "1" ]; then
  echo "[2xT4] DC  -> GPU 0 : $DC_SCRIPT"
  CUDA_ID=0 bash "$DC_SCRIPT"
fi
