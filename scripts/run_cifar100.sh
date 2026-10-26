#!/usr/bin/env bash
# Usage: bash scripts/run_cifar100.sh [TM|DM|DC] [extra FM-NeRD args...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD="${1:-TM}"
if [ "$#" -gt 0 ]; then shift; fi
DATASET=CIFAR100 METHOD="$METHOD" exec bash "$HERE/run_experiment.sh" "$@"
