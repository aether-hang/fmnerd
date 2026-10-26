#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET=CIFAR100 exec bash "$HERE/run_buffer.sh" "$@"
