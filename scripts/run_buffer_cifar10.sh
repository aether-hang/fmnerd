#!/usr/bin/env bash
# author: aether
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET=CIFAR10 exec bash "$HERE/run_buffer.sh" "$@"
