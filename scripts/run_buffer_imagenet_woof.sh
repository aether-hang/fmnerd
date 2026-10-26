#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET=ImageNet SUBSET=woof exec bash "$HERE/run_buffer.sh" "$@"
