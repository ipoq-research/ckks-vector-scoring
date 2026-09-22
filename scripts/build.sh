#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "$0")/.."
IPOQ_PYTHON="${IPOQ_PYTHON:-python3}"
"$IPOQ_PYTHON" -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12 or later required"; import numpy, seal'
command -v c++ >/dev/null
"$IPOQ_PYTHON" engine/build_native.py --hexl
"$IPOQ_PYTHON" engine/build_phase.py
"$IPOQ_PYTHON" engine/build_fresh.py
if [[ "${1:-}" == "--cuda" ]]; then
  "$IPOQ_PYTHON" engine/build_fresh.py --cuda
elif [[ -n "${1:-}" ]]; then
  echo 'Usage: bash scripts/build.sh [--cuda]' >&2
  exit 2
fi
