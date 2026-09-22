#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "$0")/.."
IPOQ_PYTHON="${IPOQ_PYTHON:-python3}"
"$IPOQ_PYTHON" -c 'assert __debug__, "Do not use Python -O or PYTHONOPTIMIZE"'
"$IPOQ_PYTHON" engine/verify_import.py
if "$IPOQ_PYTHON" -c 'from pathlib import Path; flags=Path("/proc/cpuinfo").read_text(); raise SystemExit(0 if all(x in flags for x in ("avx512f", "avx512dq", "avx512ifma")) else 1)'; then
  "$IPOQ_PYTHON" engine/verify_crt52.py
else
  echo 'SKIP: exact CRT52 SIMD tests require AVX-512 F/DQ/IFMA. CPU fallback remains tested by verify_import.py.'
fi
"$IPOQ_PYTHON" scripts/audit_evidence.py
"$IPOQ_PYTHON" -m unittest discover -s tests -v
