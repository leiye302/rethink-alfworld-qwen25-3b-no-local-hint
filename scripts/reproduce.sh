#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
if [[ $# -eq 0 || "$1" == --help ]]; then
  echo 'Usage: bash scripts/reproduce.sh --check | --run [--resume]'
  echo 'Shared full Route-SFT2000 -> no-hint GRPO+OPSD150 -> fixed seen140.'
  echo 'No API or additional baseline is launched.'
  exit 0
fi
if [[ -f .env ]]; then set -a; source .env; set +a; fi
unset RETHINK_API_KEY RETHINK_API_ENDPOINT RETHINK_API_WORKER RETHINK_ROUTER_WORKER
if [[ "$1" == --check && $# -eq 1 ]]; then
  exec "${SFT_PYTHON:-python}" src/tools/check.py
fi
if [[ "$1" != --run ]]; then
  echo 'GPU training requires --run.' >&2; exit 2
fi
: "${ALFWORLD_DATA:?Set the ALFWorld asset directory}"
"${SFT_PYTHON:-python}" src/tools/check.py --assets
shift
exec "${SFT_PYTHON:-python}" src/tools/run_pair.py "$@"
