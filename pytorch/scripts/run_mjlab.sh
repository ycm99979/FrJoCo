#!/bin/bash
# G1 WBC — mjlab 실행 래퍼
#
# 사용법 (프로젝트 루트에서):
#   bash pytorch/scripts/run_mjlab.sh                        # 기본 (IK 모드)
#   bash pytorch/scripts/run_mjlab.sh --mode dbfc            # DBFC 모드
#   bash pytorch/scripts/run_mjlab.sh --num_envs 1 --mode ik
#
# 전제: conda activate frjoco (Python 3.13 + CUDA 12.6)

set -e

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROJECT_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${_PROJECT_ROOT}:${PYTHONPATH}"

echo "[INFO] python: $(which python)"
echo "[INFO] PROJECT_ROOT: ${_PROJECT_ROOT}"

exec python -m pytorch.scripts.run_mjlab "$@"
