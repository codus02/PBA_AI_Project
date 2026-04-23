#!/usr/bin/env bash
# 3-model 순차 평가 — Gemma → Qwen → EXAONE.
# 각 모델은 독립 파이썬 프로세스라서 이전 모델 weights 가 GC 되면서 GPU 캐시도 해제됨 (OOM 회피).
#
# 사용:
#   nohup bash scripts/run_overnight_3models.sh > logs/overnight_$(date +%Y%m%d_%H%M).out 2>&1 &
#
# 저장:
#   eval_results/quantitative/{feedback,slots,rec,all}_{tag}_{stamp}.{json,txt}
#   eval_results/per_item/{feedback,slots,rec}_{tag}_{stamp}.csv     ← per-case hit 기록
#   logs/run_{tag}_{stamp}.log                                      ← stdout 그대로

set -uo pipefail
cd "$(dirname "$0")/.."

LIMIT="${LIMIT:-500}"
STAMP="$(date +%Y%m%d_%H%M)"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

run_one() {
    local tag="$1"
    local model="$2"
    local log="$LOG_DIR/run_${tag}_${STAMP}.log"

    echo "=========================================="
    echo " [$(date '+%F %T')] start  tag=$tag  model=$model  limit=$LIMIT"
    echo "=========================================="
    python -m scripts.eval_all_llm --tag "$tag" --model "$model" --limit "$LIMIT" \
        2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    echo "[$(date '+%F %T')] done   tag=$tag  rc=$rc"
    return "$rc"
}

# 1) Gemma
run_one "gemma_n${LIMIT}"  "google/gemma-2-9b-it"                || true

# 2) Qwen
run_one "qwen_n${LIMIT}"   "Qwen/Qwen3-8B"                       || true

# 3) EXAONE
run_one "exaone_n${LIMIT}" "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct" || true

echo ""
echo "=========================================="
echo " [$(date '+%F %T')] ALL DONE — see eval_results/quantitative + eval_results/per_item"
echo "=========================================="
