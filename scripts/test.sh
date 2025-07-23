#!/usr/bin/env bash
# scripts/run_experiments.sh

set -euo pipefail

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
MAX_LEN=8
OUT_DIR="test_results"
BATCH_SIZE=4
TASKS=(
  "mmlu"
  "rotten_tomatoes"
  "tweetqa"
)

mkdir -p "${OUT_DIR}"

for target in "${TASKS[@]}"; do
  for distractor in "${TASKS[@]}"; do
    echo "Running target=${target}  |  distractor=${distractor}"
    python experiment.py \
      --model "${MODEL}" \
      --target "${target}" \
      --distractor "${distractor}" \
      --max_len "${MAX_LEN}" \
      --batch_size "${BATCH_SIZE}" \
      --fp16 \
      --out_dir "${OUT_DIR}"
  done
done

echo "Finished. JSON outputs are in ${OUT_DIR}/"
