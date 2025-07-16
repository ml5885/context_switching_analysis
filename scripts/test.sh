#!/usr/bin/env bash
# scripts/run_experiments.sh
#
# Run the context‑switch experiments with history length = 2
# for all three benchmark tasks as both target and distractor.
# Results are stored in ./test_results/

set -euo pipefail

MODEL="Qwen/Qwen2.5-0.5B"
MAX_LEN=2
OUT_DIR="test_results"
BATCH_SIZE=32
TASKS=(
  "mmlu/validation"
  "rotten_tomatoes/validation"
  "tweetqa/validation"
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
    #   --fp16 \
      --no_cosine \
      --out_dir "${OUT_DIR}"
  done
done

echo "Finished. JSON outputs are in ${OUT_DIR}/"
