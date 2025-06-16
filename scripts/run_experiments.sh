#!/bin/bash
set -e

MODELS=(
"EleutherAI/pythia-70m"
# "Qwen/Qwen2.5-0.5B"
# "meta-llama/Meta-Llama-3-8B"
)

TASKS=(
    "mmlu"
    "rotten_tomatoes"
    "cnn_dailymail"
)

for MODEL in "${MODELS[@]}"; do
  for TARGET in "${TASKS[@]}"; do
    for DIST in "${TASKS[@]}"; do
      if [ "$TARGET" != "$DIST" ]; then
        python experiment.py --model "$MODEL" --target "$TARGET" --distractor "$DIST" --max_len 10 --n 1
      fi
    done
  done
done
python plot.py --result results --out_dir plots
