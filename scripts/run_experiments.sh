#!/bin/bash
set -e

MODELS=(
# "EleutherAI/pythia-70m"
# "mistralai/Mistral-7B-Instruct-v0.1"
# "meta-llama/Llama-2-7b-chat-hf"
"Qwen/Qwen2.5-1.5B"
)

TASKS=(
    # "tweet_qa"
    "mmlu"
    "rotten_tomatoes"
)

for MODEL in "${MODELS[@]}"; do
  for TARGET in "${TASKS[@]}"; do
    echo "=== RUNNING CONTROL: MODEL=$MODEL TARGET=$TARGET DISTRACTOR=$TARGET ==="
    python experiment.py --model "$MODEL" --target "$TARGET" --distractor "$TARGET" --max_len 6 --n 50

    for DIST in "${TASKS[@]}"; do
      if [ "$TARGET" != "$DIST" ]; then
        echo "=== RUNNING CONTEXT-SWITCH: MODEL=$MODEL TARGET=$TARGET DISTRACTOR=$DIST ==="
        python experiment.py --model "$MODEL" --target "$TARGET" --distractor "$DIST" --max_len 6 --n 50
      fi
    done
  done
done
python plot.py --result results --out_dir plots