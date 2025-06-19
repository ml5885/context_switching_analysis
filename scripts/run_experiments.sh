#!/bin/bash
set -e
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

MODELS=(
# "EleutherAI/pythia-70m:no-quantize"
# "mistralai/Mistral-7B-Instruct-v0.1:quantize"
# "meta-llama/Llama-2-7b-chat-hf:quantize"
"Qwen/Qwen2.5-0.5B:no-quantize"
)

TASKS=(
    "mmlu"
    "rotten_tomatoes"
    "tweet_qa"
)

for MODEL_INFO in "${MODELS[@]}"; do
  MODEL=$(echo "$MODEL_INFO" | cut -d':' -f1)
  QUANTIZE_INFO=$(echo "$MODEL_INFO" | cut -d':' -f2)

  QUANTIZE_FLAG=""
  if [ "$QUANTIZE_INFO" == "quantize" ]; then
    QUANTIZE_FLAG="--quantize"
  fi

  for TARGET in "${TASKS[@]}"; do
    echo "=== RUNNING CONTROL: MODEL=$MODEL TARGET=$TARGET DISTRACTOR=$TARGET ==="
    python experiment.py --model "$MODEL" --target "$TARGET" --distractor "$TARGET" --max_len 6 $QUANTIZE_FLAG

    for DIST in "${TASKS[@]}"; do
      if [ "$TARGET" != "$DIST" ]; then
        echo "=== RUNNING CONTEXT-SWITCH: MODEL=$MODEL TARGET=$TARGET DISTRACTOR=$DIST ==="
        python experiment.py --model "$MODEL" --target "$TARGET" --distractor "$DIST" --max_len 6 $QUANTIZE_FLAG
      fi
    done
  done
done
python plot.py --result results --out_dir plots