#!/bin/bash

#SBATCH --job-name=context_switching
#SBATCH --output=/home/%u/logs/sbatch/context_switching_%A_%a.out
#SBATCH --error=/home/%u/logs/sbatch/context_switching_%A_%a.err
#SBATCH --time=2-0:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1              
#SBATCH --constrain=A6000|L40|L40S
#SBATCH --partition=preempt
#SBATCH --mail-user=ml6@andrew.cmu.edu
#SBATCH --mail-type=START,END,FAIL
#SBATCH --array=0-17%8

export HF_HOME=/data/user_data/ml6/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HUB_OFFLINE=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

mkdir -p /home/ml6/logs/sbatch
mkdir -p "$HF_HOME"

export CUDA_VISIBLE_DEVICES=0

eval "$(conda shell.bash hook)"
conda activate context_switch

set -e
cd /home/ml6/context_switching_analysis

MODELS=(
    "mistralai/Mistral-7B-Instruct-v0.1:quantize"
    "meta-llama/Llama-2-7b-chat-hf:quantize"
)

TASKS=(
    "mmlu"
    "rotten_tomatoes"
    "tweetqa"
)

ALL_JOBS=()
for model_info in "${MODELS[@]}"; do
    for target in "${TASKS[@]}"; do
        for distractor in "${TASKS[@]}"; do
            ALL_JOBS+=("$model_info;$target;$distractor")
        done
    done
done

JOB_CONFIG=${ALL_JOBS[$SLURM_ARRAY_TASK_ID]}

IFS=';' read -r MODEL_INFO TARGET DISTRACTOR <<< "$JOB_CONFIG"

MODEL=$(echo "$MODEL_INFO" | cut -d':' -f1)
QUANTIZE_INFO=$(echo "$MODEL_INFO" | cut -d':' -f2)

QUANTIZE_FLAG=""
if [ "$QUANTIZE_INFO" == "quantize" ]; then
    QUANTIZE_FLAG="--quantize"
fi

echo "=== Job $SLURM_ARRAY_TASK_ID: model=$MODEL target=$TARGET distractor=$DISTRACTOR quantize=$QUANTIZE_INFO ==="
python experiment.py \
    --model "$MODEL" \
    --target "$TARGET" \
    --distractor "$DISTRACTOR" \
    --max_len 6 \
    $QUANTIZE_FLAG
