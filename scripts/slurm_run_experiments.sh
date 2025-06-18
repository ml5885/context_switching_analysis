#!/bin/bash
#SBATCH --job-name=context_switching
#SBATCH --output=/home/%u/logs/sbatch/context_switching_%j.out
#SBATCH --error=/home/%u/logs/sbatch/context_switching_%j.err
#SBATCH --partition=general
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G

export HF_HOME=/data/user_data/ml6/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HUB_OFFLINE=0

mkdir -p /home/ml6/logs/sbatch
mkdir -p "$HF_HOME"

export CUDA_VISIBLE_DEVICES=0

eval "$(conda shell.bash hook)"
conda activate context_switch

set -e
cd /home/ml6/context_switching_analysis

MODELS=(
    "mistralai/Mistral-7B-Instruct-v0.1"
    "meta-llama/Llama-2-7b-chat-hf"
)

TASKS=(
    "mmlu"
    "rotten_tomatoes"
    "tweetqa"
)

for MODEL in "${MODELS[@]}"; do
    for TARGET in "${TASKS[@]}"; do
        for DIST in "${TASKS[@]}"; do
            if [ "$TARGET" != "$DIST" ]; then
                echo "=== model=$MODEL  target=$TARGET  distractor=$DIST ==="
                python experiment.py \
                    --model "$MODEL" \
                    --target "$TARGET" \
                    --distractor "$DIST" \
                    --max_len 6 \
                    --n 100
            fi
        done
    done
done

echo "Running plot.py to aggregate results..."
python plot.py --result results --out_dir plots

echo "Done. All results are in results/ and plots/."
