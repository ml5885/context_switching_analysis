# Context Switching Analysis

Characterize how LLMs perform target tasks under distractor histories (context switching) and inspect layer-wise cosine similarities.
This builds on existing work by [Gupta et al.](https://arxiv.org/abs/2402.18216)

- Core: experiment.py, model.py, data.py
- Visualization: viz.py (+ scratch/run_viz.sh)
- Scripts: scripts/test.sh (local), test_debug.py (quick check)

## Setup

Requirements:

- Python 3.9+ (GPU recommended)

Install:

```bash
pip install torch transformers datasets evaluate matplotlib pandas tqdm numpy
```

Optional (Conda):

```bash
conda create -n context_switching python=3.10 -y
conda activate context_switching
```

Optional (Hugging Face caches on clusters):

```bash
export HF_HOME=/data/user_data/<user>/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
```

## Quickstart

Fast debug on CPU/GPU with a small model:

```bash
python test_debug.py
```

This runs Qwen/Qwen2.5-0.5B-Instruct on rotten_tomatoes with tiny settings.

## Running Experiments

Direct CLI:

```bash
python experiment.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --target mmlu/validation \
  --distractor tweetqa/validation \
  --max_len 8 \
  --batch_size 4 \
  --fp16 \
  --out_dir results
```

Local batch:

```bash
bash scripts/test.sh
```

## Results and Logs

Main JSON per (target, distractor):

```json
{
  "model": "...",
  "target": "...",
  "distractor": "...",
  "metric_name": "accuracy | rouge",
  "metric_by_len": {"0": ..., "1": ...},
  "cos_by_len": {"0": [...], "1": [...]},
  "debug_top5_by_len": {"0": [["token", count], ...], "...": ...}
}
```

Debug log:

```json
{
  "debug_examples": [
    {
      "prompt_text": "...",
      "model_prediction": "...",
      "expected_answer": "...",
      "config": {"history_len": h, "target_idx": i, "...": "..."}
    }
  ]
}
```

## Visualization

Plot metrics, percent change, and cosine:

```bash
python viz.py metric --result_dir test_results/meta-llama@Llama-2-7b-chat-hf__mmlu --out_dir plots_mmlu
```

Verify accuracy from debug logs:

```bash
python viz.py verify --debug_file test_results/..._debug.json
```

## Notes

- Llama-2 weights may require accepting a license on Hugging Face.
- Use smaller models (e.g., Qwen/Qwen2.5-0.5B-Instruct) for quick iteration.
- Recommended GPU setting:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
