import os
from experiment import experiment, save_results, save_debug_log

DATASETS = [
    "mmlu/validation",
    "rotten_tomatoes/validation",
    "tweetqa/validation"
]
MODEL = "Qwen/Qwen2.5-0.5B"
MAX_LEN = 2
DEVICE = None
OUT_DIR = "test_results"
QUANTIZE = False
BATCH_SIZE = 8

os.makedirs(OUT_DIR, exist_ok=True)

# Load model only once inside experiment
for target in DATASETS:
    for distractor in DATASETS:
        print(f"Running target={target} vs distractor={distractor}")
        metric_by_len, cos_by_len, metric_name, dbg, debug_log = experiment(
            MODEL,
            target,
            distractor,
            MAX_LEN,
            DEVICE,
            quantize=QUANTIZE,
            batch_size=BATCH_SIZE,
        )
        save_results(
            MODEL,
            target,
            distractor,
            metric_name,
            metric_by_len,
            cos_by_len,
            dbg,
            OUT_DIR,
        )
        save_debug_log(
            MODEL,
            target,
            distractor,
            debug_log,
            OUT_DIR,
        )

print(f"finished; results stored in {OUT_DIR}")
