import sys
from experiment import experiment

if __name__ == "__main__":
    # Use a small model and dataset for quick debugging
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    target = "rotten_tomatoes/validation"
    distractor = "rotten_tomatoes/validation"
    max_len = 1
    batch_size = 2
    fp16 = False
    no_cosine = True

    print("[TEST DEBUG] Running minimal experiment...")
    metric_by_len, cos_by_len, metric, dbg_top5, debug_examples = experiment(
        model_name, target, distractor, max_len,
        batch_size=batch_size, fp16=fp16, no_cosine=no_cosine
    )
    print("[TEST DEBUG] metric_by_len:", metric_by_len)
    print("[TEST DEBUG] cos_by_len:", cos_by_len)
    print("[TEST DEBUG] metric:", metric)
    print("[TEST DEBUG] dbg_top5:", dbg_top5)
    print("[TEST DEBUG] debug_examples:", debug_examples[:2])
