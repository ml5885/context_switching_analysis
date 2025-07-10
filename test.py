import os
import subprocess

DATASETS = ["mmlu/validation", "rotten_tomatoes/validation", "tweetqa/validation"]

MODEL = "Qwen/Qwen2.5-0.5B"
MAX_LEN = 2
DEVICE = None
OUT_DIR = "test_results"
QUANTIZE = False

os.makedirs(OUT_DIR, exist_ok=True)

for task in DATASETS:
    cmd = [
        "python", "experiment.py",
        "--model", MODEL,
        "--target", task,
        "--distractor", task,
        "--max_len", str(MAX_LEN),
        "--out_dir", OUT_DIR,
    ]
    if DEVICE:
        cmd.extend(["--device", DEVICE])
    if QUANTIZE:
        cmd.append("--quantize")
    subprocess.run(cmd, check=True)

print("finished; results stored in", OUT_DIR)
