import evaluate
from transformer_lens import HookedTransformer
from data_utils import load_split, build_prompt, dataset_config
from experiment import build_history_text
import tempfile
import json
import os
from pandas_helpers import df_from_dir
from plot import plot_metric, plot_pct_change, plot_cosine

print("Testing data loading and prompt building for all configured datasets...")

for name in dataset_config.keys():
    print(f"\n--- Testing dataset: {name} ---")
    ds = load_split(name, streaming=False)
    sample = ds[0]
    prompt, answer = build_prompt(name, sample)
    print(f"Sample prompt:\n{prompt}")
    print(f"Sample answer: {answer}")

    cfg = dataset_config[name]
    metric_name = cfg["metric"]
    print(f"Testing metric: {metric_name}")

    if metric_name == "accuracy":
        print("Accuracy is trivial to compute, so skipping")

    elif metric_name == "rouge":
        rouge_eval = evaluate.load("rouge")
        preds = [
            "hello there world",
            "general kenobi",
            "this is a test",
            "another one",
            "foo bar",
        ]
        refs = [
            "hello there",
            "general kenobi",
            "this is a test sentence",
            "another one bites the dust",
            "foo bar baz",
        ]
        rouge_res = rouge_eval.compute(predictions=preds, references=refs)
        rl = rouge_res["rougeL"]
        score = rl["fmeasure"] if isinstance(rl, dict) else float(rl)
        print(f"ROUGE-L F1 test: {score}")

print("\n--- Testing conversation history building ---")

max_len = 2
target_task = "mmlu"
distractor_task = "rotten_tomatoes"

print(f"Target: {target_task}, Distractor: {distractor_task}, max_len: {max_len}")

target_ds = list(load_split(target_task, streaming=False))[:10]
distractor_ds = list(load_split(distractor_task, streaming=False))[:10]

for h in range(max_len + 1):
    print(f"\n--- History length: {h} ---")

    history_text = build_history_text(distractor_task, distractor_ds, 0, h)
    turns = history_text.split("\n\n") if h > 0 else []

    final_p, gold = build_prompt(target_task, target_ds[0])

    tgt_cfg = dataset_config[target_task]
    assistant_prompt = "Assistant:"
    if tgt_cfg["answer_suffix"]:
        assistant_prompt += " <Answer>"

    conv = (
        (history_text + "\n\n") if history_text else ""
    ) + f"User: {final_p}\n{assistant_prompt}"

    print(f"History contains {len(turns)} turn(s) from '{distractor_task}'.")
    if h > 0:
        print("History text:\n", history_text)
    print("Final prompt:\n", f"User: {final_p}\n{assistant_prompt}")
    print("Gold answer for final prompt:", gold)
    print(f"Total conversation length (chars): {len(conv)}")

print("\n--- Testing plot generation ---")

base_dir = os.path.dirname(os.path.abspath(__file__))
tmp_dir = os.path.join(base_dir, "tmp")
os.makedirs(tmp_dir, exist_ok=True)

sample1 = {
    "metric_by_len": {"0": 0.5, "1": 0.52, "2": 0.54, "3": 0.56, "4": 0.58, "5": 0.6, "6": 0.62},
    "cos_by_len": {
        "0": [0.1, 0.2, 0.3],
        "1": [0.15, 0.25, 0.35],
        "2": [0.2, 0.3, 0.4]
    },
    "model": "test-model",
    "target": "test-task",
    "distractor": "A"
}
sample2 = {
    "metric_by_len": {"0": 0.4, "1": 0.45, "2": 0.5, "3": 0.55, "4": 0.6, "5": 0.65, "6": 0.7},
    "cos_by_len": {
        "0": [0.5, 0.6, 0.7],
        "1": [0.55, 0.65, 0.75],
        "2": [0.6, 0.7, 0.8]
    },
    "model": "test-model",
    "target": "test-task",
    "distractor": "B"
}

open(os.path.join(tmp_dir, "A.json"), "w").write(json.dumps(sample1))
open(os.path.join(tmp_dir, "B.json"), "w").write(json.dumps(sample2))

out_dir = os.path.join(tmp_dir, "plots_test")
os.makedirs(out_dir, exist_ok=True)

df, model, target, cos_dict = df_from_dir(tmp_dir)

plot_metric(df, model, target, out_dir)
plot_pct_change(df, model, target, out_dir)
plot_cosine(cos_dict, model, target, out_dir)

assert os.path.isfile(os.path.join(out_dir, "metric.png"))
assert os.path.getsize(os.path.join(out_dir, "metric.png")) > 0

assert os.path.isfile(os.path.join(out_dir, "metric_pct_change.png"))
assert os.path.getsize(os.path.join(out_dir, "metric_pct_change.png")) > 0

assert os.path.isfile(os.path.join(out_dir, "cosine_A.png"))
assert os.path.getsize(os.path.join(out_dir, "cosine_A.png")) > 0
assert os.path.isfile(os.path.join(out_dir, "cosine_B.png"))
assert os.path.getsize(os.path.join(out_dir, "cosine_B.png")) > 0

print("Plot generation tests passed.")