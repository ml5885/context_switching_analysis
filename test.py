import evaluate
from transformer_lens import HookedTransformer
from data_utils import load_split, build_prompt, dataset_config
from experiment import build_history_text

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
        preds = ["A", "B", "C", "D", "A"]
        refs = ["A", "C", "C", "D", "B"]
        acc = sum(p == r for p, r in zip(preds, refs)) / len(refs)
        print(f"Accuracy test: {acc} (expected 0.6 for default mocks)")
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

target_ds = list(load_split(target_task, streaming=False))
distractor_ds = list(load_split(distractor_task, streaming=False))

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
