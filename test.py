from transformer_lens import HookedTransformer
from data_utils import load_split, build_prompt, dataset_config

print("Testing data loading and prompt building for all configured datasets...")
for name in dataset_config.keys():
    print(f"\n--- Testing dataset: {name} ---")
    ds = load_split(name, streaming=False)
    sample = ds[0]
    prompt, answer = build_prompt(name, sample)
    print(f"Sample prompt:\n{prompt}")
    print(f"Sample answer: {answer}")

print("\n--- Testing conversation history building ---")
max_len = 2
target_task = "mmlu"
distractor_task = "rotten_tomatoes"

print(f"Target: {target_task}, Distractor: {distractor_task}, max_len: {max_len}")

target_ds = list(load_split(target_task, streaming=False))
distractor_ds = list(load_split(distractor_task, streaming=False))

for h in range(max_len + 1):
    print(f"\n--- History length: {h} ---")

    turns = []
    for j in range(h):
        hist_sample = distractor_ds[j + 1]
        pp, aa = build_prompt(distractor_task, hist_sample)
        turns.append(f"{pp}{aa}")

    history_text = "\n\n".join(turns)

    final_p, gold = build_prompt(target_task, target_ds[0])

    conv = (history_text + "\n\n" if history_text else "") + final_p

    print(f"History contains {len(turns)} turn(s) from '{distractor_task}'.")
    if h > 0:
        print("History text:\n", history_text)
    print("Final prompt:\n", final_p)
    print("Gold answer for final prompt:", gold)
    print(f"Total conversation length (chars): {len(conv)}")

print("\n--- Testing model loading ---")
try:
    model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m", device="cpu")
    print("Model 'EleutherAI/pythia-70m' loaded successfully.")
    del model
except Exception as e:
    print(f"Failed to load model: {e}")
