from transformer_lens import HookedTransformer
from data_utils import load_split, build_prompt, dataset_config

print("Testing data loading and prompt building for all configured datasets...")
for name in dataset_config.keys():
    print(f"\n--- Testing dataset: {name} ---")
    ds = load_split(name, streaming=False)
    sample = ds[0]
    prompt, answer = build_prompt(name, sample)
    print(f"Sample prompt snippet: {prompt}...")
    print(f"Sample answer: {answer}")

print("\n--- Testing model loading ---")
try:
    model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m", device="cpu")
    print("Model 'EleutherAI/pythia-70m' loaded successfully.")
    del model
except Exception as e:
    print(f"Failed to load model: {e}")
