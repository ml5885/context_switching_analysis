from transformer_lens import HookedTransformer
from data_utils import build_histories, load_split, build_prompt

hist = build_histories(max_len=4)
print("total histories:", len(hist))
for seq in hist:
    print(seq)

for name in ["mmlu", "rotten_tomatoes", "cnn_dailymail"]:
    ds = load_split(name, limit=1)
    prompt, answer = build_prompt(name, ds[0])
    print(name, "prompt snippet:", prompt[:60].replace("\n", " "), "answer snippet:", answer[:40])

model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m")
print("model loaded")
