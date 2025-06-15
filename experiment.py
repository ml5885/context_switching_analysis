import argparse
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import evaluate
from tqdm import tqdm
from transformer_lens import HookedTransformer
from data_utils import load_split, build_prompt, build_histories, dataset_config

def pick_device(choice):
    if choice:
        return choice
    return "cuda" if torch.cuda.is_available() else "cpu"

def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)

def layer_cosines(cache, final_logits, w_u, layers):
    return [
        cosine(cache["resid_post", l][:, -1, :] @ w_u, final_logits).mean().item()
        for l in range(layers)
    ]

def run_example(model, text):
    toks = model.to_tokens(text, prepend_bos=True)
    logits, cache = model.run_with_cache(toks)
    return cache, logits[:, -1, :]

def experiment(model_name, target, distractor, n, max_len, device):
    device = pick_device(device)
    print(
        f"\n=== Running {model_name} | target={target} | distractor={distractor} | n={n} | max_len={max_len} | device={device} ==="
    )
    model = HookedTransformer.from_pretrained(model_name, device=device)
    tgt_ds = load_split(target, limit=n)
    dis_ds = load_split(distractor, limit=n)
    n_examples = min(len(tgt_ds), len(dis_ds), n)
    histories = build_histories(max_len)
    metric_name = dataset_config[target]["metric"]
    metric_obj = evaluate.load(metric_name)
    metric_by_len, cos_by_len = {}, {}

    for seq in tqdm(histories, desc="histories", leave=True):
        h = len(seq)
        preds, refs, sims_collect = [], [], []

        for idx in tqdm(range(n_examples), leave=False, desc="examples"):
            turns = []
            for tag in seq:
                if tag == "A":
                    p, a = build_prompt(target, tgt_ds[idx])
                else:
                    p, a = build_prompt(distractor, dis_ds[idx])
                turns.append(f"{p}{a}")

            hist_text = "\n\n".join(turns)
            final_p, gold = build_prompt(target, tgt_ds[idx])
            conv = (hist_text + "\n\n" if hist_text else "") + final_p
            cache, fin = run_example(model, conv)

            if metric_name == "accuracy":
                preds.append(fin.argmax(-1).item())
                refs.append(model.to_tokens(" " + gold)[0, 0].item())
            else:
                gen = model.generate_text(conv, max_new_tokens=128)
                preds.append(gen[len(conv):])
                refs.append(gold)

            sims_collect.append(
                layer_cosines(cache, fin, model.W_U, model.cfg.n_layers)
            )

        score = metric_obj.compute(
            predictions=preds,
            references=refs
        )[metric_name if metric_name != "rouge" else "rougeL"]

        metric_by_len.setdefault(str(h), []).append(score)
        cos_by_len.setdefault(str(h), []).append(np.mean(sims_collect, axis=0).tolist())

    return metric_by_len, cos_by_len, metric_name

def save_results(
    model, target, distractor, metric_name, metric_by_len, cos_by_len, out_dir="results"
):
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{model.replace('/','@')}__{target}_vs_{distractor}.json"
    path = os.path.join(out_dir, fname)
    with open(path, "w") as f:
        json.dump(
            {
                "model": model,
                "target": target,
                "distractor": distractor,
                "metric_name": metric_name,
                "metric_by_len": metric_by_len,
                "cos_by_len": cos_by_len,
            },
            f,
            indent=2,
        )
    print("saved results to", path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--distractor", required=True)
    ap.add_argument("--max_len", type=int, default=10)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    metric_by_len, cos_by_len, metric_name = experiment(
        args.model,
        args.target,
        args.distractor,
        args.n,
        args.max_len,
        args.device,
    )
    save_results(
        args.model,
        args.target,
        args.distractor,
        metric_name,
        metric_by_len,
        cos_by_len,
        args.out_dir,
    )

if __name__ == "__main__":
    main()
