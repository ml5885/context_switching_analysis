import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from data_utils import load_split, build_prompt, build_histories, dataset_config

def pick_device(pref: str | None) -> str:
    if pref is None:
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    return pref

def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)

def layer_cosines(cache, final_logits, w_u, layers):
    return [
        cosine(cache["resid_post", l][:, -1, :] @ w_u, final_logits).mean().item()
        for l in range(layers)
    ]

def run_example(model: HookedTransformer, text: str):
    toks = model.to_tokens(text, prepend_bos=True)
    logits, cache = model.run_with_cache(toks)
    return cache, logits[:, -1, :]

def pick_answer(logits: torch.Tensor, cfg: dict, model: HookedTransformer) -> str:
    ids = [model.to_tokens(tok)[0, 0].item() for tok in cfg["answer_tokens"]]
    idx = torch.argmax(logits[:, ids], dim=-1).item()
    return cfg["labels"][idx]

def experiment(
    model_name: str,
    target: str,
    distractor: str,
    n: int,
    max_len: int,
    device: str | None,
):
    device = pick_device(device)
    print(
        f"\n=== Running {model_name} | target={target} | distractor={distractor} "
        f"| n={n} | max_len={max_len} | device={device} ==="
    )

    model = HookedTransformer.from_pretrained(model_name, device=device)

    tgt_ds = load_split(target).shuffle(seed=42).select(range(n))
    dis_ds = load_split(distractor).shuffle(seed=42).select(range(n))
    histories = build_histories(max_len)

    tgt_cfg = dataset_config[target]
    metric_is_acc = tgt_cfg["metric"] == "accuracy"

    metric_by_len, cos_by_len, dbg_top5 = {}, {}, {}

    for seq in tqdm(histories, desc="histories"):
        h = len(seq)
        preds, refs, sims = [], [], []
        top_counter = Counter()

        for i in tqdm(range(n), leave=False, desc="examples"):
            turns = []
            for tag in seq:
                p, a = (
                    build_prompt(target, tgt_ds[i])
                    if tag == "A"
                    else build_prompt(distractor, dis_ds[i])
                )
                turns.append(f"{p}{a}")
            history = "\n\n".join(turns)

            final_p, gold = build_prompt(target, tgt_ds[i])
            conv = (history + "\n\n" if history else "") + final_p

            cache, logits = run_example(model, conv)

            if metric_is_acc:
                preds.append(pick_answer(logits, tgt_cfg, model))
                refs.append(gold)

            sims.append(layer_cosines(cache, logits, model.W_U, model.cfg.n_layers))

            for tid in torch.topk(logits, 5, dim=-1).indices[0].tolist():
                top_counter[tid] += 1

        if metric_is_acc:
            acc = sum(p == r for p, r in zip(preds, refs)) / len(refs)
            metric_by_len.setdefault(str(h), []).append(acc)

        cos_by_len.setdefault(str(h), []).append(np.mean(sims, axis=0).tolist())
        dbg_top5[str(h)] = [
            (model.to_string(t).strip(), c) for t, c in top_counter.most_common(5)
        ]

    metric_name = "accuracy" if metric_is_acc else "none"
    return metric_by_len, cos_by_len, metric_name, dbg_top5

def save_results(
    model: str,
    target: str,
    distractor: str,
    metric_name: str,
    metric_by_len,
    cos_by_len,
    dbg,
    out_dir="results",
):
    os.makedirs(out_dir, exist_ok=True)
    fn = f"{model.replace('/','@')}__{target}_vs_{distractor}.json"
    with open(os.path.join(out_dir, fn), "w") as f:
        json.dump(
            dict(
                model=model,
                target=target,
                distractor=distractor,
                metric_name=metric_name,
                metric_by_len=metric_by_len,
                cos_by_len=cos_by_len,
                debug_top5_by_len=dbg,
            ),
            f,
            indent=2,
        )
    print("saved results to", fn)

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

    metric_by_len, cos_by_len, metric_name, dbg = experiment(
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
        dbg,
        args.out_dir,
    )

if __name__ == "__main__":
    main()
