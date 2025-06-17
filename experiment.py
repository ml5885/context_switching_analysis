import argparse
import json
import os
import random
from collections import Counter

import evaluate
import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from data_utils import load_split, build_prompt, dataset_config

def pick_device(pref):
    return pref or ("cuda" if torch.cuda.is_available() else "cpu")

def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)

@torch.no_grad()
def run_example(model, text):
    toks = model.to_tokens(text, prepend_bos=True)

    final_logits = model(toks)[:, -1, :].detach()
    w_u = model.W_U

    layer_sims = torch.empty(model.cfg.n_layers, device="cpu")

    def make_hook(idx):
        def _hook(resid_post, hook):
            last_tok = resid_post[:, -1, :]
            sim = cosine(last_tok @ w_u, final_logits)
            layer_sims[idx] = sim.item()
        return _hook

    hooks = [
        (f"blocks.{i}.hook_resid_post", make_hook(i))
        for i in range(model.cfg.n_layers)
    ]

    model.run_with_hooks(toks, fwd_hooks=hooks, return_type=None)

    return final_logits.cpu(), layer_sims.tolist()

def build_histories(tasks, max_len):
    out = []
    for h in range(1, max_len + 1):
        for t in tasks:
            out.append([t] * h)
    return out

def pick_answer(logits, cfg, model):
    ids = [model.to_tokens(tok)[0, 0].item() for tok in cfg["answer_tokens"]]
    idx = torch.argmax(logits[:, ids], dim=-1).item()
    return cfg["labels"][idx]

def greedy_generate(model, prompt, max_new_tokens=32):
    toks = model.to_tokens(prompt, prepend_bos=True)
    for _ in range(max_new_tokens):
        logits = model(toks)[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        toks = torch.cat([toks, next_id], dim=-1)
        if next_id.item() == model.tokenizer.eos_token_id:
            break
    return model.to_string(toks[0])

def sample_examples(name, n):
    ds = load_split(name, streaming=True)
    selected = []
    print(f"Sampling {n} examples for {name}...")
    for i, ex in tqdm(enumerate(ds), desc=f"Scanning {name}"):
        if i < n:
            selected.append(ex)
        else:
            j = random.randint(0, i)
            if j < n:
                selected[j] = ex
        if i >= 10 * n:
            break
    if len(selected) < n:
        print(f"Warning: Only found {len(selected)}/{n} examples for {name}.")
    return selected

def experiment(model_name, tasks, target, distractor, n, max_len, device):
    device = pick_device(device)
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device
    )

    tgt_cfg = dataset_config[target]
    dis_cfg = dataset_config[distractor]

    tgt_ds = sample_examples(target, n)
    dis_ds = sample_examples(distractor, n)

    histories = build_histories(tasks, max_len)

    metric_acc = tgt_cfg["metric"] == "accuracy"
    metric_rouge = tgt_cfg["metric"] == "rouge"
    rouge_eval = evaluate.load("rouge") if metric_rouge else None

    metric_by_len, cos_by_len, dbg_top5 = {}, {}, {}
    all_debug_info = []

    for seq in tqdm(histories, desc="Histories"):
        h = len(seq)
        preds, refs, sims = [], [], []
        top_counter = Counter()
        hist_task = seq[0]
        hist_ds = tgt_ds if hist_task == target else dis_ds

        for i in tqdm(range(n), desc=f"  - History({h}, {hist_task})", leave=False):
            if i >= len(tgt_ds) or not hist_ds:
                continue

            turns = []
            # FIX: This loop now correctly selects unique examples for the history
            # that are different from the target example at index `i`.
            for j in range(h):
                hist_idx = (i + j + 1) % len(hist_ds)
                pp, aa = build_prompt(hist_task, hist_ds[hist_idx])
                turns.append(f"{pp}{aa}")

            history_text = "\n\n".join(turns)
            final_p, gold = build_prompt(target, tgt_ds[i])
            conv = (history_text + "\n\n" if history_text else "") + final_p

            logits, cos_list = run_example(model, conv)

            predicted_answer = None
            if metric_acc:
                pred_label = pick_answer(logits, tgt_cfg, model)
                preds.append(pred_label)
                refs.append(gold)
                predicted_answer = pred_label
            elif metric_rouge:
                gen = greedy_generate(model, conv, 64)
                generated_text = gen[len(conv):]
                preds.append(generated_text)
                refs.append(gold)
                predicted_answer = generated_text

            sims.append(cos_list)
            for tid in torch.topk(logits, 5, dim=-1).indices[0].tolist():
                top_counter[tid] += 1

            debug_entry = {
                "prompt_text": conv,
                "model_prediction": predicted_answer,
                "expected_answer": gold,
                "config": {
                    "target_dataset": target,
                    "distractor_dataset": distractor if hist_task == distractor else None,
                    "history_len": h,
                    "history_content_task": hist_task,
                    "target_example_dataset_idx": i,
                }
            }
            all_debug_info.append(debug_entry)
            torch.cuda.empty_cache()

        if not refs: continue

        if metric_acc:
            acc = sum(p == r for p, r in zip(preds, refs)) / len(refs)
            metric_by_len.setdefault(str(h), []).append(acc)
        elif metric_rouge:
            scores = [
                rouge_eval.compute(predictions=[p], references=[r])["rougeL"]
                for p, r in zip(preds, refs)
            ]
            metric_by_len.setdefault(str(h), []).append(float(np.mean(scores)))

        cos_by_len.setdefault(str(h), []).append(np.mean(sims, axis=0).tolist())
        dbg_top5[str(h)] = [
            (model.to_string(t).strip(), c) for t, c in top_counter.most_common(5)
        ]

    return metric_by_len, cos_by_len, tgt_cfg["metric"], dbg_top5, all_debug_info

def save_results(model, target, distractor, metric_name, metric_by_len, cos_by_len, dbg, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{model.replace('/','@')}__{target}_vs_{distractor}.json"
    with open(os.path.join(out_dir, fname), "w") as f:
        json.dump(
            {
                "model": model,
                "target": target,
                "distractor": distractor,
                "metric_name": metric_name,
                "metric_by_len": metric_by_len,
                "cos_by_len": cos_by_len,
                "debug_top5_by_len": dbg,
            },
            f,
            indent=2,
        )

def save_debug_log(model, target, distractor, debug_info, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{model.replace('/','@')}__{target}_vs_{distractor}_debug.json"
    with open(os.path.join(out_dir, fname), "w") as f:
        json.dump(
            {
                "model": model,
                "target_task": target,
                "distractor_task": distractor,
                "debug_examples": debug_info,
            },
            f,
            indent=2,
        )

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

    tasks = [args.target, args.distractor]
    metric_by_len, cos_by_len, metric_name, dbg, debug_log = experiment(
        args.model,
        tasks,
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
    save_debug_log(
        args.model,
        args.target,
        args.distractor,
        debug_log,
        args.out_dir,
    )

if __name__ == "__main__":
    main()