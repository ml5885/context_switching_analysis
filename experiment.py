import argparse
import json
import os
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

def pick_answer(logits, cfg, model):
    ids = [model.to_tokens(tok, prepend_bos=False)[0, 0].item() for tok in cfg["answer_tokens"]]
    idx = torch.argmax(logits[:, ids], dim=-1).item()
    return cfg["labels"][idx]

def greedy_generate(model, prompt, max_new_tokens=32):
    toks = model.to_tokens(prompt, prepend_bos=True)
    prompt_len = toks.shape[1]
    for _ in range(max_new_tokens):
        logits = model(toks)[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        toks = torch.cat([toks, next_id], dim=-1)
        if next_id.item() == model.tokenizer.eos_token_id:
            break
    generated_toks = toks[:, prompt_len:]
    return model.to_string(generated_toks[0])

def sample_examples(name):
    ds = load_split(name, streaming=False)
    print(f"Loaded {len(ds)} examples from {name}.")
    return list(ds)

def build_history_text(task, samples, idx, length):
    turns = []
    cfg = dataset_config[task]
    for j in range(length):
        sample = samples[(idx + j + 1) % len(samples)]
        prompt, answer = build_prompt(task, sample)
        suffix = cfg["answer_suffix"]
        if suffix:
            full_answer = f" <Answer>{answer}{suffix}"
        else:
            full_answer = f" {answer}"
        turns.append(f"User: {prompt}\nAssistant:{full_answer}")
    return "\n\n".join(turns)

def experiment(model_name, target, distractor, max_len, device, quantize=False):
    device = pick_device(device)
    if quantize:
        model = HookedTransformer.from_pretrained_no_processing(
            model_name,
            device=device,
            dtype=torch.bfloat16
        )
    else:
        model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
        )
    tgt_cfg = dataset_config[target]
    tgt_ds = sample_examples(target)
    dis_ds = tgt_ds if target == distractor else sample_examples(distractor)
    metric_acc = tgt_cfg["metric"] == "accuracy"
    metric_rouge = tgt_cfg["metric"] == "rouge"
    rouge_eval = evaluate.load("rouge") if metric_rouge else None
    metric_by_len, cos_by_len, dbg_top5, all_debug_info = {}, {}, {}, []
    n = len(tgt_ds)
    for h in tqdm(range(max_len + 1), desc="Testing History Lengths"):
        preds, refs, sims = [], [], []
        top_counter = Counter()
        for i in range(n):
            history_text = build_history_text(distractor, dis_ds, i, h)
            final_p, gold = build_prompt(target, tgt_ds[i])
            assistant_prompt = "Assistant:"
            if tgt_cfg["answer_suffix"]:
                assistant_prompt += " <Answer>"
            conv = (
                (history_text + "\n\n") if history_text else ""
            ) + f"User: {final_p}\n{assistant_prompt}"
            logits, cos_list = run_example(model, conv)
            if metric_acc:
                pred_label = pick_answer(logits, tgt_cfg, model)
                preds.append(pred_label)
                refs.append(gold)
                predicted_answer = pred_label
            else:
                generated_text = greedy_generate(model, conv, 64)
                preds.append(generated_text)
                refs.append(gold)
                predicted_answer = generated_text
            sims.append(cos_list)
            for tid in torch.topk(logits, 5, dim=-1).indices[0].tolist():
                top_counter[tid] += 1
            all_debug_info.append({
                "prompt_text": conv,
                "model_prediction": predicted_answer,
                "expected_answer": gold,
                "config": {
                    "target_dataset": target,
                    "distractor_dataset": distractor,
                    "history_len": h,
                    "history_content_task": distractor,
                    "target_example_dataset_idx": i,
                }
            })
            torch.cuda.empty_cache()
        if not refs:
            continue
        if metric_acc:
            metric_by_len[str(h)] = sum(p == r for p, r in zip(preds, refs)) / len(refs)
        else:
            rouge_res = rouge_eval.compute(predictions=preds, references=refs)
            rl = rouge_res["rougeL"]
            metric_by_len[str(h)] = rl["fmeasure"] if isinstance(rl, dict) else float(rl)
        cos_by_len[str(h)] = np.mean(sims, axis=0).tolist()
        dbg_top5[str(h)] = [
            (model.to_string(t).strip(), c) for t, c in top_counter.most_common(5)
        ]
    return metric_by_len, cos_by_len, tgt_cfg["metric"], dbg_top5, all_debug_info

def save_results(model, target, distractor, metric_name, metric_by_len, cos_by_len, dbg, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{model.replace('/','@')}__{target}_vs_{distractor}.json"
    with open(os.path.join(out_dir, fname), "w") as f:
        json.dump({
            "model": model,
            "target": target,
            "distractor": distractor,
            "metric_name": metric_name,
            "metric_by_len": metric_by_len,
            "cos_by_len": cos_by_len,
            "debug_top5_by_len": dbg,
        }, f, indent=2)

def save_debug_log(model, target, distractor, debug_info, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{model.replace('/','@')}__{target}_vs_{distractor}_debug.json"
    with open(os.path.join(out_dir, fname), "w") as f:
        json.dump({
            "model": model,
            "target_task": target,
            "distractor_task": distractor,
            "debug_examples": debug_info,
        }, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--distractor", required=True)
    ap.add_argument("--max_len", type=int, default=6)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--quantize", action="store_true")
    args = ap.parse_args()

    metric_by_len, cos_by_len, metric_name, dbg, debug_log = experiment(
        args.model,
        args.target,
        args.distractor,
        args.max_len,
        args.device,
        args.quantize,
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
