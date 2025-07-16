import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Tuple

import evaluate
import numpy as np
import torch
from tqdm import tqdm
from core import DATASET_CFG, ModelWrapper, build_history, build_prompt, greedy_generate, load_split, run_example

def experiment(
    model_name: str,
    target: str,
    distractor: str,
    max_len: int,
    *,
    batch_size: int = 8,
    fp16: bool = False,
    no_cosine: bool = False,
):
    model = ModelWrapper(model_name, fp16=fp16)

    tgt_ds_name, tgt_split = (target.split("/", 1) + [None])[:2]
    dis_ds_name, dis_split = (distractor.split("/", 1) + [None])[:2]

    tgt_cfg = DATASET_CFG[tgt_ds_name]
    tgt_ds = list(load_split(tgt_ds_name, tgt_split))
    dis_ds = (
        tgt_ds
        if tgt_ds_name == dis_ds_name and (tgt_split == dis_split or dis_split is None)
        else list(load_split(dis_ds_name, dis_split))
    )

    metric_acc = tgt_cfg["metric"] == "accuracy"
    rouge_eval = evaluate.load("rouge") if tgt_cfg["metric"] == "rouge" else None

    if metric_acc:
        label_ids = [
            model.tokenizer.encode(tok, add_special_tokens=False)[0]
            for tok in tgt_cfg["answer_tokens"]
        ]

    prompt_cache: Dict[Tuple[int,int], str] = {}
    metric_by_len: Dict[str, float] = {}
    cos_by_len: Dict[str, List[float]] = {}
    dbg_top5: Dict[str, List[Tuple[str, int]]] = {}
    debug_examples: List[dict] = []
    n = len(tgt_ds)

    for h in range(max_len + 1):
        preds, sims, top_counter = [], [], Counter()
        prompts, golds = [], []
        
        for i in range(n):
            if (h, i) not in prompt_cache:
                history = build_history(dis_ds_name, dis_ds, i, h)
                final_p, gold = build_prompt(tgt_ds_name, tgt_ds[i])
                tag = "Assistant:" + (" <Answer>" if tgt_cfg["answer_suffix"] else "")
                conv = (history + "\n\n" if history else "") + f"User: {final_p}\n{tag}"
                prompt_cache[(h, i)] = conv
            prompts.append(prompt_cache[(h, i)])
            golds.append(gold)
            
        for i in tqdm(range(0, n, batch_size), desc=f"{tgt_ds_name}:{h}"):
            batch_prompts = prompts[i:i+batch_size]
            batch_golds   = golds[i:i+batch_size]
            if no_cosine:
                toks = model.to_tokens(batch_prompts, prepend_bos=True)
                logits_batch = model.model(toks)[:, -1, :].detach().cpu()
                cos_list_batch = [[0.0]*model.model.cfg.n_layers for _ in batch_prompts]
            else:
                logits_batch, cos_list_batch = run_example(model, batch_prompts)
            sims.extend(cos_list_batch)

            if metric_acc:
                # pick best known answer directly
                answer_logits = logits_batch[:, label_ids]               # [B, num_labels]
                choice = answer_logits.argmax(dim=-1)                    # [B]
                for j, cidx in enumerate(choice.tolist()):
                    label = tgt_cfg["labels"][cidx]
                    preds.append(label)
                    for tid in torch.topk(logits_batch[j], 5).indices.tolist():
                        top_counter[tid] += 1
                    debug_examples.append({
                        "prompt_text": batch_prompts[j],
                        "model_prediction": label,
                        "expected_answer": batch_golds[j],
                        "config": {
                            "target_dataset": tgt_ds_name,
                            "distractor_dataset": dis_ds_name,
                            "history_len": h,
                            "target_idx": i + j,
                        },
                    })
            else:
                gen_batch = greedy_generate(
                    model,
                    batch_prompts,
                    max_new_tokens=32,
                )
                for j, generated in enumerate(gen_batch):
                    label = generated.strip()
                    preds.append(label)
                    for tid in torch.topk(logits_batch[j], 5).indices.tolist():
                        top_counter[tid] += 1
                    debug_examples.append({
                        "prompt_text": batch_prompts[j],
                        "model_prediction": label,
                        "expected_answer": batch_golds[j],
                        "config": {
                            "target_dataset": tgt_ds_name,
                            "distractor_dataset": dis_ds_name,
                            "history_len": h,
                            "target_idx": i + j,
                        },
                    })
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

        if metric_acc:
            metric_by_len[str(h)] = sum(p == r for p, r in zip(preds, golds)) / n
        else:
            rl = rouge_eval.compute(predictions=preds, references=golds)["rougeL"]
            metric_by_len[str(h)] = rl["fmeasure"] if isinstance(rl, dict) else float(rl)

        cos_by_len[str(h)] = np.mean(sims, axis=0).tolist()
        dbg_top5[str(h)] = [
            (model.to_string(tid).strip(), cnt) for tid, cnt in top_counter.most_common(5)
        ]

    return metric_by_len, cos_by_len, tgt_cfg["metric"], dbg_top5, debug_examples

def _safe_json_dump(obj: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_results(
    model: str,
    target: str,
    distractor: str,
    metric_name: str,
    metric_by_len: Dict[str, float],
    cos_by_len: Dict[str, List[float]],
    dbg: Dict[str, List[Tuple[str, int]]],
    out_dir: str,
):
    fname = f"{model.replace('/','@')}__{target}_vs_{distractor}.json"
    _safe_json_dump(
        {
            "model": model,
            "target": target,
            "distractor": distractor,
            "metric_name": metric_name,
            "metric_by_len": metric_by_len,
            "cos_by_len": cos_by_len,
            "debug_top5_by_len": dbg,
        },
        os.path.join(out_dir, fname),
    )

def save_debug_log(model: str, target: str, distractor: str, examples: List[dict], out_dir: str):
    fname = f"{model.replace('/','@')}__{target}_vs_{distractor}_debug.json"
    _safe_json_dump(
        {
            "model": model,
            "target_task": target,
            "distractor_task": distractor,
            "debug_examples": examples,
        },
        os.path.join(out_dir, fname),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--distractor", required=True)
    ap.add_argument("--max_len", type=int, default=6)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--no_cosine", action="store_true")
    args = ap.parse_args()

    metrics, cosines, metric_name, dbg, debug_log = experiment(
        args.model,
        args.target,
        args.distractor,
        args.max_len,
        batch_size=args.batch_size,
        fp16=args.fp16,
        no_cosine=args.no_cosine,
    )
    save_results(args.model, args.target, args.distractor, metric_name, metrics, cosines, dbg, args.out_dir)
    save_debug_log(args.model, args.target, args.distractor, debug_log, args.out_dir)

if __name__ == "__main__":
    main()
