import argparse
import json
import os
from collections import Counter

import evaluate
import numpy as np
import torch
from tqdm import tqdm
from data import DATASET_CFG, build_history, build_prompt, load_split
from model import ModelWrapper, greedy_generate, run_example
import gc

def _score_multitoken_labels(model_wrapper, prompts, label_tok_lists):
    """
    Compute summed log-probabilities for each label sequence per prompt.
    Slow but simple: 1 forward per (batch item, label).
    Returns a tensor [B, L] of scores.
    """
    tokenizer = model_wrapper.tokenizer
    model = model_wrapper.model
    device = model_wrapper.device

    # tokenize prompts once (with padding=False to keep true lengths)
    enc = tokenizer(prompts, return_tensors="pt", padding=False, add_special_tokens=True)
    B = len(prompts)
    scores = torch.zeros(B, len(label_tok_lists), device=device)

    for b in range(B):
        prompt_ids = torch.tensor(enc["input_ids"][b], device=device)
        for li, cand in enumerate(label_tok_lists):
            cand_ids = torch.tensor(cand, device=device)
            full_ids = torch.cat([prompt_ids, cand_ids], dim=0).unsqueeze(0)
            with torch.no_grad():
                out = model(full_ids).logits  # [1, T, V]
                log_probs = torch.log_softmax(out[:, :-1, :], dim=-1)[0]  # [T-1, V]

            prompt_len = prompt_ids.size(0)
            cand_len = cand_ids.size(0)
            # first candidate token is predicted at index prompt_len-1
            idx_start = prompt_len - 1
            lp_sum = 0.0
            for t in range(cand_len):
                tok_id = cand_ids[t].item()
                lp_sum += log_probs[idx_start + t, tok_id].item()
            scores[b, li] = lp_sum

    return scores.cpu()

def experiment(model_name, target, distractor, max_len, *, batch_size=8, fp16=False, no_cosine=False):
    # Initialize model
    model_wrapper = ModelWrapper(model_name, fp16=fp16)

    # Parse dataset and split names
    tgt_ds_name, tgt_split = (target.split("/", 1) + [None])[:2]
    dis_ds_name, dis_split = (distractor.split("/", 1) + [None])[:2]

    # Load datasets
    tgt_cfg = DATASET_CFG[tgt_ds_name]
    tgt_ds = list(load_split(tgt_ds_name, tgt_split))
    if tgt_ds_name == dis_ds_name and (tgt_split == dis_split or dis_split is None):
        dis_ds = tgt_ds
    else:
        dis_ds = list(load_split(dis_ds_name, dis_split))

    # Metric setup
    metric_acc = tgt_cfg["metric"] == "accuracy"
    rouge_eval = evaluate.load("rouge") if tgt_cfg["metric"] == "rouge" else None

    # Precompute label tokenization
    if metric_acc:
        label_tok_lists = [
            model_wrapper.tokenizer.encode(tok, add_special_tokens=False)
            for tok in tgt_cfg["answer_tokens"]
        ]
        single_token = all(len(x) == 1 for x in label_tok_lists)
        if single_token:
            label_ids = [x[0] for x in label_tok_lists]
        else:
            label_ids = None  # unused in multi-token case

    # Caches and results
    prompt_cache = {}
    metric_by_len = {}
    cos_by_len = {}
    dbg_top5 = {}
    debug_examples = []

    n = len(tgt_ds)

    for h in range(max_len + 1):
        preds = []
        sims = []
        top_counter = Counter()
        prompts = []
        golds = []

        # Build prompts and gold answers
        for i in range(n):
            if (h, i) not in prompt_cache:
                history = build_history(dis_ds_name, dis_ds, i, h)
                final_p, gold = build_prompt(tgt_ds_name, tgt_ds[i])
                tag = "Assistant:" + (" <Answer>" if tgt_cfg["answer_suffix"] else "")
                conv = (history + "\n\n" if history else "") + f"User: {final_p}\n{tag}"
                prompt_cache[(h, i)] = (conv, gold)
            conv, gold = prompt_cache[(h, i)]
            prompts.append(conv)
            golds.append(gold)

        print(f"[DEBUG] History len: {h}, Example prompt sample: {prompts[0]}")
        print(f"[DEBUG] Gold answer sample: {golds[0]}")

        # Batch inference
        for i in tqdm(range(0, n, batch_size), desc=f"{tgt_ds_name}:{h}"):
            batch_prompts = prompts[i:i + batch_size]
            batch_golds = golds[i:i + batch_size]
            print(f"[DEBUG] Batch idx: {i}, batch_prompts[0]: {batch_prompts[0]}")
            print(f"[DEBUG] Batch golds: {batch_golds}")

            if no_cosine:
                toks = model_wrapper.to_tokens(batch_prompts, prepend_bos=True)
                print(f"[DEBUG] toks shape: {toks.shape}")
                logits_batch = model_wrapper.model(toks).logits[:, -1, :].detach().cpu()
                print(f"[DEBUG] logits_batch shape: {logits_batch.shape}")
                cos_list_batch = [[0.0] * model_wrapper.num_layers for _ in batch_prompts]
            else:
                logits_batch, cos_list_batch = run_example(model_wrapper, batch_prompts)
                print(f"[DEBUG] logits_batch shape: {logits_batch.shape}")
                print(f"[DEBUG] cos_list_batch: {cos_list_batch}")

            sims.extend(cos_list_batch)

            if metric_acc:
                if single_token:
                    answer_logits = logits_batch[:, label_ids]
                    print(f"[DEBUG] answer_logits: {answer_logits}")
                    choice = answer_logits.argmax(dim=-1)
                    print(f"[DEBUG] choice: {choice}")
                    for j, cidx in enumerate(choice.tolist()):
                        label = tgt_cfg["labels"][cidx]
                        preds.append(label)
                        print(f"[DEBUG] Predicted label: {label}, Gold: {batch_golds[j]}")
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
                    # multi-token scoring path
                    cand_scores = _score_multitoken_labels(model_wrapper, batch_prompts, label_tok_lists)
                    print(f"[DEBUG] cand_scores: {cand_scores}")
                    choice = cand_scores.argmax(dim=-1)
                    print(f"[DEBUG] choice: {choice}")
                    for j, cidx in enumerate(choice.tolist()):
                        label = tgt_cfg["labels"][cidx]
                        preds.append(label)
                        print(f"[DEBUG] Predicted label: {label}, Gold: {batch_golds[j]}")
                        # still log top5 from the prompt-last-token logits
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
                # generation metric (tweetqa)
                gen_batch = greedy_generate(
                    model_wrapper,
                    batch_prompts,
                    max_new_tokens=32,
                )
                print(f"[DEBUG] gen_batch: {gen_batch}")
                for j, generated in enumerate(gen_batch):
                    label = generated.strip()
                    preds.append(label)
                    print(f"[DEBUG] Generated label: {label}, Gold: {batch_golds[j]}")
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
                gc.collect()

        # Compute metrics
        if metric_acc:
            metric_by_len[str(h)] = sum(p == r for p, r in zip(preds, golds)) / n
        else:
            rl = rouge_eval.compute(predictions=preds, references=golds)["rougeL"]
            metric_by_len[str(h)] = rl["fmeasure"] if isinstance(rl, dict) else float(rl)

        cos_by_len[str(h)] = np.mean(sims, axis=0).tolist()
        dbg_top5[str(h)] = [
            (model_wrapper.to_string(tid).strip(), cnt) for tid, cnt in top_counter.most_common(5)
        ]

    return metric_by_len, cos_by_len, tgt_cfg["metric"], dbg_top5, debug_examples

def _safe_json_dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_results(model, target, distractor, metric_name, metric_by_len, cos_by_len, dbg, out_dir):
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

def save_debug_log(model, target, distractor, examples, out_dir):
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
