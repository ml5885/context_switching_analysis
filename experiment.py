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

def _score_multitoken_labels(model_wrapper, prompts, label_tok_lists):
    """
    Returns a tensor [B, L] where L = number of labels.
    For each prompt b and label l, we sum log P(label_tokens | prompt + previous label_tokens).
    """
    tok = model_wrapper.tokenizer  # shorthand
    model = model_wrapper.model
    device = model_wrapper.device

    # tokenize prompts individually to keep true lengths
    input_ids_list = [
        tok(p, return_tensors="pt", padding=False, add_special_tokens=True)["input_ids"].squeeze(0)
        for p in prompts
    ]

    scores = torch.zeros(len(prompts), len(label_tok_lists), device="cpu")

    with torch.no_grad():
        for b, prompt_ids_cpu in enumerate(input_ids_list):
            prompt_ids = prompt_ids_cpu.to(device)
            for li, cand in enumerate(label_tok_lists):
                cand_ids = torch.tensor(cand, device=device)
                full = torch.cat([prompt_ids, cand_ids], dim=0).unsqueeze(0)

                out = model(full)
                logits = out.logits  # [1, T, V]
                log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)[0]  # [T-1, V]

                # first predicted token index for candidate
                start = prompt_ids.size(0) - 1
                lp = 0.0
                for t, tok_id in enumerate(cand_ids.tolist()):
                    lp += log_probs[start + t, tok_id].item()
                scores[b, li] = lp

    return scores

def experiment(model_name, target, distractor, max_len, *, batch_size=8, fp16=False, no_cosine=False):
    # init
    model_wrapper = ModelWrapper(model_name, fp16=fp16)

    # dataset parsing
    tgt_ds_name, tgt_split = (target.split("/", 1) + [None])[:2]
    dis_ds_name, dis_split = (distractor.split("/", 1) + [None])[:2]

    tgt_cfg = DATASET_CFG[tgt_ds_name]
    tgt_ds = list(load_split(tgt_ds_name, tgt_split))
    if tgt_ds_name == dis_ds_name and (tgt_split == dis_split or dis_split is None):
        dis_ds = tgt_ds
    else:
        dis_ds = list(load_split(dis_ds_name, dis_split))

    metric_acc = tgt_cfg["metric"] == "accuracy"
    rouge_eval = evaluate.load("rouge") if tgt_cfg["metric"] == "rouge" else None

    # label tokenization
    if metric_acc:
        label_tok_lists = [
            model_wrapper.tokenizer.encode(tok, add_special_tokens=False)
            for tok in tgt_cfg["answer_tokens"]
        ]
        assert not all(len(x) == 1 for x in label_tok_lists), "You are in single-token mode. This will be wrong for Llama-2."
        all_single = all(len(x) == 1 for x in label_tok_lists)
        single_label_ids = [x[0] for x in label_tok_lists] if all_single else None

    # caches
    prompt_cache = {}
    metric_by_len = {}
    cos_by_len = {}
    dbg_top5 = {}
    debug_examples = []

    n = len(tgt_ds)

    for h in range(max_len + 1):
        preds, sims = [], []
        top_counter = Counter()
        prompts, golds = [], []

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


        # batch loop
        for i in tqdm(range(0, n, batch_size), desc=f"{tgt_ds_name}:{h}"):
            batch_prompts = prompts[i:i + batch_size]
            batch_golds = golds[i:i + batch_size]

            if no_cosine:
                toks = model_wrapper.to_tokens(batch_prompts, prepend_bos=True)
                logits_batch = model_wrapper.model(toks).logits[:, -1, :].detach().cpu()
                cos_list_batch = [[0.0] * model_wrapper.num_layers for _ in batch_prompts]
            else:
                logits_batch, cos_list_batch = run_example(model_wrapper, batch_prompts)
                logits_batch = logits_batch.cpu()

            sims.extend(cos_list_batch)

            if metric_acc:
                cand_scores = _score_multitoken_labels(model_wrapper, batch_prompts, label_tok_lists)
                choice = cand_scores.argmax(dim=-1)
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
                # generation tasks (tweetqa)
                gen_batch = greedy_generate(model_wrapper, batch_prompts, max_new_tokens=32)
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
                gc.collect()

        # metrics
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
