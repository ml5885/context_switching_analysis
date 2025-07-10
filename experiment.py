import argparse
import json
import os
from collections import Counter

import evaluate
import numpy as np
import torch
from tqdm import tqdm

from data_utils import load_split, build_prompt, dataset_config
from model_wrapper import ModelWrapper

def pick_device(pref):
    return pref or ("cuda" if torch.cuda.is_available() else "cpu")

def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)

@torch.no_grad()
def run_example(model, texts):
    toks = model.to_tokens(texts, prepend_bos=True)
    if model.tlens:
        final_logits = model.model(toks)[:, -1, :].detach()
        w_u = model.W_U
        layer_sims = torch.empty(toks.shape[0], model.model.cfg.n_layers, device="cpu")

        def make_hook(idx):
            def _hook(resid_post, hook):
                last_tok = resid_post[:, -1, :]
                sim = cosine(last_tok @ w_u, final_logits)
                layer_sims[:, idx] = sim.cpu()

            return _hook

        hooks = [(f"blocks.{i}.hook_resid_post", make_hook(i)) for i in range(model.model.cfg.n_layers)]
        model.model.run_with_hooks(toks, fwd_hooks=hooks, return_type=None)
    else:
        outputs = model.model(toks, output_hidden_states=True)
        final_logits = outputs.logits[:, -1, :].detach()
        w_u = model.W_U
        h_states = outputs.hidden_states[1:]
        sims = torch.stack([cosine(h[:, -1, :] @ w_u, final_logits) for h in h_states], dim=1)
        layer_sims = sims.cpu()
    return final_logits.cpu(), layer_sims.tolist()

@torch.no_grad()
def greedy_generate(model, prompts, max_new_tokens=512):
    toks = model.to_tokens(prompts, prepend_bos=True)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "verbose": False,
    }
    if not model.tlens:
        gen_kwargs["pad_token_id"] = model.tokenizer.pad_token_id

    gen = model.model.generate(toks, **gen_kwargs)
    gen_ids = gen[:, toks.shape[1]:]
    return [model.to_string(ids) for ids in gen_ids]


def parse_task(task):
    """Parse a task string like 'dataset' or 'dataset/split'."""
    if "/" in task:
        dataset, split = task.split("/", 1)
        return dataset, split
    return task, None

def sample_examples(name, split=None, debug=False):
    ds = load_split(name, split=split, streaming=False, debug=debug)
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

def experiment(model_name, target, distractor, max_len, device, quantize=False, debug=False, batch_size=8):
    device = pick_device(device)
    model = ModelWrapper(model_name, device=device, quantize=quantize)

    tgt_name, tgt_split = parse_task(target)
    dis_name, dis_split = parse_task(distractor)

    tgt_cfg = dataset_config[tgt_name]
    tgt_ds = sample_examples(tgt_name, split=tgt_split, debug=debug)
    dis_ds = tgt_ds if (tgt_name == dis_name and (tgt_split == dis_split or dis_split is None)) else sample_examples(dis_name, split=dis_split, debug=debug)

    metric_acc = tgt_cfg["metric"] == "accuracy"
    metric_rouge = tgt_cfg["metric"] == "rouge"
    rouge_eval = evaluate.load("rouge") if metric_rouge else None

    metric_by_len, cos_by_len, dbg_top5, all_debug_info = {}, {}, {}, []
    n = len(tgt_ds)

    for h in range(max_len + 1):

        preds, sims = [], []
        top_counter = Counter()

        prompts = []
        golds = []
        for i in range(n):
            history_text = build_history_text(dis_name, dis_ds, i, h)
            final_p, gold = build_prompt(tgt_name, tgt_ds[i])
            
            # Create the conversation prompt
            assistant_prompt = "Assistant:"
            if tgt_cfg["answer_suffix"]:
                assistant_prompt += " <Answer>"
            conv = (history_text + "\n\n" if history_text else "") + f"User: {final_p}\n{assistant_prompt}"
            prompts.append(conv)
            golds.append(gold)

        refs = golds

        for i in tqdm(range(0, n, batch_size), desc=f"{tgt_name}:{h}"):
            batch_prompts = prompts[i:i+batch_size]
            batch_golds = golds[i:i+batch_size]
            
            logits_batch, cos_list_batch = run_example(model, batch_prompts)
            sims.extend(cos_list_batch)

            # Get the model's prediction
            if metric_acc:
                max_new = 10
            else:
                max_new = 64
            
            generated_batch = greedy_generate(model, batch_prompts, max_new)

            for j, generated in enumerate(generated_batch):
                if metric_acc:
                    label = generated.strip()
                    suffix = tgt_cfg["answer_suffix"]
                    if suffix and suffix in label:
                        label = label.split(suffix)[0]
                    preds.append(label)
                    predicted_answer = label
                else:
                    preds.append(generated)
                    predicted_answer = generated

                for tid in torch.topk(logits_batch[j], 5, dim=-1).indices.tolist():
                    top_counter[tid] += 1

                all_debug_info.append({
                    "prompt_text": batch_prompts[j],
                    "model_prediction": predicted_answer,
                    "expected_answer": batch_golds[j],
                    "config": {
                        "target_dataset": tgt_name,
                        "distractor_dataset": dis_name,
                        "history_len": h,
                        "history_content_task": dis_name,
                        "target_example_dataset_idx": i + j,
                    }
                })

        if not refs:
            continue
        
        # Compute metrics
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
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()
    
    metric_by_len, cos_by_len, metric_name, dbg, debug_log = experiment(
        args.model,
        args.target,
        args.distractor,
        args.max_len,
        args.device,
        args.quantize,
        debug=args.debug,
        batch_size=args.batch_size
    )
    save_results(
        args.model,
        args.target,
        args.distractor,
        metric_name,
        metric_by_len,
        cos_by_len,
        dbg,
        args.out_dir
    )
    save_debug_log(
        args.model,
        args.target,
        args.distractor,
        debug_log,
        args.out_dir
    )

if __name__ == "__main__":
    main()
