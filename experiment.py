import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
import evaluate

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

def eval_rouge(predictions, references):
    """Evaluate using ROUGE metric"""
    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=predictions, references=references)
    
    # Add mean character count as additional metric
    mean_chars = np.mean([len(pred) for pred in predictions])
    scores["mean_num_of_chars"] = mean_chars
    
    return scores

def pick_answer(logits: torch.Tensor, cfg: dict, model: HookedTransformer) -> str:
    ids = [model.to_tokens(tok)[0, 0].item() for tok in cfg["answer_tokens"]]
    idx = torch.argmax(logits[:, ids], dim=-1).item()
    return cfg["labels"][idx]

def extract_answer_from_generation(text: str) -> str:
    """Extract answer from model generation for generative tasks"""
    # Look for text between <Answer> tags
    start_tag = "<Answer>"
    end_tag = "</Answer>"
    
    start_idx = text.find(start_tag)
    if start_idx == -1:
        # Fallback: return everything after "Answer:" if no tags found
        answer_idx = text.find("Answer:")
        if answer_idx != -1:
            return text[answer_idx + len("Answer:"):].strip()
        return text.strip()
    
    start_idx += len(start_tag)
    end_idx = text.find(end_tag, start_idx)
    
    if end_idx == -1:
        return text[start_idx:].strip()
    
    return text[start_idx:end_idx].strip()

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

    # Ensure consistent shuffling with proper seeding for reproducible results
    target_ds = load_split(target).shuffle(seed=42).select(range(n))
    dis_ds = load_split(distractor).shuffle(seed=42).select(range(n))
    histories = build_histories(max_len)
    metric_name = dataset_config[target]['metric']
    
    metric_by_len, cos_by_len, predictions_debug = {}, {}, {}

    for seq in tqdm(histories, desc="histories"):
        h = len(seq)
        preds, refs, sims = [], [], []
        sequence_predictions = []

        for i in tqdm(range(n), leave=False, desc="examples"):
            turns = []
            for tag in seq:
                p, a = (
                    build_prompt(target, target_ds[i])
                    if tag == "A"
                    else build_prompt(distractor, dis_ds[i])
                )
                turns.append(f"{p}{a}")
            history = "\n\n".join(turns)

            final_p, gold = build_prompt(target, target_ds[i])
            conv = (history + "\n\n" if history else "") + final_p

            cache, logits = run_example(model, conv)

            if metric_name == "accuracy":
                pred = pick_answer(logits, dataset_config[target], model)
                preds.append(pred)
                refs.append(gold)
                sequence_predictions.append({
                    "example_id": i,
                    "sequence_pattern": seq,
                    "predicted_answer": pred,
                    "expected_answer": gold,
                    "is_correct": pred == gold,
                    "prompt_snippet": final_p[:100] + "..." if len(final_p) > 100 else final_p
                })
            elif metric_name == "rouge":
                prompt_tokens = model.to_tokens(conv, prepend_bos=True)
                
                generated_tokens = model.generate(
                    prompt_tokens, 
                    max_new_tokens=100, 
                    temperature=0.0,
                    do_sample=False,
                    stop_at_eos=True
                )
                
                full_text = model.to_string(generated_tokens[0])
                
                prompt_text = model.to_string(prompt_tokens[0])
                generated_answer = full_text[len(prompt_text):].strip()
                
                extracted_answer = extract_answer_from_generation(generated_answer)
                
                preds.append(extracted_answer)
                refs.append(gold)
                sequence_predictions.append({
                    "example_id": i,
                    "sequence_pattern": seq,
                    "predicted_answer": extracted_answer,
                    "expected_answer": gold,
                    "full_generation": generated_answer[:200] + "..." if len(generated_answer) > 200 else generated_answer,
                    "prompt_snippet": final_p[:100] + "..." if len(final_p) > 100 else final_p
                })

            sims.append(layer_cosines(cache, logits, model.W_U, model.cfg.n_layers))

        # Calculate metrics
        if metric_name == "accuracy":
            acc = sum(p == r for p, r in zip(preds, refs)) / len(refs)
            metric_by_len.setdefault(str(h), []).append(acc)
        elif metric_name == "rouge":
            rouge_scores = eval_rouge(preds, refs)
            # Use ROUGE-L F1 score as the main metric
            metric_by_len.setdefault(str(h), []).append(rouge_scores['rougeL'])

        cos_by_len.setdefault(str(h), []).append(np.mean(sims, axis=0).tolist())
        
        predictions_debug[str(h)] = sequence_predictions

    return metric_by_len, cos_by_len, metric_name, predictions_debug

def save_results(
    model: str,
    target: str,
    distractor: str,
    metric_name: str,
    metric_by_len,
    cos_by_len,
    predictions_debug,
    out_dir="results",
):
    os.makedirs(out_dir, exist_ok=True)
    base_filename = f"{model.replace('/','@')}__{target}_vs_{distractor}"
    
    # Main results file
    main_results = {
        "experiment_metadata": {
            "description": "Context switching analysis experiment results",
            "model_name": model,
            "target_task": target,
            "distractor_task": distractor,
            "evaluation_metric": metric_name
        },
        "performance_metrics": {
            "description": f"Model performance ({metric_name}) as a function of conversation history length",
            "explanation": "Each history length may have multiple sequence patterns (e.g., [B,A], [A,B] for length 2)",
            "data_by_history_length": metric_by_len
        },
        "layer_analysis": {
            "description": "Cosine similarity between intermediate layer representations and final layer output",
            "explanation": "Shows how much each layer's representation aligns with the final prediction across different history lengths",
            "cosine_similarities_by_history_length": cos_by_len
        }
    }
    
    # Save main results
    main_file = os.path.join(out_dir, f"{base_filename}.json")
    with open(main_file, "w") as f:
        json.dump(main_results, f, indent=2)
    print("saved main results to", main_file)
    
    # Save detailed predictions for debugging
    debug_results = {
        "experiment_metadata": {
            "description": "Detailed predictions for debugging context switching analysis",
            "model_name": model,
            "target_task": target,
            "distractor_task": distractor,
            "evaluation_metric": metric_name
        },
        "predictions_by_history_length": {
            "description": "All model predictions with expected answers for each history length and sequence pattern",
            "explanation": "Shows what the model actually predicted vs what was expected, useful for debugging performance issues",
            "data": predictions_debug
        }
    }
    
    debug_file = os.path.join(out_dir, f"{base_filename}_debug.json")
    with open(debug_file, "w") as f:
        json.dump(debug_results, f, indent=2)
    print("saved debug predictions to", debug_file)

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

    metric_by_len, cos_by_len, metric_name, predictions_debug = experiment(
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
        predictions_debug,
        args.out_dir,
    )

if __name__ == "__main__":
    main()
