from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

DATASET_CFG: Dict[str, Dict[str, Any]] = {
    "mmlu": {
        "hf_name": "cais/mmlu",
        "subset": "all",
        "split": "test",
        "metric": "accuracy",
        "labels": ["A", "B", "C", "D"],
        "answer_tokens": [" A", " B", " C", " D"],
        "prompt_template": (
            "You have a multiple-choice question on {topic}. Only one of "
            "the options is correct: A, B, C, or D. Give your answer in the "
            "following format with the tags provided: <Answer> </Answer>.\n"
            "Question: {question}\n"
            "(A) {choice_a}\n(B) {choice_b}\n(C) {choice_c}\n(D) {choice_d}"
        ),
        "answer_suffix": "</Answer>",
    },
    "rotten_tomatoes": {
        "hf_name": "cornell-movie-review-data/rotten_tomatoes",
        "subset": None,
        "split": "test",
        "metric": "accuracy",
        "labels": ["negative", "positive"],
        "answer_tokens": [" negative", " positive"],
        "prompt_template": (
            "Can you choose only one sentiment ['negative', 'positive'] for this review.\n"
            "review: {review}\n"
            "Return only the sentiment label without any other text. Follow the format:\n"
            "<Answer> positive / negative </Answer>."
        ),
        "answer_suffix": "</Answer>",
    },
    "tweetqa": {
        "hf_name": "ucsbnlp/tweet_qa",
        "subset": None,
        "split": "validation",
        "metric": "rouge",
        "labels": [],
        "answer_tokens": [],
        "prompt_template": (
            "Read the given tweet and answer the corresponding question.\n"
            "tweet: {tweet}\n"
            "question: {question}"
        ),
        "answer_suffix": "",
    },
}

def load_split(name: str, split: str | None = None, *, streaming: bool = False) -> Any:
    cfg = DATASET_CFG[name]
    split_to_use = split or cfg["split"]
    return load_dataset(
        cfg["hf_name"],
        cfg["subset"],
        split=split_to_use,
        streaming=streaming,
        download_mode="force_redownload",
    )

def build_prompt(dataset_name: str, sample: Dict[str, Any]) -> Tuple[str, str]:
    cfg = DATASET_CFG[dataset_name]

    if dataset_name == "mmlu":
        prompt = cfg["prompt_template"].format(
            topic=sample.get("subject", ""),
            question=sample["question"],
            choice_a=sample["choices"][0],
            choice_b=sample["choices"][1],
            choice_c=sample["choices"][2],
            choice_d=sample["choices"][3],
        )
        answer = cfg["labels"][sample["answer"]]

    elif dataset_name == "rotten_tomatoes":
        prompt = cfg["prompt_template"].format(review=sample["text"])
        answer = "positive" if sample["label"] == 1 else "negative"

    elif dataset_name == "tweetqa":
        prompt = cfg["prompt_template"].format(
            tweet=sample["Tweet"],
            question=sample["Question"],
        )
        answer = sample["Answer"][0] if sample["Answer"] else ""

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    return prompt, answer

class ModelWrapper:
    def __init__(self, name: str, fp16: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and fp16:
            self.model = HookedTransformer.from_pretrained(
                name,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
        elif self.device == "cuda":
            self.model = HookedTransformer.from_pretrained(name, device_map="auto")
        else:
            self.model = HookedTransformer.from_pretrained(name, device="cpu")
        self.tokenizer = self.model.tokenizer

    def to_tokens(self, text: Sequence[str] | str, *, prepend_bos: bool = False):
        return self.model.to_tokens(
            text, prepend_bos=prepend_bos, move_to_device=True
        )

    def to_string(self, ids: torch.Tensor) -> str:
        return self.model.to_string(ids)

    @property
    def W_U(self) -> torch.Tensor:
        return self.model.W_U

def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)

@torch.no_grad()
def run_example(model: ModelWrapper, texts: List[str]) -> Tuple[torch.Tensor, List[List[float]]]:
    toks = model.to_tokens(texts, prepend_bos=True)
    final_logits = model.model(toks)[:, -1, :].detach()

    w_u = model.W_U
    num_layers = model.model.cfg.n_layers
    sims = torch.empty(len(texts), num_layers, device="cpu")

    def _hook(layer_idx: int):
        def record(resid_post, hook):
            last = resid_post[:, -1, :]
            sims[:, layer_idx] = cosine(last @ w_u, final_logits).cpu()

        return record

    hooks = [(f"blocks.{i}.hook_resid_post", _hook(i)) for i in range(num_layers)]
    model.model.run_with_hooks(toks, fwd_hooks=hooks, return_type=None)
    return final_logits.cpu(), sims.tolist()

@torch.no_grad()
def greedy_generate(model: ModelWrapper, prompts: List[str], *, max_new_tokens: int = 64) -> List[str]:
    toks = model.to_tokens(prompts, prepend_bos=True)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        verbose=False,
    )

    gen = model.model.generate(toks, **gen_kwargs)
    gen_ids = gen[:, toks.shape[1]:]
    return [model.to_string(ids) for ids in gen_ids]

def build_history(
    task: str, samples: Sequence[Dict[str, Any]], idx: int, history_len: int
) -> str:
    turns: List[str] = []
    cfg = DATASET_CFG[task]
    for j in range(history_len):
        sample = samples[(idx + j + 1) % len(samples)]
        prompt, answer = build_prompt(task, sample)
        suffix = cfg["answer_suffix"]
        answer_text = f" <Answer>{answer}{suffix}" if suffix else f" {answer}"
        turns.append(f"User: {prompt}\nAssistant:{answer_text}")
    return "\n\n".join(turns)
