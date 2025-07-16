from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        dtype = torch.float16 if self.device == "cuda" and fp16 else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
            output_hidden_states=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model.to(self.device)

    def to_tokens(self, text: Sequence[str] | str, *, prepend_bos: bool = False) -> torch.Tensor:
        enc = self.tokenizer(
            list(text) if isinstance(text, (list, tuple)) else [text],
            return_tensors="pt",
            add_special_tokens=not prepend_bos,
            padding=True,
        )
        return enc.input_ids.to(self.device)

    def to_string(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def lm_head_weight(self) -> torch.Tensor:
        return self.model.lm_head.weight

def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)

@torch.no_grad()
def run_example(model: ModelWrapper, texts: List[str]) -> Tuple[torch.Tensor, List[List[float]]]:
    toks = model.to_tokens(texts, prepend_bos=True)

    outputs = model.model(input_ids=toks)
    logits = outputs.logits[:, -1, :].detach().cpu()
    hidden_states = outputs.hidden_states
    num_layers = len(hidden_states) - 1

    sims: List[List[float]] = []
    lm_w = model.lm_head_weight.to("cpu")

    for i in range(len(texts)):
        final_logit = logits[i]
        sims.append([])

        for layer in range(1, num_layers + 1):
            h = hidden_states[layer][i, -1, :].to("cpu")
            layer_logits = h @ lm_w.T
            sims[-1].append(
                cosine(layer_logits.unsqueeze(0), final_logit.unsqueeze(0)).item()
            )

    return logits, sims

@torch.no_grad()
def greedy_generate(
    model: ModelWrapper,
    prompts: List[str],
    *,
    max_new_tokens: int = 64
) -> List[str]:
    toks = model.to_tokens(prompts, prepend_bos=True)

    gen = model.model.generate(
        input_ids=toks,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=False,
    )

    generated: List[str] = []
    for i, inp in enumerate(toks):
        new_tokens = gen[i, inp.shape[0]:]
        generated.append(model.to_string(new_tokens))

    return generated

def build_history(
    task: str,
    samples: Sequence[Dict[str, Any]],
    idx: int,
    history_len: int
) -> str:
    turns: List[str] = []
    cfg = DATASET_CFG[task]

    for j in range(history_len):
        sample = samples[(idx + j + 1) % len(samples)]
        prompt, answer = build_prompt(task, sample)
        suffix = cfg["answer_suffix"]

        answer_text = (
            f" <Answer>{answer}{suffix}" if suffix else f" {answer}"
        )

        turns.append(f"User: {prompt}\nAssistant:{answer_text}")

    return "\n\n".join(turns)
