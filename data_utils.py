import random
from datasets import load_dataset

dataset_config = {
    "mmlu": {
        "hf_name": "lukaemon/mmlu",
        "subset": "abstract_algebra",
        "prompt_prefix": (
            "You have a multiple choice question on Abstract Algebra. "
            "Only one of the options is correct: A, B, C, or D. "
            "Give your answer in the following format with the tags provided: "
            "<Answer> </Answer>. Please read the following question and options and answer the question.\n"
        ),
        "metric": "accuracy",
        "labels": ["A", "B", "C", "D"],
        "answer_tokens": [" A", " B", " C", " D"],
    },
    "rotten_tomatoes": {
        "hf_name": "cornell-movie-review-data/rotten_tomatoes",
        "subset": None,
        "prompt_prefix": (
            "Classify the following movie review as Positive or Negative. "
            "Give your answer in the format <Answer> </Answer>.\nReview:"
        ),
        "metric": "accuracy",
        "labels": ["Negative", "Positive"],
        "answer_tokens": [" Negative", " Positive"],
    },
    "cnn_dailymail": {
        "hf_name": "abisee/cnn_dailymail",
        "subset": "3.0.0",
        "prompt_prefix": (
            "Summarize the following article in 2-3 sentences. "
            "Provide the summary inside the tags <Answer> </Answer>.\n"
        ),
        "metric": "rouge",
        "labels": [],
        "answer_tokens": [],
    },
}

def load_split(name, split="test", limit=None, shuffle=True, seed=42):
    cfg = dataset_config[name]
    ds = load_dataset(cfg["hf_name"], cfg["subset"], split=split)
    
    # Shuffle the dataset for randomness
    if shuffle:
        ds = ds.shuffle(seed=seed)
    
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds

def build_prompt(dataset_name, sample):
    cfg = dataset_config[dataset_name]
    pfx = cfg["prompt_prefix"]
    if dataset_name == "mmlu":
        stem = sample.get("question", sample.get("input", ""))
        choices = sample["choices"] if "choices" in sample else [sample[k] for k in ["A", "B", "C", "D"]]
        options = "\n".join(f"{c}. {o}" for c, o in zip(["A", "B", "C", "D"], choices))
        prompt = pfx + stem + "\n" + options + "\nAnswer:"
        answer = sample.get("answer", sample.get("target", "A"))
    elif dataset_name == "rotten_tomatoes":
        prompt = pfx + " " + sample["text"] + "\nAnswer:"
        answer = "Positive" if sample["label"] == 1 else "Negative"
    else:
        prompt = pfx + sample["article"] + "\nAnswer:"
        answer = sample["highlights"]
    return prompt, answer

def build_histories(max_len=10):
    histories = []
    for h in range(1, max_len + 1):
        for j in range(1, h + 1):
            i = h - j
            seq = ["B"] * j + ["A"] * i
            histories.append(seq)
    return histories
