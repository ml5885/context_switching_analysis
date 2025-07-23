from collections import Counter

from datasets import load_dataset

DATASET_CFG = {
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
        "split": "test",
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

def load_split(name, split=None, *, streaming=False):
    cfg = DATASET_CFG[name]
    split_to_use = split or cfg["split"]
    return load_dataset(
        cfg["hf_name"],
        cfg["subset"],
        split=split_to_use,
        streaming=streaming,
        download_mode="force_redownload",
    )

def build_prompt(dataset_name, sample):
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
        prompt = cfg["prompt_template"].format(
            review=sample["text"]
        )
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

def build_history(task, samples, idx, history_len):
    turns = []
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