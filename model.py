import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import DATASET_CFG

class ModelWrapper:
    def __init__(self, name, fp16=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if fp16 and self.device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=dtype
        ).to(self.device)

        self.num_layers = self.model.config.num_hidden_layers
        print(f"[DEBUG] Loaded model {name} on {self.device} with dtype {dtype}")

        # --- DEBUG: Print answer token ids for each dataset ---
        for ds_name, cfg in DATASET_CFG.items():
            if cfg["answer_tokens"]:
                print(f"[DEBUG] {ds_name} answer_tokens:")
                for tok in cfg["answer_tokens"]:
                    ids = self.tokenizer.encode(tok, add_special_tokens=False)
                    print(f"  '{tok}' -> {ids}")
                    if len(ids) != 1:
                        print(f"  [WARNING] Answer token '{tok}' is not a single token for this tokenizer!")

    def to_tokens(self, text, *, prepend_bos=False):
        toks = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=not prepend_bos,
        ).input_ids.to(self.device)
        print(f"[DEBUG] to_tokens: {text[:60]}... -> shape {toks.shape}")
        return toks

    def to_string(self, ids):
        s = self.tokenizer.decode(ids, skip_special_tokens=True)
        print(f"[DEBUG] to_string: {ids} -> {s}")
        return s

    @property
    def W_U(self):
        return self.model.get_output_embeddings().weight

def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)

@torch.no_grad()
def run_example(model_wrapper, texts):
    toks = model_wrapper.to_tokens(texts, prepend_bos=True)
    output = model_wrapper.model(toks)
    final_logits = output.logits[:, -1, :].detach()
    sims = torch.zeros(len(texts), model_wrapper.num_layers, device="cpu")
    handles = []
    w_u = model_wrapper.W_U

    def make_hook(idx):
        # Hook to capture the output of each layer for the last token
        def hook(module, inp, out):
            last_token = out[0][:, -1, :]
            # Compute cosine similarity between projected hidden state and final logits
            sims[:, idx] = cosine(last_token @ w_u.T, final_logits).cpu()
        return hook

    layers = getattr(model_wrapper.model.model, "layers", None)
    if layers is None:
        raise AttributeError("Cannot find decoder layers in model structure.")

    # Register hooks for each layer
    handles = [block.register_forward_hook(make_hook(i)) for i, block in enumerate(layers)]
    
    _ = model_wrapper.model(toks)
    
    for h in handles: 
        h.remove()
    
    return final_logits.cpu(), sims.tolist()

@torch.no_grad()
def greedy_generate(model_wrapper, prompts, *, max_new_tokens=64):
    toks = model_wrapper.to_tokens(prompts, prepend_bos=True)
    print(f"[DEBUG] greedy_generate toks shape: {toks.shape}")
    gen = model_wrapper.model.generate(
        toks,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    gen_ids = gen[:, toks.shape[1]:]
    print(f"[DEBUG] greedy_generate gen_ids: {gen_ids}")
    results = [model_wrapper.to_string(ids) for ids in gen_ids]
    print(f"[DEBUG] greedy_generate results: {results}")
    return results
