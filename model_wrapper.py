import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

class ModelWrapper:
    def __init__(self, name, device="cpu", quantize=False):
        self.tlens = False
        try:
            self.model = HookedTransformer.from_pretrained(name, device=device)
            self.tokenizer = self.model.tokenizer
            self.tlens = True
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            dtype = torch.bfloat16 if quantize else None
            self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device)
            self.model.eval()

    def to_tokens(self, text, prepend_bos=False):
        if self.tlens:
            return self.model.to_tokens(text, prepend_bos=prepend_bos)
        
        self.tokenizer.padding_side = "left"
        enc = self.tokenizer(text, return_tensors="pt", padding=True).input_ids.to(self.model.device)
        
        if prepend_bos and self.tokenizer.bos_token_id is not None:
            bos = torch.tensor([[self.tokenizer.bos_token_id]], device=enc.device).expand(enc.shape[0], -1)
            enc = torch.cat([bos, enc], dim=1)
        return enc

    def to_string(self, ids):
        if self.tlens:
            return self.model.to_string(ids)
        if ids.dim() == 2:
            ids = ids[0]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def W_U(self):
        return self.model.W_U if self.tlens else self.model.lm_head.weight
