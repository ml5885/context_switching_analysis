import torch
from transformer_lens import HookedTransformer

class ModelWrapper:
    def __init__(self, name, device="cpu", quantize=False):
        self.model = HookedTransformer.from_pretrained(name, device=device)
        self.tokenizer = self.model.tokenizer
        self.tlens = True

    def to_tokens(self, text, prepend_bos=False):
        return self.model.to_tokens(text, prepend_bos=prepend_bos)

    def to_string(self, ids):
        return self.model.to_string(ids)

    @property
    def W_U(self):
        return self.model.W_U