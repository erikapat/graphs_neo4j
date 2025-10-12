from __future__ import annotations
from typing import List, Optional
import numpy as np

class RoBertaSentenceEncoder:
    def __init__(self, model_name: str, max_length: int = 256, device: Optional[str] = None):
        from transformers import AutoTokenizer, AutoModel
        import torch
        self.device = device or "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.mdl = AutoModel.from_pretrained(model_name)
        self.mdl.to(self.device).eval()
        self.max_length = max_length
        self._torch = torch

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        return summed / counts

    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        out = []
        with self._torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = self.tok(batch, padding=True, truncation=True,
                                  max_length=self.max_length, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                reps = self.mdl(**inputs).last_hidden_state
                pooled = self._mean_pool(reps, inputs["attention_mask"]).cpu().numpy()
                if normalize:
                    pooled = pooled / (np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-12)
                out.append(pooled)
        return np.concatenate(out, 0)

def load_encoder(name: str, max_len: int, device: Optional[str] = None) -> RoBertaSentenceEncoder:
    return RoBertaSentenceEncoder(name, max_length=max_len, device=device)
