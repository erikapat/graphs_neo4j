# sentiment_utils.py
from __future__ import annotations
import warnings, re, numpy as np
from typing import List

def _heuristic(texts: List[str]) -> np.ndarray:
    pos = {"good","great","excel","excellent","amazing","happy","satisfied","recommend","love","quick",
           "rápido","bueno","genial","excelente","feliz","satisfecho","recomiendo","me encanta","ágil","rápida"}
    neg = {"bad","terrible","awful","angry","sad","poor","slow","issue","problem","fraud","scam","claim denied",
           "malo","terrible","horrible","enojado","triste","lento","problema","fraude","estafa","rechazado","denegado"}
    out = []
    for t in texts:
        tt = (t or "").lower()
        p = sum(w in tt for w in pos); n = sum(w in tt for w in neg)
        out.append(0.0 if p==0 and n==0 else (p-n)/max(1,(p+n)))
    return np.array(out, dtype=float)

def predict_sentiment(texts: List[str], model_name: str) -> np.ndarray:
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name, use_safetensors=True)
        mdl.eval()
        scores = []
        with torch.no_grad():
            for t in texts:
                inputs = tok(t, truncation=True, padding=True, return_tensors="pt")
                logits = mdl(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                neg, neu, pos = probs.tolist()
                scores.append(pos - neg)
        return np.array(scores, dtype=float)
    except Exception as e:
        warnings.warn(f"Sentiment model unavailable; using heuristic. Details: {e}")
        return _heuristic(texts)
