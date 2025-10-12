# data_prep.py
from __future__ import annotations
import os, string, warnings
import pandas as pd
from dataclasses import dataclass

PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def configure_runtime(cfg):
    if getattr(cfg, "force_cpu", True):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_READ_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ENABLE_TELEMETRY", "0")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        import torch
        torch.set_num_threads(max(1, int(getattr(cfg, "torch_num_threads", 1))))
    except Exception:
        pass
    try:
        import numpy as _np
        if int(_np.version.version.split(".")[0]) >= 2:
            warnings.warn("NumPy>=2 detected; if you hit runtime issues, try numpy<2.")
    except Exception:
        pass

def _clean(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.replace("\n", " ").strip()
    s = s.translate(PUNCT_TABLE)
    return " ".join(s.split())

def load_data(cfg) -> pd.DataFrame:
    df = pd.read_csv(cfg.input_csv)
    ren = {}
    if "Description" in df.columns and "text" not in df.columns:
        ren["Description"] = "text"
    if "Fraud_Label" in df.columns and "label_type" not in df.columns:
        ren["Fraud_Label"] = "label_type"
    if ren: df = df.rename(columns=ren)
    if "text" not in df.columns:
        raise ValueError("Input must contain 'text' (or 'Description').")
    df["text_clean"] = df["text"].apply(_clean)
    if "label_type" in df.columns:
        df["label_type"] = df["label_type"].astype(str)
    return df
