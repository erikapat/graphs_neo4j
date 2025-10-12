from __future__ import annotations
import string
import pandas as pd
from config import Config

_PUNCT = str.maketrans("", "", string.punctuation)

def basic_clean(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.replace("\n", " ").strip()
    s = s.translate(_PUNCT)
    return " ".join(s.split())

def load_data(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.input_csv)
    rename = {}
    if "Description" in df.columns and "text" not in df.columns:
        rename["Description"] = "text"
    if "Fraud_Label" in df.columns and "label_type" not in df.columns:
        rename["Fraud_Label"] = "label_type"
    if rename:
        df = df.rename(columns=rename)
    if "text" not in df.columns:
        raise ValueError("Input must contain 'text' or 'Description'.")
    df["text_clean"] = df["text"].apply(basic_clean)
    if "label_type" in df.columns:
        df["label_type"] = df["label_type"].astype(str)
    return df
