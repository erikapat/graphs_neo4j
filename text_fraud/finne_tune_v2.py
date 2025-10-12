from __future__ import annotations
# ============================================================
# SetFit + label-aware topics (UMAP supervised, robust training)
# - Trains a SetFit classifier on XLM-R
# - If contrastive training fails (e.g., frozen grads), falls back to head-only
# - Encodes string labels -> integer category codes for UMAP supervision
# - Saves predictions, 2D UMAP, and topic words
# ============================================================

import os
import json
import warnings
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd


# -------------------------------
# Config
# -------------------------------
@dataclass
class Config:
    # Data
    input_csv: str = "data/synth/claims_llm.csv"   # columns: Description/text, Fraud_Label/label_type
    output_dir: str = "./artifacts"

    # Encoder / SetFit
    encoder_model: str = "xlm-roberta-base"
    setfit_num_epochs: int = 2
    setfit_batch_size: int = 16
    setfit_learning_rate: float = 2e-5
    setfit_num_iterations: int = 1
    max_length: int = 256

    # IMPORTANT: Unfreeze by default so SetFit can do contrastive training
    freeze_encoder: bool = False

    # Split
    test_size: float = 0.2
    random_state: int = 42

    # Topics (UMAP/HDBSCAN/BERTopic)
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.0
    hdbscan_min_cluster_size: int = 20
    hdbscan_min_samples: int = 5
    kmeans_fallback_k: int = 12

    # Runtime
    force_cpu: bool = True
    torch_num_threads: int = 1


# -------------------------------
# Runtime/env
# -------------------------------
def configure_runtime(cfg: Config):
    if cfg.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        import torch
        torch.set_num_threads(max(1, int(cfg.torch_num_threads)))
    except Exception:
        pass


# -------------------------------
# Data
# -------------------------------
def basic_clean(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\n", " ").strip()
    s = " ".join(s.split())
    return s

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
        raise ValueError("Input must contain 'text' (or 'Description').")
    if "label_type" not in df.columns:
        warnings.warn("No 'label_type' column found. Creating a single dummy class.")
        df["label_type"] = "unknown"

    df["text_clean"] = df["text"].apply(basic_clean)
    df["label_type"] = df["label_type"].astype(str)
    return df


# -------------------------------
# SetFit training (robust)
# -------------------------------
def _train_head_only(model, X_train, y_train, X_test, y_test, id2label):
    """Train only the sklearn head on frozen embeddings (robust fallback)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    train_embs = model.model_body.encode(X_train.tolist(), show_progress_bar=True, normalize_embeddings=True)
    test_embs  = model.model_body.encode(X_test.tolist(),  show_progress_bar=True, normalize_embeddings=True)

    clf = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="auto")
    clf.fit(train_embs, y_train.astype(int).tolist())
    preds = clf.predict(test_embs)
    metrics = {"f1_weighted": float(f1_score(y_test.astype(int).tolist(), preds, average="weighted"))}

    # Attach the trained head back to the SetFit model
    model.model_head = clf
    return model, metrics

def train_setfit_classifier(df: pd.DataFrame, cfg: Config):
    from sklearn.model_selection import train_test_split
    from datasets import Dataset
    try:
        from setfit.losses import CosineSimilarityLoss
    except Exception:
        from sentence_transformers.losses import CosineSimilarityLoss
    from setfit import SetFitModel, SetFitTrainer

    X_train, X_test, y_train_str, y_test_str = train_test_split(
        df["text_clean"], df["label_type"],
        test_size=cfg.test_size, random_state=cfg.random_state, stratify=df["label_type"]
    )

    label2id = {l: i for i, l in enumerate(sorted(df["label_type"].unique()))}
    id2label = {i: l for l, i in label2id.items()}
    y_train = y_train_str.map(label2id)
    y_test  = y_test_str.map(label2id)

    ds_train = Dataset.from_pandas(pd.DataFrame({"text": X_train, "label": y_train.astype(int)}))
    ds_test  = Dataset.from_pandas(pd.DataFrame({"text": X_test,  "label": y_test.astype(int)}))

    model = SetFitModel.from_pretrained(cfg.encoder_model, labels=list(range(len(label2id))))
    model.to("cpu")

    # Only freeze if explicitly requested
    if cfg.freeze_encoder:
        try:
            trf = model.model_body[0]
            hf = getattr(trf, "auto_model", None)
            if hf is not None:
                for p in hf.parameters():
                    p.requires_grad = False
        except Exception as e:
            warnings.warn(f"Could not freeze encoder: {e}")

    # Try full SetFit contrastive training
    trainer = SetFitTrainer(
        model=model,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        loss_class=CosineSimilarityLoss,
        num_iterations=cfg.setfit_num_iterations,
        num_epochs=cfg.setfit_num_epochs,
        batch_size=cfg.setfit_batch_size,
        learning_rate=cfg.setfit_learning_rate,
    )

    try:
        trainer.train()
        metrics = trainer.evaluate()
        return model, label2id, id2label, metrics
    except RuntimeError as e:
        # Typical when everything was frozen or grad graph is broken
        warnings.warn(f"SetFit contrastive training failed, falling back to head-only. Details: {e}")
        model, metrics = _train_head_only(model, X_train, y_train, X_test, y_test, id2label)
        return model, label2id, id2label, metrics


# -------------------------------
# Predict with SetFit (probabilities)
# -------------------------------
def predict_probs_setfit(model, texts: List[str], id2label: Dict[int, str]) -> List[Dict[str, float]]:
    embs = model.model_body.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    head = model.model_head
    if hasattr(head, "predict_proba"):
        probs = head.predict_proba(embs)
    else:
        scores = head.decision_function(embs)
        exps = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs = exps / (exps.sum(axis=1, keepdims=True) + 1e-12)
    out = []
    for row in probs:
        out.append({id2label[i]: float(row[i]) for i in range(len(id2label))})
    return out


# -------------------------------
# Label-aware topics (UMAP supervised fix)
# -------------------------------
def label_aware_topics(texts: List[str], embs: np.ndarray, labels: List[Any], cfg: Config):
    """
    Supervised UMAP guided by labels; encodes labels to integer codes.
    If only one class, falls back to unsupervised.
    """
    from umap import UMAP
    import hdbscan
    from sklearn.feature_extraction.text import CountVectorizer
    from bertopic import BERTopic
    from sklearn.cluster import KMeans

    # Encode labels -> integer codes if >=2 unique
    y = None
    if labels is not None:
        lab_ser = pd.Series(list(labels)).astype("string")
        nunique = lab_ser.nunique(dropna=True)
        if nunique >= 2:
            y = lab_ser.astype("category").cat.codes.to_numpy()

    reducer = UMAP(
        n_neighbors=cfg.umap_n_neighbors,
        n_components=2,
        min_dist=cfg.umap_min_dist,
        random_state=cfg.random_state,
        target_metric="categorical" if y is not None else "euclidean",
    )
    reduced = reducer.fit_transform(embs, y=y) if y is not None else reducer.fit_transform(embs)

    hdb = hdbscan.HDBSCAN(
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )
    vec = CountVectorizer(ngram_range=(1, 3), min_df=5)

    tm = BERTopic(
        umap_model=None,                 # pre-reduced
        hdbscan_model=hdb,
        vectorizer_model=vec,
        language="multilingual",
        calculate_probabilities=True,
        min_topic_size=cfg.hdbscan_min_cluster_size,
        verbose=False,
    )
    topics, probs = tm.fit_transform(texts, reduced)

    # Fallback if too many outliers
    outlier_rate = (np.array(topics) == -1).mean()
    if outlier_rate > 0.4:
        km = KMeans(n_clusters=cfg.kmeans_fallback_k, random_state=cfg.random_state)
        hard = km.fit_predict(reduced)
        tm.set_topics(hard)
        topics, probs = hard, None

    return tm, topics, probs, reduced


# -------------------------------
# Pipeline
# -------------------------------
def run_pipeline(cfg: Optional[Config] = None) -> Dict[str, Any]:
    cfg = cfg or Config()
    configure_runtime(cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 1) Load
    df = load_data(cfg)

    # 2) Train SetFit (with robust fallback if needed)
    model, label2id, id2label, clf_metrics = train_setfit_classifier(df, cfg)

    # 3) Predict probs on all rows
    df["clf_probs"] = predict_probs_setfit(model, df["text_clean"].tolist(), id2label)
    df["pred_label"] = df["clf_probs"].apply(lambda d: max(d, key=d.get) if d else None)
    df["pred_conf"]  = df["clf_probs"].apply(lambda d: max(d.values()) if d else 0.0)

    # 4) Embeddings (for topics)
    embs = model.model_body.encode(df["text_clean"].tolist(), show_progress_bar=True, normalize_embeddings=True)

    # 5) Label-aware topics
    tm, topics, probs_topics, reduced_2d = label_aware_topics(
        texts=df["text_clean"].tolist(),
        embs=embs,
        labels=df["label_type"].tolist(),   # raw strings OK; encoded internally
        cfg=cfg
    )
    df["topic_id"] = topics
    df["topic_prob"] = probs_topics.max(axis=1) if isinstance(probs_topics, np.ndarray) else 1.0

    # 6) Save artifacts
    pred_cols = ["text", "label_type", "pred_label", "pred_conf", "topic_id", "topic_prob"]
    out_pred = df[pred_cols].copy()
    out_pred_path = os.path.join(cfg.output_dir, "predictions_setfit_umap.csv")
    out_pred.to_csv(out_pred_path, index=False)

    red = pd.DataFrame(reduced_2d, columns=["umap_x", "umap_y"])
    red["label_type"] = df["label_type"].values
    red["topic_id"] = df["topic_id"].values
    red_path = os.path.join(cfg.output_dir, "reduced_2d.csv")
    red.to_csv(red_path, index=False)

    topic_info = tm.get_topic_info()
    topic_words_path = os.path.join(cfg.output_dir, "topic_info.csv")
    topic_info.to_csv(topic_words_path, index=False)

    print("\n=== Artifacts ===")
    print("Predictions: ", os.path.abspath(out_pred_path))
    print("UMAP 2D:     ", os.path.abspath(red_path))
    print("Topic info:  ", os.path.abspath(topic_words_path))
    print("Metrics:     ", clf_metrics)

    return {
        "metrics": clf_metrics,
        "artifacts": {
            "predictions_csv": out_pred_path,
            "reduced_2d_csv": red_path,
            "topic_info_csv": topic_words_path
        }
    }


if __name__ == "__main__":
    print(json.dumps(run_pipeline(Config()), indent=2))



