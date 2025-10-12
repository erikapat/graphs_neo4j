# pipeline.py
from __future__ import annotations
import os, json, warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from data_prep import configure_runtime, load_data
from train_setfit import train_setfit_classifier
from topics_label_aware import build_topics_label_aware
from sentiment_utils import predict_sentiment
from blacklist_miner import extract_blacklist

@dataclass
class Config:
    # data
    input_csv: str = "data/synth/claims_llm.csv"
    output_dir: str = "./artifacts"

    # SetFit body (XLM-R via SentenceTransformers)
    st_body: str = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
    max_length: int = 256
    setfit_num_epochs: int = 2
    setfit_batch_size: int = 16
    setfit_learning_rate: float = 2e-5
    setfit_num_iterations: int = 1
    unfreeze_last_n: int = 2

    # split/runtime
    test_size: float = 0.2
    random_state: int = 42
    force_cpu: bool = True
    torch_num_threads: int = 1

    # sentiment
    sentiment_model: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    # topics (label-aware)
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.0
    hdbscan_min_cluster_size: int = 20
    hdbscan_min_samples: int = 5
    kmeans_fallback_k: int = 12

    # risk weights
    w_clf: float = 0.6
    w_sent: float = 0.1
    w_topic: float = 0.3

    # blacklist knobs
    min_cluster_freq: int = 3
    max_other_frac: float = 0.05
    dedupe_similarity: float = 0.88

def risk_score_row(clf_probs: Dict[str, float], sentiment_score: float, topic_risk: float, cfg: Config) -> float:
    p_max = max(clf_probs.values()) if clf_probs else 0.0
    s = (sentiment_score + 1) / 2
    return float(cfg.w_clf * p_max + cfg.w_sent * s + cfg.w_topic * topic_risk)

def generate_recommendation(clf_label: str, topic_label: str, score: float) -> str:
    pri = "High" if score > 0.75 else "Medium" if score > 0.5 else "Low"
    return f"Priority {pri}. Route to analyst specializing in '{clf_label}'. Check patterns related to topic '{topic_label}'."

def run_pipeline(cfg: Optional[Config] = None) -> Dict[str, Any]:
    cfg = cfg or Config()
    configure_runtime(cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 1) data
    df = load_data(cfg)   # expects columns: text, text_clean, label_type (str)

    # 2) SetFit (RoBERTa/XLM-R body)
    model, label2id, id2label, clf_metrics = train_setfit_classifier(df, cfg)

    # 3) predict probs all rows
    probs = model.predict_proba(df["text_clean"].tolist())
    df["clf_probs"] = [{id2label[i]: float(p[i]) for i in range(len(id2label))} for p in probs]
    df["pred_label"] = df["clf_probs"].apply(lambda d: max(d, key=d.get) if d else None)
    df["pred_conf"] = df["clf_probs"].apply(lambda d: max(d.values()) if d else 0.0)

    # 4) embeddings from SetFit body -> SentenceTransformer.encode
    embs = model.model_body.encode(df["text_clean"].tolist(), batch_size=32, normalize_embeddings=True)

    # 5) label-aware topics (UMAP with y=labels)
    tm, topics, probs_topics, reduced = build_topics_label_aware(
        texts=df["text_clean"].tolist(),
        embeddings=embs,
        labels=df["label_type"].tolist(),
        cfg=cfg
    )
    df["topic_id"] = topics
    df["topic_prob"] = probs_topics.max(axis=1) if isinstance(probs_topics, np.ndarray) else 1.0

    # 6) sentiment
    df["sentiment"] = predict_sentiment(df["text_clean"].tolist(), cfg.sentiment_model)

    # 7) topic prior from confidence
    topic_prior = df.groupby("topic_id")["pred_conf"].mean()
    topic_prior = (topic_prior - topic_prior.min()) / (topic_prior.max() - topic_prior.min() + 1e-6)
    df["topic_risk"] = df["topic_id"].map(topic_prior.to_dict()).fillna(0.0)

    # 8) risk score + recs
    df["risk_score"] = [
        risk_score_row(cp, s, tr, cfg)
        for cp, s, tr in zip(df["clf_probs"], df["sentiment"], df["topic_risk"])
    ]
    topic_info = tm.get_topic_info()
    topic_words = {row.Topic: row.Name for _, row in topic_info.iterrows()}
    df["recommendation"] = [
        generate_recommendation(lbl, topic_words.get(tid, str(tid)), rs)
        for lbl, tid, rs in zip(df["pred_label"], df["topic_id"], df["risk_score"])
    ]

    # 9) blacklist from high-risk clusters
    high_clusters = df.loc[df["risk_score"] > df["risk_score"].quantile(0.75), "topic_id"].unique().tolist()
    if len(high_clusters) == 0:
        warnings.warn("No high-risk clusters at P75; falling back to median.")
        high_clusters = df.loc[df["risk_score"] > df["risk_score"].median(), "topic_id"].unique().tolist()
    bl_df = extract_blacklist(
        df,
        target_clusters=high_clusters,
        min_cluster_freq=cfg.min_cluster_freq,
        max_other_frac=cfg.max_other_frac,
        dedupe_similarity=cfg.dedupe_similarity
    )

    # 10) persist
    key_id = "claim_id" if "claim_id" in df.columns else ("Client_ID" if "Client_ID" in df.columns else None)
    id_series = df[key_id] if key_id else pd.Series(range(len(df)))
    df_out = pd.DataFrame({
        "id": id_series,
        "text": df["text"],
        "pred_label": df["pred_label"],
        "pred_conf": df["pred_conf"],
        "topic_id": df["topic_id"],
        "topic_prob": df["topic_prob"],
        "sentiment": df["sentiment"],
        "topic_risk": df["topic_risk"],
        "risk_score": df["risk_score"],
        "recommendation": df["recommendation"],
    })

    os.makedirs(cfg.output_dir, exist_ok=True)
    p_csv = os.path.join(cfg.output_dir, "predictions.csv")
    df_out.to_csv(p_csv, index=False)
    p_parq = os.path.join(cfg.output_dir, "predictions.parquet")
    try:
        df_out.to_parquet(p_parq, index=False)
    except Exception as e:
        warnings.warn(f"Parquet skipped: {e}")
        p_parq = ""

    bl_all = os.path.join(cfg.output_dir, "blacklist_candidates.csv")
    (bl_df if bl_df is not None else pd.DataFrame()).to_csv(bl_all, index=False)
    if bl_df is not None and not bl_df.empty and "label" in bl_df.columns:
        bl_top = (bl_df.groupby("label", as_index=False)
                  .apply(lambda g: g.sort_values("score", ascending=False).head(50))
                  .reset_index(drop=True))
    else:
        bl_top = pd.DataFrame(columns=["label","term","score","freq_cluster","freq_others","cov_cluster","cov_others"])
    bl_top_p = os.path.join(cfg.output_dir, "blacklist_top50.csv")
    bl_top.to_csv(bl_top_p, index=False)

    topics_dir = os.path.join(cfg.output_dir, "bertopic_model")
    tm.save(topics_dir)

    print("\n=== Artifacts ===")
    print("Dir:               ", os.path.abspath(cfg.output_dir))
    print("Predictions CSV:   ", os.path.abspath(p_csv))
    print("Predictions Parq:  ", os.path.abspath(p_parq) if p_parq else "(skipped)")
    print("Blacklist all:     ", os.path.abspath(bl_all))
    print("Blacklist top50:   ", os.path.abspath(bl_top_p))
    print("BERTopic model:    ", os.path.abspath(topics_dir))
    print("Metrics:           ", json.dumps(clf_metrics, indent=2))
    return {"metrics": clf_metrics}

if __name__ == "__main__":
    print(json.dumps(run_pipeline(Config()), indent=2))
