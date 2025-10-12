from __future__ import annotations
import os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from encoder import load_encoder
from config import Config

def _load_preds(art_dir: str) -> pd.DataFrame:
    p_csv  = os.path.join(art_dir, "predictions.csv")
    p_parq = os.path.join(art_dir, "predictions.parquet")
    if os.path.exists(p_parq):
        return pd.read_parquet(p_parq)
    if os.path.exists(p_csv):
        return pd.read_csv(p_csv)
    raise FileNotFoundError(f"predictions.* not found in {art_dir}")

def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    df = _load_preds(cfg.output_dir)
    print(f"Loaded {len(df)} rows")

    # --- quick stats
    print("\n== Stats ==")
    print(df["pred_label"].value_counts(dropna=False).head(20))
    print("\nRisk summary:\n", df["risk_score"].describe())

    # --- plots dir
    pdir = os.path.join(cfg.output_dir, "plots")
    os.makedirs(pdir, exist_ok=True)

    # Risk histogram
    plt.figure()
    df["risk_score"].hist(bins=30)
    plt.title("Risk score distribution")
    plt.xlabel("risk_score"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(os.path.join(pdir, "risk_hist.png")); plt.close()

    # Topic counts
    plt.figure()
    df["topic_id"].value_counts().sort_index().plot(kind="bar")
    plt.title("Topic sizes")
    plt.xlabel("topic_id"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(os.path.join(pdir, "topic_sizes.png")); plt.close()

    # 2D scatter via t-SNE on encoder embeddings (sample if large)
    sample = df.sample(n=min(1500, len(df)), random_state=cfg.random_state)
    enc = load_encoder(cfg.encoder_model, cfg.max_length, device="cpu")
    embs = enc.encode(sample["text"].astype(str).tolist(), batch_size=32, normalize=True)
    ts = TSNE(n_components=2, init="pca", random_state=cfg.random_state, perplexity=30, n_iter=1000)
    xy = ts.fit_transform(embs)
    plt.figure()
    sc = plt.scatter(xy[:,0], xy[:,1], c=sample["topic_id"], s=9, alpha=0.8)
    plt.title("t-SNE of embeddings colored by topic")
    plt.tight_layout(); plt.savefig(os.path.join(pdir, "tsne_topics.png")); plt.close()

    # Top terms per topic (from BERTopic model folders is heavier; use blacklist_top50 for a simple view)
    bl_top = os.path.join(cfg.output_dir, "blacklist_top50.csv")
    if os.path.exists(bl_top):
        terms = pd.read_csv(bl_top)
        print("\n== Blacklist top-terms (head) ==")
        print(terms.groupby("label").head(5)[["label","term","score"]].to_string(index=False))
    else:
        print("\nNo blacklist_top50.csv found.")

    print("\nReport images written to:", os.path.abspath(pdir))

if __name__ == "__main__":
    main()
