#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_report_old.py

Reads artifacts produced by all_v2.py and creates:
- 2D scatter of documents via TF-IDF + TruncatedSVD (colored by topic, sized by risk).
- Top topics by average risk (bar chart).
- Blacklist preview tables (overall + per topic label if present).
- A compact HTML report linking all outputs.

No transformers / torch required. Only pandas, numpy, scikit-learn, matplotlib.

Usage:
  python generate_report_old.py --artifacts ./artifacts --out ./artifacts/report

If --out is omitted, defaults to <artifacts>/report
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import warnings
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

# Matplotlib backend-safe import (non-interactive)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------
# Utilities
# ---------------------------

def _abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _load_pred_table(artifacts_dir: str) -> pd.DataFrame:
    """Load predictions.{parquet|csv} written by all_v2.py."""
    p_parquet = os.path.join(artifacts_dir, "predictions.parquet")
    p_csv     = os.path.join(artifacts_dir, "predictions.csv")

    if os.path.isfile(p_parquet):
        try:
            df = pd.read_parquet(p_parquet)
            return df
        except Exception as e:
            warnings.warn(f"Could not read Parquet ({p_parquet}): {e}. Falling back to CSV...")

    if os.path.isfile(p_csv):
        return pd.read_csv(p_csv)

    raise FileNotFoundError(
        f"Could not find predictions in {artifacts_dir} "
        f"(looked for predictions.parquet or predictions.csv). "
        f"Run all_v2.py first."
    )


def _maybe_load_blacklist(artifacts_dir: str) -> Optional[pd.DataFrame]:
    p_csv = os.path.join(artifacts_dir, "blacklist_candidates.csv")
    if os.path.isfile(p_csv):
        try:
            df = pd.read_csv(p_csv)
            # support either the advanced extractor outputs or simple c-TFIDF outputs
            # normalize expected columns
            if "cluster" in df.columns and "term" in df.columns and "kind" in df.columns:
                # advanced format (cluster, term, kind, ...)
                # create a 'label' column compatible with older reports
                df["label"] = df["cluster"].astype(str)
            elif "label" not in df.columns and "cluster" in df.columns:
                df["label"] = df["cluster"].astype(str)
            return df
        except Exception as e:
            warnings.warn(f"Could not read blacklist ({p_csv}): {e}")
            return None
    return None


def _safe_series(x, default):
    try:
        return x
    except Exception:
        return default


def _coerce_topic(x):
    # safely coerce topic ID to int (or -1)
    try:
        return int(x)
    except Exception:
        return -1


# ---------------------------
# Lightweight document projection
# ---------------------------

def compute_2d_projection(df: pd.DataFrame,
                          text_col: str = "text",
                          max_features: int = 5000,
                          n_components: int = 2,
                          random_state: int = 42) -> np.ndarray:
    """
    Project documents to 2D using TF-IDF -> TruncatedSVD (LSA).
    This avoids heavy embeddings; fast & sufficient for overview plots.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import Normalizer

    texts = df[text_col].astype(str).fillna("").tolist()
    if len(texts) == 0:
        raise ValueError("Empty text column; cannot compute projection.")

    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)
    X = tfidf.fit_transform(texts)

    n_components = min(n_components, max(2, min(X.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    X_reduced = svd.fit_transform(X)
    X_reduced = Normalizer(copy=False).fit_transform(X_reduced)
    return X_reduced  # shape (n_samples, 2)


# ---------------------------
# Plotting
# ---------------------------

def plot_scatter_2d(df: pd.DataFrame,
                    coords: np.ndarray,
                    out_png: str,
                    color_col: str = "topic_id",
                    size_col: str = "risk_score",
                    title: str = "Documents (2D projection)") -> None:
    """
    Scatter colored by topics, sized by risk score.
    """
    x = coords[:, 0]
    y = coords[:, 1]

    # sanitize columns
    color_vals = df[color_col].map(_coerce_topic) if color_col in df.columns else pd.Series([-1]*len(df))
    size_vals = df[size_col] if size_col in df.columns else pd.Series([0.5]*len(df))
    size_vals = np.clip(np.array(size_vals, dtype=float), 0.0, None)
    sizes = 30.0 + 120.0 * (size_vals / (size_vals.max() + 1e-9))  # 30..150

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(x, y, c=color_vals, s=sizes, alpha=0.7, edgecolors="none")
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    cbar = plt.colorbar(sc)
    cbar.set_label(color_col)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_top_topics_by_risk(df: pd.DataFrame,
                            out_png: str,
                            topic_col: str = "topic_id",
                            risk_col: str = "risk_score",
                            top_k: int = 15,
                            title: str = "Top topics by average risk") -> None:
    if topic_col not in df.columns or risk_col not in df.columns:
        warnings.warn("Missing columns for risk-by-topic chart; skipping.")
        return
    df2 = df.copy()
    df2[topic_col] = df2[topic_col].map(_coerce_topic)
    agg = (df2.groupby(topic_col, as_index=False)[risk_col]
           .mean()
           .rename(columns={risk_col: "avg_risk"}))
    agg = agg.sort_values("avg_risk", ascending=False).head(top_k)

    plt.figure(figsize=(10, 6))
    plt.bar(agg[topic_col].astype(str), agg["avg_risk"])
    plt.title(title)
    plt.xlabel("Topic ID")
    plt.ylabel("Average risk")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# ---------------------------
# HTML report builder
# ---------------------------

def _html_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_html_report(out_dir: str,
                      artifacts_dir: str,
                      images: List[Tuple[str, str]],
                      tables: List[Tuple[str, pd.DataFrame]],
                      extras: dict) -> str:
    """Write a minimal HTML summary pointing to PNGs and CSV snippets."""
    html_path = os.path.join(out_dir, "index.html")

    def table_to_html(df: pd.DataFrame, max_rows: int = 50) -> str:
        if df is None or df.empty:
            return "<em>(empty)</em>"
        show = df.head(max_rows).copy()
        # escape cells
        show = show.applymap(lambda v: _html_escape(str(v)))
        return show.to_html(index=False, escape=False)

    parts = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>"
                 "<title>Text Fraud Report</title>"
                 "<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px;} "
                 "h1,h2{margin-top:1.2em;} img{max-width:100%;height:auto;border:1px solid #ddd;} "
                 "code,pre{background:#f5f5f5;padding:2px 4px;}</style></head><body>")

    parts.append("<h1>Text Fraud Report</h1>")
    parts.append(f"<p><b>Artifacts dir:</b> {_html_escape(_abs(artifacts_dir))}</p>")
    if extras.get("metrics"):
        parts.append("<h2>Classifier metrics</h2>")
        parts.append("<pre>" + _html_escape(json.dumps(extras["metrics"], indent=2)) + "</pre>")

    if images:
        parts.append("<h2>Visualizations</h2>")
        for caption, rel_path in images:
            parts.append(f"<h3>{_html_escape(caption)}</h3>")
            parts.append(f"<img src='{_html_escape(os.path.basename(rel_path))}' alt='{_html_escape(caption)}'/>")

    if tables:
        parts.append("<h2>Tables (preview)</h2>")
        for caption, df in tables:
            parts.append(f"<h3>{_html_escape(caption)}</h3>")
            parts.append(table_to_html(df, max_rows=50))

    parts.append("</body></html>")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

    return html_path


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate visual report from pipeline artifacts.")
    ap.add_argument("--artifacts", type=str, default="./artifacts", help="Directory with predictions.csv/parquet and blacklist_candidates.csv")
    ap.add_argument("--out", type=str, default=None, help="Output directory for the report (default: <artifacts>/report)")
    ap.add_argument("--max_features", type=int, default=5000, help="Max TF-IDF features for 2D projection")
    ap.add_argument("--top_topics", type=int, default=15, help="Top-K topics by mean risk to plot")
    args = ap.parse_args()

    artifacts_dir = _abs(args.artifacts)
    out_dir = _abs(args.out or os.path.join(artifacts_dir, "report"))
    _mkdir(out_dir)

    # Load inputs
    df = _load_pred_table(artifacts_dir)
    bl = _maybe_load_blacklist(artifacts_dir)

    # Minimal sanity: required columns in predictions
    for col in ["text", "topic_id", "risk_score"]:
        if col not in df.columns:
            raise KeyError(f"Predictions table must include column '{col}'. Found columns: {list(df.columns)}")

    # Compute light 2D projection
    coords = compute_2d_projection(df, text_col="text", max_features=args.max_features)

    # Build plots
    scatter_png = os.path.join(out_dir, "scatter_2d.png")
    plot_scatter_2d(df, coords, scatter_png, color_col="topic_id", size_col="risk_score",
                    title="Documents (TF-IDF → LSA 2D) — color: topic_id, size: risk_score")

    topics_png = os.path.join(out_dir, "top_topics_by_risk.png")
    plot_top_topics_by_risk(df, topics_png, topic_col="topic_id", risk_col="risk_score", top_k=args.top_topics)

    # Prepare table previews
    tables = []
    # 1) small sample of predictions
    pred_preview_cols = [c for c in ["id", "pred_label", "pred_conf", "topic_id", "topic_prob", "sentiment", "risk_score", "recommendation", "text"] if c in df.columns]
    tables.append(("Predictions (preview)", df[pred_preview_cols].head(50).copy()))

    # 2) blacklist previews, if present
    images = [
        ("2D document scatter", scatter_png),
        ("Top topics by avg risk", topics_png),
    ]

    if bl is not None and not bl.empty:
        # Write full blacklist copies
        bl_all_path = os.path.join(out_dir, "blacklist_all.csv")
        try:
            bl.to_csv(bl_all_path, index=False)
        except Exception as e:
            warnings.warn(f"Could not write blacklist_all.csv: {e}")

        # Create a "top per label" view if label column exists; otherwise top overall
        if "label" in bl.columns and "enrichment" in bl.columns:
            top_per_label = (bl.sort_values(["label", "enrichment"], ascending=[True, False])
                             .groupby("label", as_index=False)
                             .head(50))
            bl_top_path = os.path.join(out_dir, "blacklist_top50_per_label.csv")
            try:
                top_per_label.to_csv(bl_top_path, index=False)
            except Exception as e:
                warnings.warn(f"Could not write blacklist_top50_per_label.csv: {e}")

            tables.append(("Blacklist (top 50 per label)", top_per_label.head(200).copy()))
        else:
            # Fallback: just top 200 overall by any 'score-like' column we can find
            score_col = "enrichment" if "enrichment" in bl.columns else ( "score" if "score" in bl.columns else None )
            if score_col is not None:
                bl_top = bl.sort_values(score_col, ascending=False).head(200)
            else:
                bl_top = bl.head(200)
            tables.append(("Blacklist (preview)", bl_top.copy()))
    else:
        warnings.warn("No blacklist_candidates.csv found; skipping blacklist section.")

    # Pull metrics if present from pipeline JSON (optional)
    metrics_json = os.path.join(artifacts_dir, "metrics.json")
    metrics = None
    if os.path.isfile(metrics_json):
        try:
            with open(metrics_json, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            pass

    # Write HTML report
    html_path = write_html_report(out_dir, artifacts_dir, images, tables, extras={"metrics": metrics})

    # Console summary
    print("\n=== Report generated ===")
    print("Artifacts dir:   ", artifacts_dir)
    print("Report dir:      ", out_dir)
    print("2D scatter PNG:  ", scatter_png)
    print("Top topics PNG:  ", topics_png)
    if bl is not None and not bl.empty:
        print("Blacklist CSV(s):")
        print("  - Full (if writable): ", os.path.join(out_dir, "blacklist_all.csv"))
        print("  - Top50/label (if available): ", os.path.join(out_dir, "blacklist_top50_per_label.csv"))
    print("HTML summary:    ", html_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Print a clear message and re-raise for non-zero exit
        print(f"[generate_report] ERROR: {e}", file=sys.stderr)
        raise

