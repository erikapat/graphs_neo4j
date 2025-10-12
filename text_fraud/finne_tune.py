from __future__ import annotations

# ============================================================
# Multilingual FAST pipeline (no sentence-transformers).
# Encodings via transformers (XLM-RoBERTa) + mean pooling.
# Blacklist miner: entity/NP-aware phrases with strong filtering.
# Always writes CSVs, Parquet if available. Robust to empty outputs.
# ============================================================

import os
import json
import re
import string
import warnings
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd


# -------------------------------
# Configuration
# -------------------------------
@dataclass
class Config:
    # Data
    input_csv: str = "data/synth/claims_llm.csv"
    output_dir: str = "./artifacts"

    # Multilingual RoBERTa backbone (encoder)
    encoder_model: str = "xlm-roberta-base"
    max_length: int = 256

    # Optional heavy seq-classifier (not used in fast mode)
    xlm_roberta_seqcls: str = "xlm-roberta-base"

    # Sentiment model (cardiffnlp multilingual). If repo lacks safetensors
    # on old torch or network issues happen, we fall back to a heuristic.
    sentiment_model: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    # Topic modeling
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.0
    hdbscan_min_cluster_size: int = 20
    hdbscan_min_samples: int = 5
    kmeans_fallback_k: int = 12
    # Optional: force fixed-k KMeans topics for stable IDs
    kmeans_override_k: Optional[int] = None

    # FAST mode (default): train LogisticRegression on frozen embeddings
    fast_linear: bool = False

    # (Unused when fast_linear=True; kept for parity)
    setfit_num_epochs: int = 2
    setfit_batch_size: int = 16
    setfit_learning_rate: float = 2e-5
    setfit_num_iterations: int = 1
    setfit_max_length: int = 256
    unfreeze_last_n: int = 0

    # Data split
    test_size: float = 0.2
    random_state: int = 42

    # Device / runtime
    force_cpu: bool = True
    torch_num_threads: int = 1  # keeps CPU usage sane

    # Risk scoring weights
    w_clf: float = 0.6
    w_sent: float = 0.1
    w_topic: float = 0.3

    # -------- Blacklist mining knobs --------
    spacy_model: str = "xx_sent_ud_sm"
    ban_location_entities: bool = True
    min_cluster_freq: int = 3
    max_other_frac: float = 0.05
    dedupe_similarity: float = 0.88
    regex_drop: Tuple[str, ...] = (
        r"\b(as usual|work as usual)\b",
        r"\b(claim number|policy number)\b",
        r"\bplease (let me know|advise)\b",
        r"\b(read through|last night|two days)\b",
        r"\b(asap|as soon as possible)\b",
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\bnear\s+[a-záéíóúüñ]+\b",
        r"\bon the a\d+\b",
        r"\bon the m\d+\b",
    )
    generic_phrases: Tuple[str, ...] = (
        "please let me know", "as usual", "work as usual", "claim number",
        "policy number", "thank you", "best regards", "let me know",
    )


# -------------------------------
# Runtime / env guards
# -------------------------------
def configure_runtime(cfg: Config):
    if cfg.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    # Quiet tokenizers' fork warning
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # HF Hub timeouts & quiet
    os.environ.setdefault("HF_HUB_READ_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ENABLE_TELEMETRY", "0")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    # Keep torch on few threads to avoid CPU oversubscription on macOS
    try:
        import torch
        torch.set_num_threads(max(1, int(cfg.torch_num_threads)))
    except Exception:
        pass

    # Be lenient with NumPy 2.x: just warn
    try:
        import numpy as _np
        if int(_np.version.version.split(".")[0]) >= 2:
            warnings.warn(
                "NumPy >= 2.x detected. Some compiled wheels (e.g., torch) may be slower/incompatible. "
                "If you see runtime errors, try:  pip install 'numpy<2' --upgrade"
            )
    except Exception:
        pass


# -------------------------------
# Data Loading & Cleaning
# -------------------------------
PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.replace("\n", " ").strip()
    t = t.translate(PUNCT_TABLE)
    t = " ".join(t.split())
    return t

def load_data(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.input_csv)
    rename_map = {}
    if "Description" in df.columns and "text" not in df.columns:
        rename_map["Description"] = "text"
    if "Fraud_Label" in df.columns and "label_type" not in df.columns:
        rename_map["Fraud_Label"] = "label_type"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "text" not in df.columns:
        raise ValueError("Input must contain a 'text' (or 'Description') column.")
    df["text_clean"] = df["text"].apply(basic_clean)
    if "label_type" in df.columns:
        df["label_type"] = df["label_type"].astype(str)
    return df


# -------------------------------
# Encoder using transformers (XLM-R) + mean pooling
# -------------------------------
class RoBertaSentenceEncoder:
    def __init__(self, model_name: str, max_length: int = 256, device: str | None = None):
        from transformers import AutoTokenizer, AutoModel
        import torch
        self.device = device or "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.mdl = AutoModel.from_pretrained(model_name)
        self.mdl.to(self.device)
        self.mdl.eval()
        self.max_length = max_length
        self._torch = torch

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode(self, texts, batch_size: int = 32, normalize: bool = True):
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
        return np.concatenate(out, axis=0)

def load_encoder(model_name: str, max_len: int = 256, device: Optional[str] = None) -> RoBertaSentenceEncoder:
    return RoBertaSentenceEncoder(model_name, max_length=max_len, device=device)

def encode_texts(encoder: RoBertaSentenceEncoder, texts: List[str], normalize: bool = True) -> np.ndarray:
    return encoder.encode(texts, batch_size=32, normalize=normalize)


# -------------------------------
# FAST classifier: LogisticRegression on frozen embeddings
# -------------------------------
def train_fast_linear_clf(df: pd.DataFrame, cfg: Config):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    if "label_type" not in df.columns:
        raise ValueError("Supervised training requires 'label_type' column.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text_clean"], df["label_type"],
        test_size=cfg.test_size, random_state=cfg.random_state, stratify=df["label_type"]
    )

    label2id = {l: i for i, l in enumerate(sorted(df["label_type"].unique()))}
    id2label = {i: l for l, i in label2id.items()}

    device = "cpu"
    enc = load_encoder(cfg.encoder_model, cfg.max_length, device=device)

    train_embs = encode_texts(enc, X_train.tolist(), normalize=True)
    test_embs  = encode_texts(enc, X_test .tolist(), normalize=True)

    clf = LogisticRegression(
        max_iter=500, n_jobs=1, solver="lbfgs", multi_class="auto", verbose=0
    )
    clf.fit(train_embs, [label2id[y] for y in y_train])

    preds = clf.predict(test_embs)
    metrics = {
        "f1_weighted": float(f1_score([label2id[y] for y in y_test], preds, average="weighted"))
    }

    class SimpleEmbedClf:
        def __init__(self, encoder, clf):
            self.model_body = encoder
            self._clf = clf
        def model_head(self, X: np.ndarray) -> np.ndarray:
            if hasattr(self._clf, "predict_proba"):
                return self._clf.predict_proba(X)
            scores = self._clf.decision_function(X)
            exps = np.exp(scores - scores.max(axis=1, keepdims=True))
            return exps / (exps.sum(axis=1, keepdims=True) + 1e-12)

    return SimpleEmbedClf(enc, clf), label2id, id2label, metrics


# -------------------------------
# Topic modeling (BERTopic + fallback + optional override)
# -------------------------------
def topic_model(texts: List[str], embeddings: np.ndarray, cfg: Config):
    from bertopic import BERTopic
    from umap import UMAP
    import hdbscan
    from sklearn.feature_extraction.text import CountVectorizer

    umap_model = UMAP(
        n_neighbors=cfg.umap_n_neighbors,
        n_components=5,
        min_dist=cfg.umap_min_dist,
        random_state=cfg.random_state
    )
    hdb_model = hdbscan.HDBSCAN(
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    vectorizer_model = CountVectorizer(ngram_range=(1, 3), min_df=5)

    tm = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdb_model,
        vectorizer_model=vectorizer_model,
        language="multilingual",
        calculate_probabilities=True,
        min_topic_size=cfg.hdbscan_min_cluster_size,
        verbose=True,
    )

    topics, probs = tm.fit_transform(texts, embeddings)

    # Optional: force KMeans topics (stable number / IDs)
    if cfg.kmeans_override_k and cfg.kmeans_override_k > 1:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=int(cfg.kmeans_override_k), random_state=cfg.random_state)
        hard = kmeans.fit_predict(embeddings)
        tm.set_topics(hard)
        topics = hard
        probs = None

    # Fallback if too many outliers
    outlier_rate = (np.array(topics) == -1).mean()
    if outlier_rate > 0.4 and not cfg.kmeans_override_k:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=cfg.kmeans_fallback_k, random_state=cfg.random_state)
        hard = kmeans.fit_predict(embeddings)
        tm.set_topics(hard)
        topics = hard
        probs = None

    return tm, topics, probs


# -------------------------------
# Sentiment
# -------------------------------
def _heuristic_sentiment(texts: List[str]) -> np.ndarray:
    pos = {
        "good","great","excel","excellent","amazing","happy","satisfied","recommend","love","quick",
        "rápido","bueno","genial","excelente","feliz","satisfecho","recomiendo","me encanta","ágil","rápida"
    }
    neg = {
        "bad","terrible","awful","angry","sad","poor","slow","issue","problem","fraud","scam","claim denied",
        "malo","terrible","horrible","enojado","triste","lento","problema","fraude","estafa","rechazado","denegado"
    }
    scores = []
    for t in texts:
        t_low = (t or "").lower()
        p = sum(w in t_low for w in pos)
        n = sum(w in t_low for w in neg)
        if p == 0 and n == 0:
            scores.append(0.0)
        else:
            s = (p - n) / max(1, (p + n))
            scores.append(float(s))
    return np.array(scores, dtype=float)

def predict_sentiment(texts: List[str], cfg: Config) -> np.ndarray:
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        tok = AutoTokenizer.from_pretrained(cfg.sentiment_model, use_fast=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(
            cfg.sentiment_model, use_safetensors=True
        )
        mdl.eval()
        scores = []
        with torch.no_grad():
            for t in texts:
                inputs = tok(t, truncation=True, padding=True, return_tensors="pt")
                logits = mdl(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                neg, neu, pos = probs.tolist()
                scores.append(pos - neg)  # [-1, 1]
        return np.array(scores, dtype=float)
    except Exception as e:
        warnings.warn(
            "Sentiment model not available with safetensors or network error; using heuristic. "
            f"Details: {e}"
        )
        return _heuristic_sentiment(texts)


# -------------------------------
# Improved blacklist miner
# -------------------------------
def extract_discriminative_phrases(
        df: pd.DataFrame,
        cluster_col: str,
        text_col: str,
        target_clusters: List[int],
        *,
        spacy_model: str = "xx_sent_ud_sm",
        max_phrases_per_cluster: int = 200,
        min_len_chars: int = 4,
        min_len_tokens: int = 2,
        boilerplate_phrases: Tuple[str, ...] = (),
        ban_location_entities: bool = True,
        min_cluster_freq: int = 3,
        max_other_frac: float = 0.05,
        dedupe_similarity: float = 0.88,
        regex_drop: Tuple[str, ...] = (),
) -> pd.DataFrame:
    """
    Entity/NP-aware mining with:
      • NER & NP spans (fallback to 2–3grams)
      • Contrastive log-odds * log(1+freq)
      • Coverage filters (cluster/other)
      • Regex & day/edge-stopword cleanup
      • Fuzzy dedupe of near-duplicates
    Returns: label, term, score, freq_cluster, freq_others, cov_cluster, cov_others, kind
    """
    from collections import Counter, defaultdict
    from difflib import SequenceMatcher

    # Ensure we always return expected columns even if empty
    EMPTY_SCHEMA = pd.DataFrame(
        columns=["label","term","score","freq_cluster","freq_others","cov_cluster","cov_others","kind"]
    )

    if not target_clusters:
        return EMPTY_SCHEMA.copy()

    stop_edge = {"and","or","of","to","de","la","el","the","a","an","y","o","del","al"}
    day_words = {"monday","tuesday","wednesday","thursday","friday","saturday","sunday"}
    drop_ent_types = {"GPE","LOC","FAC"} if ban_location_entities else set()
    rx_drop = [re.compile(pat, flags=re.I) for pat in regex_drop]

    # try spaCy (optional)
    nlp = None
    try:
        import spacy
        nlp = spacy.load(spacy_model, disable=["lemmatizer","parser","textcat"])
    except Exception:
        warnings.warn("spaCy model not available; using simple n-gram fallback.")

    df = df.copy()
    df["_doc_id"] = np.arange(len(df))

    def clean_boilerplate(s: str) -> str:
        x = (s or "").lower()
        for p in boilerplate_phrases:
            x = x.replace(p, " ")
        for rgx in rx_drop:
            if rgx.search(x):
                return ""
        return re.sub(r"\s+", " ", x).strip()

    def good_span(txt: str) -> bool:
        if len(txt) < min_len_chars:
            return False
        toks = txt.split()
        if len(toks) < min_len_tokens:
            return False
        if toks[0] in stop_edge or toks[-1] in stop_edge:
            return False
        if any(t in day_words for t in toks):
            return False
        return True

    def mine_doc(text: str):
        text = clean_boilerplate(text)
        if not text:
            return []
        out = []
        if nlp is None:
            toks = re.findall(r"[A-Za-zÀ-ÿ0-9]+", text.lower())
            toks = [t for t in toks if len(t) >= 3]
            for n in (2, 3):
                for i in range(len(toks) - n + 1):
                    span = " ".join(toks[i:i+n])
                    if good_span(span):
                        out.append(("NG"+str(n), span))
            return out

        doc = nlp(text)

        for ent in doc.ents:
            if ent.label_ in drop_ent_types:
                continue
            term = ent.text.strip().lower()
            if good_span(term):
                out.append(("ENT:"+ent.label_, term))

        i = 0
        while i < len(doc):
            j = i
            bag = []
            while j < len(doc) and doc[j].pos_ in {"ADJ","NOUN","PROPN"}:
                bag.append(doc[j].text)
                j += 1
            if bag and doc[j-1].pos_ in {"NOUN","PROPN"}:
                term = " ".join(bag).strip().lower()
                if good_span(term):
                    out.append(("NP", term))
                i = j
            else:
                i += 1
        return out

    clusters = list(target_clusters)
    cluster_term: Dict[Tuple[int,str], Counter] = defaultdict(Counter)
    other_term: Dict[Tuple[int,str], Counter]   = defaultdict(Counter)
    cluster_doc: Dict[Tuple[int,str], Counter]  = defaultdict(Counter)
    other_doc: Dict[Tuple[int,str], Counter]    = defaultdict(Counter)

    for k in clusters:
        sub = df.loc[df[cluster_col] == k, [text_col, "_doc_id"]]
        for txt, did in sub.itertuples(index=False):
            seen = set()
            for kind, term in mine_doc(str(txt)):
                cluster_term[(k, kind)][term] += 1
                seen.add(term)
            for t in seen:
                cluster_doc[(k, "ALL")][t] += 1

        sub_o = df.loc[df[cluster_col] != k, [text_col, "_doc_id"]]
        for txt, did in sub_o.itertuples(index=False):
            seen = set()
            for kind, term in mine_doc(str(txt)):
                other_term[(k, kind)][term] += 1
                seen.add(term)
            for t in seen:
                other_doc[(k, "ALL")][t] += 1

    n_docs_c = {k: int((df[cluster_col] == k).sum()) for k in clusters}
    n_docs_o = {k: int((df[cluster_col] != k).sum()) for k in clusters}

    rows = []
    for k in clusters:
        l1 = [t for (_k, t) in cluster_term if _k == k]
        l2 = [t for (_k, t) in other_term   if _k == k]
        kinds = set(l1) | set(l2)  # FIXED: union on sets (not lists)

        for kind in kinds:
            cc = cluster_term.get((k,kind), Counter())
            oc = other_term.get((k,kind), Counter())
            V = len(set(cc) | set(oc)) or 1
            sum_c = sum(cc.values()) + V
            sum_o = sum(oc.values()) + V
            for term in set(cc) | set(oc):
                fc = cc.get(term, 0) + 1
                fo = oc.get(term, 0) + 1
                cov_c = cluster_doc[(k,"ALL")].get(term, 0) / max(1, n_docs_c[k])
                cov_o = other_doc[(k,"ALL")].get(term, 0) / max(1, n_docs_o[k])

                if (fc-1) < min_cluster_freq:
                    continue
                if cov_o > max_other_frac:
                    continue

                score = np.log(fc / sum_c) - np.log(fo / sum_o)
                score *= np.log1p(fc)

                tnorm = term.strip(",.;: ").lower()
                if len(tnorm) < 4:
                    continue

                rows.append({
                    "label": f"cluster_{k}",
                    "term": tnorm,
                    "score": float(score),
                    "freq_cluster": int(fc-1),
                    "freq_others": int(fo-1),
                    "cov_cluster": float(cov_c),
                    "cov_others": float(cov_o),
                    "kind": kind
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return EMPTY_SCHEMA.copy()

    from difflib import SequenceMatcher
    def dedupe_block(df_block: pd.DataFrame) -> pd.DataFrame:
        kept: List[Dict[str, Any]] = []
        for _, row in df_block.sort_values("score", ascending=False).iterrows():
            t = row["term"]
            if any(SequenceMatcher(None, t, r["term"]).ratio() >= dedupe_similarity for r in kept):
                continue
            kept.append(row.to_dict())
        return pd.DataFrame(kept)

    out = out.groupby("label", group_keys=False).apply(dedupe_block)
    out = out.groupby("label", as_index=False).head(max_phrases_per_cluster)
    return out.sort_values(["label","score"], ascending=[True, False])


# -------------------------------
# Risk & recommendations
# -------------------------------
def risk_score_row(clf_probs: Dict[str, float], sentiment_score: float, topic_risk: float, cfg: Config) -> float:
    p_max = max(clf_probs.values()) if clf_probs else 0.0
    s = (sentiment_score + 1) / 2
    return float(cfg.w_clf * p_max + cfg.w_sent * s + cfg.w_topic * topic_risk)

def generate_recommendation(clf_label: str, topic_label: str, score: float) -> str:
    pri = "High" if score > 0.75 else "Medium" if score > 0.5 else "Low"
    return f"Priority {pri}. Route to analyst specializing in '{clf_label}'. Check patterns related to topic '{topic_label}'."


# -------------------------------
# Pipeline
# -------------------------------
def run_pipeline(cfg: Optional[Config] = None) -> Dict[str, Any]:
    cfg = cfg or Config()
    configure_runtime(cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)

    df = load_data(cfg)

    # 1) Train classifier (FAST: LogReg on frozen XLM-R embeddings)
    model, label2id, id2label, clf_metrics = train_fast_linear_clf(df, cfg)

    # 2) Predict class probabilities for all rows
    def predict_probs(texts: List[str]) -> List[Dict[str, float]]:
        embs = encode_texts(model.model_body, texts, normalize=True)
        probs = model.model_head(embs)
        out = []
        for row in probs:
            out.append({id2label[i]: float(row[i]) for i in range(len(id2label))})
        return out

    df["clf_probs"] = predict_probs(df["text_clean"].tolist())
    df["pred_label"] = df["clf_probs"].apply(lambda d: max(d, key=d.get) if d else None)
    df["pred_conf"] = df["clf_probs"].apply(lambda d: max(d.values()) if d else 0.0)

    # 3) Embeddings for topic modeling (same encoder)
    embs = encode_texts(model.model_body, df["text_clean"].tolist(), normalize=True)

    # 4) Topics
    tm, topics, probs = topic_model(df["text_clean"].tolist(), embs, cfg)
    df["topic_id"] = topics
    df["topic_prob"] = probs.max(axis=1) if isinstance(probs, np.ndarray) else 1.0

    # 5) Sentiment
    df["sentiment"] = predict_sentiment(df["text_clean"].tolist(), cfg)

    # 6) Topic prior
    topic_prior = df.groupby("topic_id")["pred_conf"].mean()
    topic_prior = (topic_prior - topic_prior.min()) / (topic_prior.max() - topic_prior.min() + 1e-6)
    df["topic_risk"] = df["topic_id"].map(topic_prior.to_dict()).fillna(0.0)

    # 7) Risk score
    df["risk_score"] = [
        risk_score_row(cp, s, tr, cfg)
        for cp, s, tr in zip(df["clf_probs"], df["sentiment"], df["topic_risk"])
    ]

    # 8) Recommendation
    topic_info = tm.get_topic_info()
    topic_words = {row.Topic: row.Name for _, row in topic_info.iterrows()}
    df["recommendation"] = [
        generate_recommendation(lbl, topic_words.get(tid, str(tid)), rs)
        for lbl, tid, rs in zip(df["pred_label"], df["topic_id"], df["risk_score"])
    ]

    # 9) Blacklist candidates (new miner)
    high_clusters = df.loc[df["risk_score"] > df["risk_score"].quantile(0.75), "topic_id"].unique().tolist()
    if len(high_clusters) == 0:
        warnings.warn("No high-risk clusters found; lowering threshold to median.")
        high_clusters = df.loc[df["risk_score"] > df["risk_score"].median(), "topic_id"].unique().tolist()

    bl_df = extract_discriminative_phrases(
        df,
        cluster_col="topic_id",
        text_col="text_clean",
        target_clusters=high_clusters,
        spacy_model=cfg.spacy_model,
        boilerplate_phrases=cfg.generic_phrases,
        max_phrases_per_cluster=200,
        ban_location_entities=cfg.ban_location_entities,
        min_cluster_freq=cfg.min_cluster_freq,
        max_other_frac=cfg.max_other_frac,
        dedupe_similarity=cfg.dedupe_similarity,
        regex_drop=cfg.regex_drop,
    )

    # 10) Persist (always CSV; Parquet if available)
    def _safe_write_predictions(df_out: pd.DataFrame, outdir: str):
        os.makedirs(outdir, exist_ok=True)
        csv_path = os.path.join(outdir, "predictions.csv")
        df_out.to_csv(csv_path, index=False)

        parquet_path = os.path.join(outdir, "predictions.parquet")
        wrote_parquet = False
        try:
            df_out.to_parquet(parquet_path, index=False)
            wrote_parquet = True
        except Exception as e:
            warnings.warn(f"Could not write Parquet ({parquet_path}): {e}. CSV is still saved.")
        return csv_path, parquet_path if wrote_parquet else None

    def _ensure_columns(df_in: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Ensure df has all columns; add empty ones if missing."""
        for c in cols:
            if c not in df_in.columns:
                df_in[c] = []  # empty column
        return df_in[cols]

    def _safe_write_blacklist(blacklist_df: pd.DataFrame, outdir: str):
        os.makedirs(outdir, exist_ok=True)
        expected_cols = ["label","term","score","freq_cluster","freq_others","cov_cluster","cov_others","kind"]
        if blacklist_df is None or blacklist_df.empty:
            blacklist_df = pd.DataFrame(columns=expected_cols)
        else:
            blacklist_df = _ensure_columns(blacklist_df, expected_cols)

        # write full list
        p_all = os.path.join(outdir, "blacklist_candidates.csv")
        blacklist_df.to_csv(p_all, index=False)

        # write top-50 per label safely
        if not blacklist_df.empty and "label" in blacklist_df.columns:
            top50 = (blacklist_df.groupby("label", as_index=False)
                     .apply(lambda g: g.sort_values("score", ascending=False).head(50))
                     .reset_index(drop=True))
        else:
            top50 = pd.DataFrame(columns=expected_cols)
        p_top = os.path.join(outdir, "blacklist_top50.csv")
        top50.to_csv(p_top, index=False)
        return p_all, p_top

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
    csv_path, parquet_path = _safe_write_predictions(df_out, cfg.output_dir)
    bl_all_path, bl_top_path = _safe_write_blacklist(bl_df, cfg.output_dir)

    topics_dir = os.path.join(cfg.output_dir, "bertopic_model")
    tm.save(topics_dir)

    # Console summary
    print("\n=== Artifacts written ===")
    print("Artifacts dir:       ", os.path.abspath(cfg.output_dir))
    print("Predictions CSV:     ", os.path.abspath(csv_path))
    if parquet_path:
        print("Predictions Parquet: ", os.path.abspath(parquet_path))
    else:
        print("Predictions Parquet: (skipped; pyarrow/fastparquet not found)")
    print("Blacklist (all):     ", os.path.abspath(bl_all_path))
    print("Blacklist (top50):   ", os.path.abspath(bl_top_path))
    print("BERTopic model dir:  ", os.path.abspath(topics_dir))

    return {
        "metrics": clf_metrics,
        "artifacts": {
            "predictions_csv": os.path.join(cfg.output_dir, "predictions.csv"),
            "predictions_parquet": os.path.join(cfg.output_dir, "predictions.parquet"),
            "blacklist_all": os.path.join(cfg.output_dir, "blacklist_candidates.csv"),
            "blacklist_top50": os.path.join(cfg.output_dir, "blacklist_top50.csv"),
            "encoder_model": cfg.encoder_model,
            "topics_model": os.path.join(cfg.output_dir, "bertopic_model"),
        },
        "mode": "fast_linear",
    }


if __name__ == "__main__":
    cfg = Config()
    print(json.dumps(run_pipeline(cfg), indent=2))
