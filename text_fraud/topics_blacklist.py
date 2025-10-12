from __future__ import annotations
import re, warnings
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from config import Config

def topic_model(
        texts: List[str],
        embeddings: np.ndarray,
        cfg,
        labels: Optional[List[str]] = None,   # <-- NEW
):
    """
    Build BERTopic with UMAP/HDBSCAN.
    If cfg.label_aware_topics is True *and* labels are provided, we pass them to UMAP/BERTopic
    so that the reduction is 'semi-supervised' (points with the same label are nudged together).
    """
    from bertopic import BERTopic
    from umap import UMAP
    import hdbscan
    from sklearn.feature_extraction.text import CountVectorizer

    # UMAP: set up for label-aware if requested
    if cfg.label_aware_topics and labels is not None:
        umap_model = UMAP(
            n_neighbors=cfg.umap_n_neighbors,
            n_components=5,
            min_dist=cfg.umap_min_dist,
            random_state=cfg.random_state,
            target_metric=getattr(cfg, "umap_target_metric", "categorical"),
            target_weight=getattr(cfg, "umap_target_weight", 0.4),
        )
        _y = list(labels)  # ensure list
    else:
        umap_model = UMAP(
            n_neighbors=cfg.umap_n_neighbors,
            n_components=5,
            min_dist=cfg.umap_min_dist,
            random_state=cfg.random_state,
        )
        _y = None

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

    # IMPORTANT: pass y to fit_transform when available.
    # Newer BERTopic versions accept y and will forward it to UMAP.
    try:
        topics, probs = tm.fit_transform(texts, embeddings, y=_y)
    except TypeError:
        # Older BERTopic: no y support -> fallback to unsupervised
        topics, probs = tm.fit_transform(texts, embeddings)

    # Fallback if many outliers and no explicit K override
    outlier_rate = (np.array(topics) == -1).mean()
    if outlier_rate > 0.4 and not getattr(cfg, "kmeans_override_k", None):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=cfg.kmeans_fallback_k, random_state=cfg.random_state)
        hard = kmeans.fit_predict(embeddings)
        tm.set_topics(hard)
        topics = hard
        probs = None

    # Optional override to a fixed-K solution for stable IDs
    if getattr(cfg, "kmeans_override_k", None):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=int(cfg.kmeans_override_k), random_state=cfg.random_state)
        hard = kmeans.fit_predict(embeddings)
        tm.set_topics(hard)
        topics = hard
        probs = None

    return tm, topics, probs

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
        boilerplate_phrases: Tuple[str,...] = (),
        ban_location_entities: bool = True,
        min_cluster_freq: int = 3,
        max_other_frac: float = 0.05,
        dedupe_similarity: float = 0.88,
        regex_drop: Tuple[str,...] = (),
) -> pd.DataFrame:
    """Entity/NP-aware miner. Returns label, term, score, freq_cluster, freq_others, cov_cluster, cov_others, kind"""
    from collections import Counter, defaultdict
    from difflib import SequenceMatcher

    EMPTY = pd.DataFrame(columns=["label","term","score","freq_cluster","freq_others","cov_cluster","cov_others","kind"])
    if not target_clusters:
        return EMPTY.copy()

    stop_edge = {"and","or","of","to","de","la","el","the","a","an","y","o","del","al"}
    day_words = {"monday","tuesday","wednesday","thursday","friday","saturday","sunday"}
    drop_ent_types = {"GPE","LOC","FAC"} if ban_location_entities else set()
    rx_drop = [re.compile(p, flags=re.I) for p in regex_drop]

    # spaCy optional
    nlp = None
    try:
        import spacy
        nlp = spacy.load(spacy_model, disable=["lemmatizer","parser","textcat"])
    except Exception:
        warnings.warn("spaCy model not available; using n-gram fallback.")

    def clean_boilerplate(s: str) -> str:
        x = (s or "").lower()
        for p in boilerplate_phrases: x = x.replace(p, " ")
        for rgx in rx_drop:
            if rgx.search(x): return ""
        return re.sub(r"\s+"," ",x).strip()

    def good_span(txt: str) -> bool:
        if len(txt) < min_len_chars: return False
        toks = txt.split()
        if len(toks) < min_len_tokens: return False
        if toks[0] in stop_edge or toks[-1] in stop_edge: return False
        if any(t in day_words for t in toks): return False
        return True

    def mine_doc(text: str):
        text = clean_boilerplate(text)
        if not text: return []
        out = []
        if nlp is None:
            toks = re.findall(r"[A-Za-zÀ-ÿ0-9]+", text.lower())
            toks = [t for t in toks if len(t) >= 3]
            for n in (2,3):
                for i in range(len(toks)-n+1):
                    span = " ".join(toks[i:i+n])
                    if good_span(span): out.append(("NG"+str(n), span))
            return out
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in drop_ent_types: continue
            term = ent.text.strip().lower()
            if good_span(term): out.append(("ENT:"+ent.label_, term))
        i = 0
        while i < len(doc):
            j = i; bag = []
            while j < len(doc) and doc[j].pos_ in {"ADJ","NOUN","PROPN"}:
                bag.append(doc[j].text); j += 1
            if bag and doc[j-1].pos_ in {"NOUN","PROPN"}:
                term = " ".join(bag).strip().lower()
                if good_span(term): out.append(("NP", term))
                i = j
            else:
                i += 1
        return out

    df = df.copy()
    df["_doc_id"] = np.arange(len(df))
    from collections import defaultdict, Counter
    cluster_term, other_term = defaultdict(Counter), defaultdict(Counter)
    cluster_doc,  other_doc  = defaultdict(Counter), defaultdict(Counter)

    clusters = list(target_clusters)
    for k in clusters:
        sub = df.loc[df[cluster_col]==k, [text_col,"_doc_id"]]
        for txt, did in sub.itertuples(index=False):
            seen = set()
            for kind, term in mine_doc(str(txt)):
                cluster_term[(k,kind)][term] += 1; seen.add(term)
            for t in seen: cluster_doc[(k,"ALL")][t] += 1

        sub_o = df.loc[df[cluster_col]!=k, [text_col,"_doc_id"]]
        for txt, did in sub_o.itertuples(index=False):
            seen = set()
            for kind, term in mine_doc(str(txt)):
                other_term[(k,kind)][term] += 1; seen.add(term)
            for t in seen: other_doc[(k,"ALL")][t] += 1

    n_docs_c = {k: int((df[cluster_col]==k).sum()) for k in clusters}
    n_docs_o = {k: int((df[cluster_col]!=k).sum()) for k in clusters}

    rows = []
    for k in clusters:
        kinds = set([t for (_k,t) in cluster_term if _k==k]) | set([t for (_k,t) in other_term if _k==k])
        for kind in kinds:
            cc = cluster_term.get((k,kind), Counter())
            oc = other_term.get((k,kind), Counter())
            V = len(set(cc)|set(oc)) or 1
            sum_c = sum(cc.values()) + V
            sum_o = sum(oc.values()) + V
            for term in set(cc)|set(oc):
                fc = cc.get(term,0) + 1
                fo = oc.get(term,0) + 1
                cov_c = cluster_doc[(k,"ALL")].get(term,0) / max(1, n_docs_c[k])
                cov_o = other_doc[(k,"ALL")].get(term,0) / max(1, n_docs_o[k])
                if (fc-1) < min_cluster_freq: continue
                if cov_o > max_other_frac:   continue
                score = np.log(fc/sum_c) - np.log(fo/sum_o)
                score *= np.log1p(fc)
                tnorm = term.strip(",.;: ").lower()
                if len(tnorm) < 4: continue
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
    if out.empty: return EMPTY.copy()

    from difflib import SequenceMatcher
    def dedupe_block(df_block: pd.DataFrame) -> pd.DataFrame:
        kept = []
        for _, row in df_block.sort_values("score", ascending=False).iterrows():
            t = row["term"]
            if any(SequenceMatcher(None, t, r["term"]).ratio() >= dedupe_similarity for r in kept):
                continue
            kept.append(row.to_dict())
        return pd.DataFrame(kept)

    out = out.groupby("label", group_keys=False).apply(dedupe_block)
    out = out.groupby("label", as_index=False).head(max_phrases_per_cluster)
    return out.sort_values(["label","score"], ascending=[True, False])
