# blacklist_miner.py
from __future__ import annotations
import re, numpy as np, pandas as pd
from collections import Counter
from difflib import SequenceMatcher
from typing import List

def extract_blacklist(
        df: pd.DataFrame,
        target_clusters: List[int],
        min_cluster_freq: int = 3,
        max_other_frac: float = 0.05,
        dedupe_similarity: float = 0.88
) -> pd.DataFrame:
    if not target_clusters:
        return pd.DataFrame(columns=["label","term","score","freq_cluster","freq_others","cov_cluster","cov_others"])

    def clean(s: str) -> str:
        x = (s or "").lower()
        x = re.sub(r"\s+", " ", x).strip()
        return x

    def ngrams(txt: str):
        toks = re.findall(r"[A-Za-zÀ-ÿ0-9]+", txt.lower())
        toks = [t for t in toks if len(t) >= 3]
        out = []
        for n in (2,3):
            for i in range(len(toks)-n+1):
                out.append(" ".join(toks[i:i+n]))
        return out

    rows = []
    for k in target_clusters:
        sub = df[df["topic_id"]==k]["text_clean"].map(clean).tolist()
        oth = df[df["topic_id"]!=k]["text_clean"].map(clean).tolist()
        sub = [s for s in sub if s]; oth = [s for s in oth if s]
        if not sub: continue

        c_sub, c_oth = Counter(), Counter()
        d_sub, d_oth = Counter(), Counter()

        for s in sub:
            seen=set()
            for g in ngrams(s): c_sub[g]+=1; seen.add(g)
            for g in seen: d_sub[g]+=1
        for s in oth:
            seen=set()
            for g in ngrams(s): c_oth[g]+=1; seen.add(g)
            for g in seen: d_oth[g]+=1

        Ns = max(1,len(sub)); No = max(1,len(oth))
        V = len(set(c_sub)|set(c_oth)) or 1
        sum_s = sum(c_sub.values())+V
        sum_o = sum(c_oth.values())+V

        for term in set(c_sub)|set(c_oth):
            fc = c_sub.get(term,0); fo = c_oth.get(term,0)
            cov_s = d_sub.get(term,0)/Ns
            cov_o = d_oth.get(term,0)/No
            if fc < min_cluster_freq: continue
            if cov_o > max_other_frac: continue
            score = (np.log((fc+1)/sum_s) - np.log((fo+1)/sum_o)) * np.log1p(fc)
            rows.append({
                "label": f"cluster_{k}",
                "term": term,
                "score": float(score),
                "freq_cluster": int(fc),
                "freq_others": int(fo),
                "cov_cluster": float(cov_s),
                "cov_others": float(cov_o),
            })

    out = pd.DataFrame(rows)
    if out.empty: return out

    def dedupe(g):
        kept=[]
        for _, r in g.sort_values("score", ascending=False).iterrows():
            t = r["term"]
            if any(SequenceMatcher(None, t, x["term"]).ratio() >= dedupe_similarity for x in kept):
                continue
            kept.append(r.to_dict())
        return pd.DataFrame(kept)

    out = out.groupby("label", group_keys=False).apply(dedupe)
    return out.sort_values(["label","score"], ascending=[True, False])
