# topics_label_aware.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional

def build_topics_label_aware(
        texts: List[str],
        embeddings: np.ndarray,
        labels: List[str],
        cfg
) -> Tuple[object, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Label-aware UMAP (y=labels) -> HDBSCAN; BERTopic over reduced space.
    KMeans fallback if too many outliers.
    """
    from umap import UMAP
    import hdbscan
    from sklearn.feature_extraction.text import CountVectorizer
    from bertopic import BERTopic
    from sklearn.cluster import KMeans

    y = np.array(labels)
    reducer = UMAP(
        n_neighbors=cfg.umap_n_neighbors,
        n_components=5,
        min_dist=cfg.umap_min_dist,
        random_state=cfg.random_state,
    )
    reduced = reducer.fit_transform(embeddings, y=y)

    hdb_model = hdbscan.HDBSCAN(
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )
    vectorizer_model = CountVectorizer(ngram_range=(1, 3), min_df=5)

    tm = BERTopic(
        umap_model=None,  # we pass pre-reduced features
        hdbscan_model=hdb_model,
        vectorizer_model=vectorizer_model,
        language="multilingual",
        calculate_probabilities=True,
        min_topic_size=cfg.hdbscan_min_cluster_size,
        verbose=False,
    )
    topics, probs = tm.fit_transform(texts, reduced)

    outlier_rate = (np.array(topics) == -1).mean()
    if outlier_rate > 0.4:
        kmeans = KMeans(n_clusters=cfg.kmeans_fallback_k, random_state=cfg.random_state)
        hard = kmeans.fit_predict(reduced)
        tm.set_topics(hard)
        topics = hard
        probs = None

    return tm, topics, probs, reduced
