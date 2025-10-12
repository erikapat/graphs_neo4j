from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class Config:
    # Data
    input_csv: str = "data/synth/claims_llm.csv"
    output_dir: str = "./artifacts"

    # Encoder backbone
    encoder_model: str = "xlm-roberta-base"
    max_length: int = 256

    # SetFit trainer
    setfit_num_epochs: int = 2
    setfit_batch_size: int = 16
    setfit_learning_rate: float = 2e-5
    setfit_num_iterations: int = 1
    setfit_max_length: int = 256
    unfreeze_last_n: int = 0

    # Fast-linear toggle (False = SetFit)
    fast_linear: bool = False

    # Data split
    test_size: float = 0.2
    random_state: int = 42

    # Runtime
    force_cpu: bool = True
    torch_num_threads: int = 1

    # Sentiment
    sentiment_model: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    # Topic modeling
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.0
    hdbscan_min_cluster_size: int = 20
    hdbscan_min_samples: int = 5
    kmeans_fallback_k: int = 12
    kmeans_override_k: Optional[int] = None  # None = let BERTopic/HDBSCAN decide

    # Risk weights
    w_clf: float = 0.6
    w_sent: float = 0.1
    w_topic: float = 0.3

    # Blacklist miner
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
        "please let me know","as usual","work as usual","claim number",
        "policy number","thank you","best regards","let me know",
    )
