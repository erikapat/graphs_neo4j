# train_setfit.py
from __future__ import annotations
import warnings
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple

def train_setfit_classifier(df: pd.DataFrame, cfg) -> Tuple[object, Dict[str,int], Dict[int,str], Dict[str,float]]:
    """
    Trains SetFit with a SentenceTransformer (XLM-R) body.
    Avoids no-grad error by unfreezing last N transformer layers.
    """
    if "label_type" not in df.columns:
        raise ValueError("Supervised training requires 'label_type' column.")

    from sklearn.model_selection import train_test_split
    from datasets import Dataset
    try:
        from setfit.losses import CosineSimilarityLoss
    except Exception:
        from sentence_transformers.losses import CosineSimilarityLoss
    from setfit import SetFitModel, SetFitTrainer

    X_tr, X_te, y_tr, y_te = train_test_split(
        df["text_clean"], df["label_type"],
        test_size=cfg.test_size, random_state=cfg.random_state, stratify=df["label_type"]
    )

    label2id = {l: i for i, l in enumerate(sorted(df["label_type"].unique()))}
    id2label = {i: l for l, i in label2id.items()}

    ds_tr = Dataset.from_pandas(pd.DataFrame({"text": X_tr, "label": [label2id[y] for y in y_tr]}))
    ds_te = Dataset.from_pandas(pd.DataFrame({"text": X_te, "label": [label2id[y] for y in y_te]}))

    model = SetFitModel.from_pretrained(cfg.st_body, labels=list(range(len(label2id))))

    # Unfreeze last N XLM-R layers (when accessible); else enable grads for body
    try:
        trf = model.model_body[0]  # Transformer module inside SentenceTransformer
        hf = getattr(trf, "auto_model", None)
        if hf is not None and cfg.unfreeze_last_n > 0:
            for p in hf.parameters():
                p.requires_grad = False
            enc = getattr(hf, "encoder", None)
            if enc and hasattr(enc, "layer"):
                for layer in enc.layer[-cfg.unfreeze_last_n:]:
                    for p in layer.parameters():
                        p.requires_grad = True
        else:
            for p in model.model_body.parameters():
                p.requires_grad = True
    except Exception as e:
        warnings.warn(f"Selective unfreeze failed, enabling grads for body: {e}")
        for p in model.model_body.parameters():
            p.requires_grad = True

    trainer = SetFitTrainer(
        model=model,
        train_dataset=ds_tr,
        eval_dataset=ds_te,
        loss_class=CosineSimilarityLoss,
        num_iterations=cfg.setfit_num_iterations,
        num_epochs=cfg.setfit_num_epochs,
        batch_size=cfg.setfit_batch_size,
        learning_rate=cfg.setfit_learning_rate,
        column_mapping={"text": "text", "label": "label"},
    )
    trainer.train()
    metrics = trainer.evaluate()
    metrics = {f"setfit_{k}": float(v) for k, v in metrics.items()}
    return model, label2id, id2label, metrics

