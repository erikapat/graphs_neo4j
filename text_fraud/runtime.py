from __future__ import annotations
import os, warnings
from config import Config

def configure_runtime(cfg: Config):
    if cfg.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_READ_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ENABLE_TELEMETRY", "0")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        import torch
        torch.set_num_threads(max(1, int(cfg.torch_num_threads)))
    except Exception:
        pass
    try:
        import numpy as _np
        if int(_np.version.version.split(".")[0]) >= 2:
            warnings.warn("NumPy >=2.x detected; if torch wheels complain, try: pip install 'numpy<2'")
    except Exception:
        pass
