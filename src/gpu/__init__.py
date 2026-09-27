"""GPU jobs. `reranker_module` picks the band-reranker backend for a trained model dir."""
import importlib
import json
from pathlib import Path

# meta.json "kind" -> module. Dirs written before `kind` existed are e5 rerankers.
BACKENDS = {"e5": "gpu.reranker", "laya": "gpu.laya_rr"}


def reranker_module(model_dir):
    """The module (serialize, score) that reads `model_dir`, from its meta.json `kind`."""
    meta = Path(model_dir) / "meta.json"
    kind = json.loads(meta.read_text()).get("kind", "e5") if meta.exists() else "e5"
    if kind not in BACKENDS:
        raise ValueError(f"{model_dir}: unknown reranker kind {kind!r}; expected one of {sorted(BACKENDS)}")
    return importlib.import_module(BACKENDS[kind])
