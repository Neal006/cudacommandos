"""Plain run log, written next to every training run.

    runs/<run_id>/metrics.jsonl  one JSON object per eval tick (train/valid loss per fold)
    runs/<run_id>/summary.json   end-of-run numbers (OOF score, per-fold scores, decision)
    runs/<run_id>/folds.tsv      s1_id<TAB>fold
"""
import json
import math
import numbers
from pathlib import Path


def _finite(o):
    """NaN/inf -> None, so summary.json stays strict JSON."""
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, numbers.Real) and not math.isfinite(o):   # also numpy float32/64
        return None
    return o


class RunLog:
    def __init__(self, run_dir=None):
        self.dir = Path(run_dir or "runs/adhoc")
        self.dir.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.dir / "metrics.jsonl", "a", encoding="utf-8")

    def tick(self, **kw):
        """Append one metrics line; flushed so `tail -f` sees it immediately."""
        self._fh.write(json.dumps(kw) + "\n")
        self._fh.flush()

    def lgb_callback(self, fold, every=10):
        """LightGBM callback: logs train/valid loss every `every` iterations.

        Requires valid_sets=[train_ds, valid_ds], valid_names=["train", "valid"].
        """
        def _cb(env):
            if env.iteration % every:
                return
            vals = {name: v for name, _metric, v, _hib in env.evaluation_result_list}
            self.tick(fold=fold, iter=env.iteration,
                      train_loss=vals.get("train"), valid_loss=vals.get("valid"))
        _cb.order = 30
        return _cb

    def write_folds(self, s1_ids, folds):
        with open(self.dir / "folds.tsv", "w", encoding="utf-8", newline="\n") as fh:
            fh.write("s1_id\tfold\n")
            fh.writelines(f"{i}\t{f}\n" for i, f in zip(s1_ids, folds))

    def write_summary(self, **summary):
        with open(self.dir / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(_finite(summary), fh, indent=2, default=float, allow_nan=False)

    def end(self):
        self.tick(event="end")
        self._fh.close()
