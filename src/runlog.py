"""Run log consumed by tools/mlguard (the Rust trust checker).

    runs/<run_id>/metrics.jsonl  one JSON object per eval tick   -> `mlguard watch` (live, in background)
    runs/<run_id>/summary.json   end-of-run numbers              -> `mlguard run` (CI + after training)
    runs/<run_id>/folds.tsv      s1_id<TAB>fold                  -> `mlguard split`
    runs/<run_id>/MLGUARD_STOP   written BY mlguard on violation -> training stops at next eval

Schema of summary.json: docs/master-plan/MLGUARD.md.
"""
import json
import os
from pathlib import Path


class RunLog:
    def __init__(self, run_dir=None):
        self.dir = Path(run_dir or os.environ.get("MLGUARD_RUN_DIR", "runs/adhoc"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.dir / "metrics.jsonl", "a", encoding="utf-8")

    def tick(self, **kw):
        """Append one metrics line; flushed so the watcher sees it immediately."""
        self._fh.write(json.dumps(kw) + "\n")
        self._fh.flush()

    def stop_requested(self) -> bool:
        return (self.dir / "MLGUARD_STOP").exists()

    def lgb_callback(self, fold, every=10):
        """LightGBM callback: logs train/valid loss and honours MLGUARD_STOP.

        Requires valid_sets=[train_ds, valid_ds], valid_names=["train", "valid"].
        """
        def _cb(env):
            if env.iteration % every:
                return
            vals = {name: v for name, _metric, v, _hib in env.evaluation_result_list}
            self.tick(fold=fold, iter=env.iteration,
                      train_loss=vals.get("train"), valid_loss=vals.get("valid"))
            if self.stop_requested():
                import lightgbm as lgb
                raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)
        _cb.order = 30
        return _cb

    def write_folds(self, s1_ids, folds):
        with open(self.dir / "folds.tsv", "w", encoding="utf-8", newline="\n") as fh:
            fh.write("s1_id\tfold\n")
            fh.writelines(f"{i}\t{f}\n" for i, f in zip(s1_ids, folds))

    def write_summary(self, **summary):
        """Required keys: run_id, oof_score, threshold_source, folds (list of
        {fold, train_score, valid_score, best_iter, max_iter}). Optional keys in MLGUARD.md."""
        with open(self.dir / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=float)

    def end(self):
        self.tick(event="end")
        self._fh.close()
