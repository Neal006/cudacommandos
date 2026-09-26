"""Smoke test: runlog.py writes what mlguard reads. Run: python tools/mlguard/test_runlog.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from runlog import RunLog  # noqa: E402


class _Env:  # the slice of lightgbm's CallbackEnv the callback reads
    def __init__(self, it, tr, va):
        self.iteration = it
        self.evaluation_result_list = [("train", "binary_logloss", tr, False),
                                       ("valid", "binary_logloss", va, False)]


with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
    log = RunLog(d)
    cb = log.lgb_callback(fold=0, every=1)
    for i in range(3):
        cb(_Env(i, 0.5 - 0.1 * i, 0.55 - 0.1 * i))
    assert not log.stop_requested()
    log.write_folds(["S1-1", "S1-2"], [0, 1])
    log.write_summary(run_id="t", oof_score=0.9, threshold_source="oof",
                      folds=[{"fold": 0, "train_score": 0.91, "valid_score": 0.9}])
    log.end()
    lines = [json.loads(x) for x in (Path(d) / "metrics.jsonl").read_text().splitlines()]
    assert lines[0] == {"fold": 0, "iter": 0, "train_loss": 0.5, "valid_loss": 0.55}, lines[0]
    assert lines[-1] == {"event": "end"}
    assert json.loads((Path(d) / "summary.json").read_text())["threshold_source"] == "oof"
    (Path(d) / "MLGUARD_STOP").write_text("3\tdiverging_valid: fold 3 iter 90\n")
    assert log.stop_requested() and log.stop_requested(3) and not log.stop_requested(0)
    cb3 = RunLog(d).lgb_callback(fold=3, every=10)
    try:
        for i, va in enumerate([0.3, 0.2, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32], start=1):
            cb3(_Env(i, 0.1, va))   # best valid at iter 2; the stop is read at iter 10
        raise AssertionError("fold 3 should have been stopped")
    except Exception as e:  # lightgbm's EarlyStopException
        assert type(e).__name__ == "EarlyStopException", e
        assert e.best_iteration == 2, e.best_iteration      # rolled back, not iter 10
    RunLog(d).write_summary(run_id="t", oof_score=0.9, threshold_source="oof", folds=[],
                            decision={"thr": float("nan"), "miss": float("inf")})
    assert json.loads((Path(d) / "summary.json").read_text())["decision"] == {"thr": None, "miss": None}
    RunLog(d).lgb_callback(fold=0, every=1)(_Env(10, 0.1, 0.2))  # other folds keep training
print("runlog ok")
