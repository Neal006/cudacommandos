"""v2 pipeline: lean ingest -> cached blocking -> vectorized features -> stage 1 ->
stage 2 -> decision layer -> outputs. Every stage is logged for tools/mlguard.

    python src/run_v2.py --sample 150000                 # full run: train OOF + test outputs
    python src/run_v2.py --sample 30000 --train-only     # OOF only (no test blocking)
    python src/run_v2.py --sample 30000 --train-only --ablate   # + old-features baseline
    tools/mlguard/train_guarded.sh 004_v2 --sample 150000     # PIPELINE=src/run_v2.py

Shares the candidate cache with run_pipeline.py (same key, same blobs), so a test
cache Priyanshu already paid 4-5 h for is reused as-is.
"""
import argparse
import gc
import os
import pickle
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import config as C
import data as D
import decide
import features as F
import features_v2 as F2
import ingest
import stage2 as S2
from metrics import blocking_recall
from run_pipeline import cached_candidates
from runlog import RunLog

T0 = time.time()
PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
              num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, verbosity=-1, seed=C.SEED, num_threads=0)
MAX_ROUNDS = 2000


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def stats_for(which):
    """Label-free S1 statistics for a split, cached (computed from the FULL S1 file)."""
    path = C.INTERIM / f"stats_{which}_v1.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    st = F2.split_stats(D.source_paths(which)[0])
    path.write_bytes(pickle.dumps(st))
    return st


def featurize(pairs, which, stats, extra=True):
    p1, p2, p3 = D.source_paths(which)
    L = F2.record_table(F2.load_records([p1], set(pairs["s1_id"])))
    R = F2.record_table(F2.load_records([p2, p3], set(pairs["cand_id"])))
    X = F2.build_pair_features(pairs, L, R, stats=stats, extra=extra)
    X = F.add_rank_features(X, pairs, "core_token_sort")
    return X, L, R


def entity_f05(pairs, y, p, thr, truth_count):
    sel = pairs.assign(y=y, p=p)
    return decide.macro_f05(sel[sel["p"] >= thr], truth_count)


def fit_cv(X, y, pairs, folds, truth_count, run, stage):
    """GroupKFold LightGBM. Returns OOF scores, fold models, mlguard fold rows."""
    oof = np.zeros(len(X))
    models, rows = [], []
    rng = np.random.default_rng(C.SEED)
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        dtr = lgb.Dataset(X[tr], y[tr])
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        m = lgb.train(PARAMS, dtr, MAX_ROUNDS, valid_sets=[dtr, dva], valid_names=["train", "valid"],
                      callbacks=[lgb.early_stopping(100, verbose=False),
                                 run.lgb_callback(fold=int(stage * 10 + k))])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        # overfit signal for mlguard: F0.5@0.5 on a training-fold entity subsample vs the valid fold
        tr_ents = pairs.loc[tr, "s1_id"].unique()
        sub = np.isin(pairs["s1_id"].to_numpy(), rng.choice(tr_ents, min(5000, len(tr_ents)), replace=False))
        p_tr = m.predict(X[sub], num_iteration=m.best_iteration)
        tc_tr = truth_count.loc[pairs.loc[sub, "s1_id"].unique()]
        tc_va = truth_count.loc[pairs.loc[va, "s1_id"].unique()]
        rows.append(dict(fold=int(stage * 10 + k), best_iter=int(m.best_iteration), max_iter=MAX_ROUNDS,
                         train_score=entity_f05(pairs[sub], y[sub], p_tr, 0.5, tc_tr),
                         valid_score=entity_f05(pairs[va], y[va], oof[va], 0.5, tc_va)))
        models.append(m)
        log(f"stage {stage} fold {k}: best_iter {m.best_iteration}  "
            f"F@0.5 train {rows[-1]['train_score']:.4f} valid {rows[-1]['valid_score']:.4f}")
    return oof, models, rows


def rerank_feature(model_dir, pairs, p1, L, R, band):
    """Cross-encoder probability for pairs with band[0] <= p1 <= band[1]; -1 elsewhere."""
    from gpu import reranker as RR
    m = (p1 >= band[0]) & (p1 <= band[1])
    out = np.full(len(pairs), -1.0)
    t = time.time()
    out[m] = RR.score(model_dir, RR.serialize(L, pairs.loc[m, "s1_id"]), RR.serialize(R, pairs.loc[m, "cand_id"]))
    log(f"reranker: scored {int(m.sum()):,} band pairs ({m.mean():.1%}) in {time.time() - t:.0f}s")
    return out


def predict(models, X):
    return np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0)


def rate_stats(selected, ids, country):
    """Predicted singleton rate and links/entity, overall and per country."""
    n = selected.groupby("s1_id").size().reindex(ids, fill_value=0)
    by = pd.DataFrame({"n": n.to_numpy(), "c": country.reindex(ids).to_numpy()})
    return ({c: float((g["n"] == 0).mean()) for c, g in by.groupby("c")},
            {c: float(g["n"].mean()) for c, g in by.groupby("c")},
            float((by["n"] == 0).mean()), float(by["n"].mean()))


def main(a):
    run_id = a.run_id or time.strftime("v2_%Y%m%d_%H%M")
    run = RunLog(a.run_dir or os.environ.get("MLGUARD_RUN_DIR") or Path(C.ROOT) / "runs" / run_id)
    log(f"run {run_id} -> {run.dir}")

    # ---------------- train: candidates, labels, features
    s1, s2, s3 = ingest.load_split_lean("train", sample=a.sample)
    cands, pairs = cached_candidates(s1, s2, s3, "train", a.sample)
    s1_ids = s1[C.ID].astype(str).tolist()
    country = pd.Series(s1[C.COUNTRY].astype(str).to_numpy(), index=s1_ids)
    del s2, s3, s1
    gc.collect()
    pairs = pairs.reset_index(drop=True)
    truth = D.read_ground_truth(C.TRAIN_GT, keep_ids=s1_ids)
    truth_count = pd.Series({k: len(truth.get(k, ())) for k in s1_ids})
    y = np.fromiter((c in truth[s] for s, c in zip(pairs["s1_id"], pairs["cand_id"])), bool, len(pairs)).astype(int)
    rec = blocking_recall(truth, pairs.groupby("s1_id")["cand_id"].apply(set).to_dict())
    log(f"train: {len(s1_ids):,} entities, {len(pairs):,} pairs, {int(y.sum()):,} positives, "
        f"blocking recall {rec['pair_recall']:.4f}")

    stats = stats_for("train")
    X, L, R = featurize(pairs, "train", stats, extra=True)
    log(f"features {X.shape}")

    grp = pairs["s1_id"].to_numpy()
    folds = np.empty(len(pairs), dtype=int)
    for k, (_, va) in enumerate(GroupKFold(n_splits=C.N_FOLDS).split(X, y, grp)):
        folds[va] = k
    ent_fold = pd.Series(folds, index=grp).groupby(level=0).first()
    run.write_folds(ent_fold.index, ent_fold.to_numpy())

    base_df = pairs[["s1_id", "cand_id"]].assign(y=y)
    report = {}
    if a.ablate:  # the pre-v2 feature set, same folds: the honest baseline
        old_cols = F2.OLD_COLUMNS + ["rank_in_entity", "max_in_entity", "gap_to_best", "n_candidates", "is_best"]
        oof0, _, _ = fit_cv(X[old_cols].to_numpy(), y, pairs, folds, truth_count, run, stage=9)
        _, t0 = decide.tune(base_df.assign(p=oof0), truth_count)
        report["old_features_threshold"] = float(t0[(t0["assign"] == "none") & (t0["select"] == "threshold")]["f05"].max())
        log(f"ablation (old features, global threshold): {report['old_features_threshold']:.4f}")

    # ---------------- stage 1
    feat1 = list(X.columns)
    oof1, models1, rows1 = fit_cv(X.to_numpy(), y, pairs, folds, truth_count, run, stage=1)
    _, t1 = decide.tune(base_df.assign(p=oof1), truth_count)
    report["stage1_threshold"] = float(t1[(t1["assign"] == "none") & (t1["select"] == "threshold")]["f05"].max())
    log(f"stage 1 (v2 features, global threshold): {report['stage1_threshold']:.4f}")

    # ---------------- optional GPU reranker on the uncertain band (feature for stage 2)
    rr = None
    if a.rerank:
        seen = set((Path(a.rerank) / "entities.txt").read_text(encoding="utf-8").split())
        leak = seen & set(s1_ids)
        if leak:
            raise SystemExit(f"reranker was trained on {len(leak)} entities of this sample — leakage; "
                             f"retrain with --exclude {run.dir / 'folds.tsv'}")
        rr = rerank_feature(a.rerank, pairs, oof1, L, R, a.band)

    # ---------------- stage 2
    final, models2, rows, feat2 = oof1, None, rows1, None
    if not a.no_stage2:
        X2 = pd.concat([X, S2.build(pairs, oof1, R)], axis=1)
        if rr is not None:
            X2["rr"] = rr
        feat2 = list(X2.columns)
        oof2, models2, rows2 = fit_cv(X2.to_numpy(), y, pairs, folds, truth_count, run, stage=2)
        _, t2 = decide.tune(base_df.assign(p=oof2), truth_count)
        report["stage2_threshold"] = float(t2[(t2["assign"] == "none") & (t2["select"] == "threshold")]["f05"].max())
        log(f"stage 2 (global threshold): {report['stage2_threshold']:.4f}")
        final, rows = oof2, rows2
        imp = dict(zip(feat2, models2[0].feature_importance("gain")))
    else:
        imp = dict(zip(feat1, models1[0].feature_importance("gain")))

    # ---------------- decision layer (calibration cross-fitted by fold)
    cal = decide.crossfit_calibrate(final, y, folds)
    best, table = decide.tune(base_df.assign(p=cal), truth_count)
    table.to_csv(run.dir / "decision_table.csv", index=False)
    report["decision_best"] = float(best["f05"])
    log(f"decision layer best: {best}")
    sel = decide.apply(base_df.assign(p=cal), best)
    per_country = {c: decide.macro_f05(sel[sel["s1_id"].map(country) == c], truth_count[country == c])
                   for c in sorted(country.unique())}
    _, _, oof_sing, oof_links = rate_stats(sel, s1_ids, country)
    log(f"per country: {per_country}  predicted singletons {oof_sing:.3f}  links/entity {oof_links:.2f}")

    summary = dict(run_id=run_id, oof_score=report["decision_best"], threshold_source="oof", folds=rows,
                   blocking_recall=rec["pair_recall"], per_country=per_country,
                   oof_pred_singleton_rate=oof_sing,
                   true_singleton_rate=float((truth_count == 0).mean()),
                   oof_pred_links_per_entity=oof_links,
                   feature_importance={k: float(v) for k, v in imp.items()},
                   decision=best, ablation=report, sample=a.sample)
    run.write_summary(**summary)
    calibrator = decide.fit_calibrator(final, y)
    (run.dir / "model.pkl").write_bytes(pickle.dumps(dict(
        models1=models1, models2=models2, feat1=feat1, feat2=feat2, calibrator=calibrator, decision=best)))
    del X, L, R, pairs
    gc.collect()

    if a.train_only:
        run.end()
        log("train-only: done")
        return

    # ---------------- test
    t1_, t2_, t3_ = ingest.load_split_lean("test")
    test_ids = t1_[C.ID].astype(str).tolist()
    t_country = pd.Series(t1_[C.COUNTRY].astype(str).to_numpy(), index=test_ids)
    t_cands, t_pairs = cached_candidates(t1_, t2_, t3_, "test", None)
    del t1_, t2_, t3_
    gc.collect()
    t_pairs = t_pairs.reset_index(drop=True)
    TX, TL, TR = featurize(t_pairs, "test", stats_for("test"), extra=True)
    p = predict(models1, TX[feat1].to_numpy())
    if models2 is not None:
        TX2 = pd.concat([TX, S2.build(t_pairs, p, TR)], axis=1)
        if "rr" in feat2:
            TX2["rr"] = rerank_feature(a.rerank, t_pairs, p, TL, TR, a.band)
        p = predict(models2, TX2[feat2].to_numpy())
    tdf = t_pairs[["s1_id", "cand_id"]].assign(p=calibrator.predict(p))
    tsel = decide.apply(tdf, best)
    matches = tsel.groupby("s1_id")["cand_id"].apply(set).to_dict()
    cand_sets = t_pairs.groupby("s1_id")["cand_id"].apply(set).to_dict()
    out = D.write_outputs(test_ids, {s: matches.get(s, set()) for s in test_ids},
                          {s: cand_sets.get(s, set()) for s in test_ids})
    sing, links, _, _ = rate_stats(tsel, test_ids, t_country)
    summary.update(test_pred_singleton_rate=sing, test_pred_links_per_entity=links)
    run.write_summary(**summary)
    run.end()
    log(f"test outputs: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=C.TRAIN_SAMPLE)
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--no-stage2", action="store_true")
    ap.add_argument("--ablate", action="store_true", help="also train on the pre-v2 features (baseline)")
    ap.add_argument("--rerank", default=None, help="reranker model dir (src/gpu/reranker.py train)")
    ap.add_argument("--band", type=float, nargs=2, default=(0.2, 0.8), help="stage-1 band sent to the reranker")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--run-dir", default=None)
    main(ap.parse_args())
