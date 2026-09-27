"""End-to-end: blocking -> pairwise matcher -> thresholded matches -> TSVs.

    python src/run_pipeline.py --blocking-only   # recall ceiling at several K
    python src/run_pipeline.py                   # full run, writes output/

Order matters. Blocking recall caps everything downstream, so it is reported
before a single model is trained: if the ceiling is low, tuning the matcher is
wasted time and TOP_K / BLOCK_MAX_DF are the knobs to turn.

Scale notes: training subsamples Source-1 entities (C.TRAIN_SAMPLE) because a
pairwise matcher does not need 2.2M of them and the RAM here will not hold
them. Test blocking and inference always run over the FULL test set.
"""
import argparse
import gc
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

import config as C
import blocking
import data as D
import features as F
from normalize import add_feature_columns
from metrics import macro_f_beta, scores_breakdown, blocking_recall

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def cached_candidates(s1, s2, s3, which, sample, frame_only=False):
    """Blocking, memoized to disk.

    Test blocking is ~52 min (run 007: France 1066 q/s over 1.43M records,
    US 800 over 3.82M, India 397 over 4.72M). Recomputing it on every matcher
    experiment would still make iteration painful, so the flat candidate frame
    is written to parquet and keyed on every parameter that changes the result.

    `frame_only=True` returns (None, frame) and skips rebuilding the dict form.
    That rebuild is ~5 GB of Python objects on test -- 52M tuples, 52M float
    objects and 1.73M lists -- and run_v2 never reads it. Ask for it only if
    you actually want it.
    """
    key = (f"{which}_k{C.TOP_K}_df{C.BLOCK_MAX_DF}_mdf{C.BLOCK_MIN_DF}"
       f"_ctry{int(C.BLOCK_WITHIN_COUNTRY)}_n{sample or 'all'}_union1")
    path = C.INTERIM / f"cands_{key}.parquet"

    if path.exists():
        log(f"loading cached candidates: {path.name}")
        frame = pd.read_parquet(path)
    else:
        from blocking_union import generate_candidates_union
        cands, skel_source_ids = generate_candidates_union(s1, s2, s3, which)
        frame = blocking.candidates_to_frame(cands)
        frame["is_skel_source"] = frame["cand_id"].isin(skel_source_ids).astype(int)
        assert "is_skel_source" in frame.columns
        frame.to_parquet(path, index=False)
        log(f"cached candidates -> {path.name} ({len(frame):,} pairs, "
            f"{frame['is_skel_source'].sum():,} skel-only)")
        return (None if frame_only else cands), frame

    if frame_only:
        return None, frame

    # rebuild the dict form, preserving the every-entity-gets-a-key invariant
    cands = {sid: [] for sid in s1[C.ID]}
    for sid, grp in frame.groupby("s1_id", sort=False):
        cands[sid] = list(zip(grp["cand_id"], grp["block_sim"]))
    return cands, frame


def recall_at_k(truth, cands, ks=(5, 10, 20, 30, 50)):
    """Recall ceiling as a function of candidates kept per entity.

    Cheaper than re-running blocking per K: candidates already come back sorted
    by similarity, so truncating the list simulates a smaller TOP_K.
    """
    rows = []
    for k in ks:
        if k > C.TOP_K:
            continue
        truncated = {sid: {c for c, _ in v[:k]} for sid, v in cands.items()}
        r = blocking_recall(truth, truncated)
        rows.append({"K": k, "pair_recall": round(r["pair_recall"], 4),
                     "mean_candidates": round(r["mean_candidates"], 1)})
    return pd.DataFrame(rows)


def featurize(pairs, which):
    """Build pair features, re-reading source text for surviving records only.

    Blocking drops business_name / business_address to stay inside the RAM
    budget, so the text is streamed back from disk here — for the fraction of
    records that actually appear in a candidate pair, not all 10M.
    """
    p1, p2, p3 = D.source_paths(which)
    need_s1 = set(pairs["s1_id"].unique())
    need_ot = set(pairs["cand_id"].unique())
    log(f"re-reading text for {len(need_s1):,} S1 and {len(need_ot):,} S2/S3 records")

    s1f = add_feature_columns(D.load_records_by_id(p1, need_s1))
    otf = add_feature_columns(pd.concat(
        [D.load_records_by_id(p2, need_ot), D.load_records_by_id(p3, need_ot)],
        ignore_index=True,
    ))
    feat = F.build_pair_features(pairs, s1f, otf)
    del s1f, otf
    gc.collect()
    return F.add_rank_features(feat, pairs, "core_token_sort")


def tune_threshold(truth, pairs, scores, s1_ids):
    """Sweep the decision threshold for macro F_0.5.

    F_0.5 weights precision 2x, so the optimum sits above 0.5 — but only 5.6%
    of entities are singletons and the mean is 3.46 true links, so pushing the
    threshold too high costs far more than it protects. Sweeping settles it.
    """
    best = (C.DEFAULT_THRESHOLD, -1.0)
    df = pairs.assign(score=scores)
    for thr in np.arange(0.05, 0.96, 0.01):
        keep = df[df["score"] >= thr]
        preds = keep.groupby("s1_id")["cand_id"].apply(set).to_dict()
        s = macro_f_beta(truth, {sid: preds.get(sid, set()) for sid in s1_ids})
        if s > best[1]:
            best = (float(thr), s)
    return best


def main(blocking_only=False, sample=None):
    missing = D.check_test_files()
    if missing:
        print(f"\n*** WARNING: missing test files {missing} — "
              f"no candidate from that source can be predicted. "
              f"A submission built now will lose every match from it. ***\n")

    sample = C.TRAIN_SAMPLE if sample is None else (sample or None)
    s1, s2, s3 = D.load_split("train", sample=sample)
    truth = D.read_ground_truth(C.TRAIN_GT, keep_ids=s1[C.ID])
    log(f"train: S1={len(s1):,} (sampled) S2={len(s2):,} S3={len(s3):,} gt={len(truth):,}")

    cands, pairs = cached_candidates(s1, s2, s3, "train", sample)
    log("train blocking done")

    cand_sets = {k: {c for c, _ in v} for k, v in cands.items()}
    rec = blocking_recall(truth, cand_sets)
    log(f"BLOCKING RECALL CEILING @K={C.TOP_K}: {rec['pair_recall']:.4f} "
        f"({rec['mean_candidates']:.1f} candidates/entity)")
    print(recall_at_k(truth, cands).to_string(index=False))
    print(f"  orphans: {blocking.orphan_check(cands, s1[C.ID].tolist())['n_orphans']:,}")
    print("  -> no matcher can exceed the ceiling. Raise TOP_K / BLOCK_MAX_DF if short.")
    if blocking_only:
        return

    # S2/S3 are done with — featurize() re-reads the text it needs from disk.
    del s2, s3
    gc.collect()

    pairs = pairs.copy()
    pairs["y"] = [
        1 if cid in truth.get(sid, set()) else 0
        for sid, cid in zip(pairs["s1_id"], pairs["cand_id"])
    ]
    log(f"positives {int(pairs['y'].sum()):,} / {len(pairs):,} pairs")

    X = featurize(pairs, "train")
    y = pairs["y"].values
    groups = pairs["s1_id"].values
    log(f"features built: {X.shape}")

    params = dict(
        objective="binary", metric="binary_logloss", learning_rate=0.05,
        num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=1, verbosity=-1, seed=C.SEED,
        num_threads=0,
    )

    oof = np.zeros(len(X))
    models = []
    # Group by Source-1 entity so an entity's pairs never straddle folds.
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=C.N_FOLDS).split(X, y, groups)):
        m = lgb.train(
            params, lgb.Dataset(X.iloc[tr], y[tr]), num_boost_round=2000,
            valid_sets=[lgb.Dataset(X.iloc[va], y[va])],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        models.append(m)
        log(f"fold {fold} done (best_iter={m.best_iteration})")

    s1_ids = s1[C.ID].tolist()
    thr, cv = tune_threshold(truth, pairs, oof, s1_ids)
    log(f"BEST THRESHOLD {thr:.2f} -> OOF macro F_0.5 = {cv:.4f}")

    keep = pairs.assign(score=oof)
    keep = keep[keep["score"] >= thr]
    preds = keep.groupby("s1_id")["cand_id"].apply(set).to_dict()
    bd = scores_breakdown(truth, {sid: preds.get(sid, set()) for sid in s1_ids})
    print("  breakdown:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in bd.items()})

    imp = sorted(zip(X.columns, models[0].feature_importance("gain")),
                 key=lambda t: -t[1])[:12]
    print("  top features:", [f"{k}={v:.0f}" for k, v in imp])

    del X, pairs, cands, cand_sets, s1
    gc.collect()

    # --- test: full set, never sampled ---
    t1, t2, t3 = D.load_split("test")
    log(f"test: S1={len(t1):,} S2={len(t2):,} S3={len(t3):,}")
    t_cands, t_pairs = cached_candidates(t1, t2, t3, "test", None)
    log("test blocking done")

    del t2, t3
    gc.collect()

    log(f"test pairs: {len(t_pairs):,}")
    TX = featurize(t_pairs, "test")
    TX = TX.reindex(columns=models[0].feature_name(), fill_value=0.0)
    scores = np.mean([m.predict(TX, num_iteration=m.best_iteration) for m in models], axis=0)

    sel = t_pairs.assign(score=scores)
    sel = sel[sel["score"] >= thr]
    matches = sel.groupby("s1_id")["cand_id"].apply(set).to_dict()

    test_ids = t1[C.ID].tolist()
    summary = D.write_outputs(
        test_ids,
        {sid: matches.get(sid, set()) for sid in test_ids},
        {sid: {c for c, _ in t_cands.get(sid, [])} for sid in test_ids},
    )
    log("wrote outputs")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nValidate before uploading (costs nothing, a rejection costs a submission):")
    print("  python validate_submission.py --matching output/matching_results.tsv \\")
    print("      --candidate output/candidate_pairs.tsv --test-dir <dataset/test>")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocking-only", action="store_true",
                    help="measure the recall ceiling at several K and stop")
    ap.add_argument("--sample", type=int, default=None,
                    help="Source-1 entities to train on (0 = all)")
    a = ap.parse_args()
    main(blocking_only=a.blocking_only, sample=a.sample)
