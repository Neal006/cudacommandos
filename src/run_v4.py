"""v4: train stage 2 with the contention it will actually meet at inference.

THE BUG THIS FIXES
------------------
Stage 2's competition features are raw counts over whatever Source-1 set
happens to be present:

    n_claims          how many S1 entities list this record as a candidate
    n_strong_claims   how many of them score it >= 0.5
    claim_rank        this entity's rank among those claimers

We train them with a 150k-entity sample and apply them to a test set with
1,732,544 entities. Measured mean n_claims:

    train  30k  1.371
    train 150k  1.957
    TEST        5.549          <- 2.8x the training distribution

`n_strong_claims` is the SECOND most important feature in stage 2 (importance
2.52M, behind only p1's 14.35M). Our most important context signal means a
different thing at inference than it did in training.

This is invisible to cross-validation, because the out-of-fold split carries
the same wrong contention as the training data. It can only show up on the
leaderboard -- and a 0.0102 OOF-to-leaderboard gap is exactly what we saw.

THE FIX
-------
Compute the claim features over the FULL Source-1 set (~2.2M entities), while
still training the GBDT on a sample. Train-time contention becomes ~6.6
against test's 5.55 -- close -- instead of 1.96.

    A  block every train S1 entity                    -> ~66M pairs (cached)
    B  train stage 1 on the sampled rows only          (unchanged, OOF)
    C  score stage 1 over ALL 66M pairs                 chunked
    D  build claim features over the whole frame        the point of all this
    E  train stage 2 on the sampled rows, with those claims
    F  tune the decision layer, score test

Leakage: sampled rows keep their OUT-OF-FOLD p1 in step C, exactly as
run_v2 does. Non-sampled rows are scored by the mean of the fold models,
which is sound because they are never training targets -- they exist only to
create realistic competition for the records the sampled entities want.

This also subsumes "train on more entities": raising the sample helps partly
*because* it moves contention toward test, and this gets that effect in full
without paying to train on 2.2M entities.

    python src/run_v4.py --sample 150000 --rerank models/rr_e5s

Written as a separate module so run_v2.py is untouched and still runnable.
"""
import argparse
import gc
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
import run_v2 as V2
from run_v2 import (entity_f05, featurize, fit_cv, log, predict,
                    predict_test_chunked, rate_stats, rerank_feature, stats_for)


def _codes(s: pd.Series) -> np.ndarray:
    """Stable integer codes for a string column: 66M object pointers -> int32."""
    return pd.factorize(s, sort=False)[0].astype(np.int32)


def score_frame_chunked(pairs, which, stats, models, feats, chunk, log_every=5):
    """Stage-1 score for every row of `pairs`, streaming features chunk by chunk.

    Same shape as run_v2.predict_test_chunked's first pass, but for any split.
    Nothing here is order-dependent beyond add_rank_features, which groups by
    s1_id -- so chunks are cut only where s1_id changes.
    """
    import polars as pl
    from ingest import read_polars

    p1p, p2p, p3p = D.source_paths(which)
    L = F2.record_table(F2.load_records([p1p], set(pairs["s1_id"])))
    src = pl.concat([read_polars(p2p), read_polars(p3p)])

    s = pairs["s1_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    bounds, lo = [], 0
    for b in starts[1:]:
        if b - lo >= chunk:
            bounds.append((lo, int(b)))
            lo = int(b)
    bounds.append((lo, len(pairs)))
    log(f"scoring {len(pairs):,} {which} pairs in {len(bounds)} chunks "
        f"(L {len(L):,}, source pool {len(src):,})")

    out = np.empty(len(pairs), dtype=np.float32)
    for n, (a, b) in enumerate(bounds, 1):
        sl = pairs.iloc[a:b]
        ids = pl.Series(sl["cand_id"].unique(), dtype=pl.String)
        R = F2.record_table(src.filter(pl.col(C.ID).is_in(ids.implode())).to_pandas())
        X = F2.build_pair_features(sl, L, R, stats=stats, extra=True)
        X = F.add_rank_features(X, sl, "core_token_sort")
        out[a:b] = predict(models, X[feats].to_numpy()).astype(np.float32)
        del X, R
        gc.collect()
        if n % log_every == 0 or n == len(bounds):
            log(f"  chunk {n}/{len(bounds)} rows {a:,}-{b:,}")
    del L, src
    gc.collect()
    return out


def main(a):
    # fit_cv reads run_v2.MAX_ROUNDS directly. Fold 1 hit the 2000 default at
    # 150k -- it was still improving when the cap cut it off -- so --rounds
    # raises it for this process only. run_v2.py on disk is untouched.
    if a.rounds != V2.MAX_ROUNDS:
        log(f"raising MAX_ROUNDS {V2.MAX_ROUNDS} -> {a.rounds}")
        V2.MAX_ROUNDS = a.rounds
    run_id = a.run_id or time.strftime("v4_%Y%m%d_%H%M")
    run = RunLog(Path(os.environ.get("MLGUARD_RUN_DIR") or (Path(C.ROOT) / "runs" / run_id)))
    log(f"run {run_id} -> {run.dir}")
    summary = {"run_id": run_id, "sample": a.sample, "variant": "v4_full_contention"}

    # ---------------- A. the full frame, and the sampled subset inside it
    s1_all, s2, s3 = ingest.load_split_lean("train")
    _, full_pairs = cached_candidates(s1_all, s2, s3, "train", None, frame_only=True)
    full_pairs = full_pairs.sort_values("s1_id", kind="mergesort").reset_index(drop=True)
    log(f"full train frame: {len(full_pairs):,} pairs over {full_pairs['s1_id'].nunique():,} entities")

    # country for the sampled entities, kept before the big frames are dropped
    country_of = pd.Series(s1_all[C.COUNTRY].astype(str).to_numpy(),
                           index=s1_all[C.ID].astype(str).to_numpy())
    s1_s, _, _ = ingest.load_split_lean("train", sample=a.sample)
    sample_ids = set(s1_s[C.ID].astype(str))
    del s1_all, s2, s3, s1_s
    gc.collect()

    in_sample = full_pairs["s1_id"].isin(sample_ids).to_numpy()
    pairs = full_pairs[in_sample].reset_index(drop=True)
    log(f"sampled subset: {len(pairs):,} pairs over {pairs['s1_id'].nunique():,} entities")

    cand_codes_full = _codes(full_pairs["cand_id"])
    log(f"contention: full frame mean n_claims "
        f"{len(full_pairs) / len(np.unique(cand_codes_full)):.3f}  "
        f"(sample-only would be {len(pairs) / pairs['cand_id'].nunique():.3f}, test is 5.549)")

    # ---------------- labels and folds, on the sample only
    s1_ids = sorted(sample_ids)
    truth = D.read_ground_truth(C.TRAIN_GT, keep_ids=s1_ids)
    truth_count = pd.Series({k: len(truth.get(k, ())) for k in s1_ids})
    country = pd.Series(country_of.reindex(s1_ids).to_numpy(), index=s1_ids)
    y = np.fromiter((c in truth[s] for s, c in zip(pairs["s1_id"], pairs["cand_id"])),
                    bool, len(pairs)).astype(int)
    rec = blocking_recall(truth, pairs.groupby("s1_id")["cand_id"].apply(set).to_dict())
    log(f"train: {len(s1_ids):,} entities, {len(pairs):,} pairs, {int(y.sum()):,} positives, "
        f"blocking recall {rec['pair_recall']:.4f}")

    stats = stats_for("train")
    X, L, R = featurize(pairs, "train", stats, extra=True)
    feat1 = list(X.columns)
    log(f"features {X.shape}")

    grp = pairs["s1_id"].to_numpy()
    folds = np.empty(len(pairs), dtype=int)
    for k, (_, va) in enumerate(GroupKFold(n_splits=C.N_FOLDS).split(X, y, grp)):
        folds[va] = k
    ent_fold = pd.Series(folds, index=grp).groupby(level=0).first()
    run.write_folds(ent_fold.index, ent_fold.to_numpy())
    base_df = pairs[["s1_id", "cand_id"]].assign(y=y)

    # ---------------- B. stage 1 on the sample, out of fold
    p1_oof, models1, rows1 = fit_cv(X.to_numpy(), y, pairs, folds, truth_count, run, stage=1)
    _, t1 = decide.tune(base_df.assign(p=p1_oof), truth_count)
    s1_score = float(t1["f05"].max())
    log(f"stage 1: {s1_score:.4f}")
    del X
    gc.collect()

    # ---------------- C. stage 1 over the WHOLE frame
    # Sampled rows keep their out-of-fold score; everything else is scored by
    # the fold mean. Those rows are never training targets -- they exist only
    # to create the competition test will actually have.
    t0 = time.time()
    p1_full = score_frame_chunked(full_pairs, "train", stats, models1, feat1, a.chunk)
    p1_full[in_sample] = p1_oof.astype(np.float32)
    log(f"full-frame stage-1 scoring done in {(time.time() - t0) / 60:.1f} min")

    # ---------------- D. claim features over the whole frame
    claims_full = S2.build_claims(
        pd.DataFrame({"cand_id": cand_codes_full}), p1_full).astype(np.float32)
    claims = claims_full[in_sample].reset_index(drop=True)
    naive = S2.build_claims(pairs[["cand_id"]], p1_oof)
    log(f"claims: full-frame mean n_claims {claims['n_claims'].mean():.3f} "
        f"vs sample-only {naive['n_claims'].mean():.3f} (test is 5.549)")
    summary["train_mean_n_claims"] = float(claims["n_claims"].mean())
    summary["train_mean_n_claims_naive"] = float(naive["n_claims"].mean())
    del claims_full, p1_full, cand_codes_full, full_pairs, naive
    gc.collect()

    # ---------------- E. stage 2 on the sample, with honest claims
    X, _, _ = featurize(pairs, "train", stats, extra=True)
    X2 = pd.concat([X.reset_index(drop=True),
                    S2.build(pairs, p1_oof, R, claims=claims).reset_index(drop=True)], axis=1)
    del X
    gc.collect()
    if a.rerank:
        X2["rr"] = rerank_feature(a.rerank, pairs, p1_oof, L, R, a.band)
    feat2 = list(X2.columns)
    log(f"stage 2 features {X2.shape}")
    p2_oof, models2, rows2 = fit_cv(X2.to_numpy(), y, pairs, folds, truth_count, run, stage=2)
    _, t2 = decide.tune(base_df.assign(p=p2_oof), truth_count)
    s2_score = float(t2["f05"].max())
    log(f"stage 2: {s2_score:.4f}")
    summary["stage1"] = s1_score
    summary["stage2"] = s2_score

    # ---------------- F. decision layer
    cal = decide.crossfit_calibrate(p2_oof, y, folds)
    best, table = decide.tune(base_df.assign(p=cal), truth_count)
    table.to_csv(run.dir / "decision_table.csv", index=False)
    log(f"decision layer best: {best}")
    sel = decide.apply(base_df.assign(p=cal), best)
    per_country = {c: decide.macro_f05(sel[sel["s1_id"].map(country) == c], truth_count[country == c])
                   for c in sorted(country.unique())}
    _, _, oof_sing, oof_links = rate_stats(sel, s1_ids, country)
    log(f"per country: {per_country}  singletons {oof_sing:.3f}  links/entity {oof_links:.2f}")

    calibrator = decide.fit_calibrator(p2_oof, y)
    summary.update(oof_score=float(best["f05"]), threshold_source="oof", folds=rows1 + rows2,
                   blocking_recall=rec["pair_recall"], per_country=per_country,
                   oof_pred_singleton_rate=oof_sing, oof_pred_links_per_entity=oof_links,
                   true_singleton_rate=float((truth_count == 0).mean()), decision=best)
    run.write_summary(**summary)
    (run.dir / "model.pkl").write_bytes(pickle.dumps(dict(
        models1=models1, models2=models2, feat1=feat1, feat2=feat2,
        calibrator=calibrator, decision=best)))
    log(f"OOF {best['f05']:.4f}  (stage1 {s1_score:.4f} -> stage2 {s2_score:.4f})")

    if a.train_only:
        run.end()
        return

    # ---------------- G. test. Its claims are already right: test blocking
    # covers every test S1 entity, so nothing needs correcting there.
    del X2, L, R, pairs, base_df
    gc.collect()
    t1_, t2_, t3_ = ingest.load_split_lean("test")
    test_ids = t1_[C.ID].astype(str).tolist()
    t_country = pd.Series(t1_[C.COUNTRY].astype(str).to_numpy(), index=test_ids)
    _, t_pairs = cached_candidates(t1_, t2_, t3_, "test", None, frame_only=True)
    del t1_, t2_, t3_
    gc.collect()
    t_pairs = t_pairs.reset_index(drop=True)
    p = predict_test_chunked(t_pairs, stats_for("test"), models1, feat1, models2, feat2,
                             calibrator, chunk=a.chunk,
                             rerank_dir=a.rerank, band=tuple(a.band),
                             cache_p1=C.INTERIM / f"v4p1_{run_id}.npy",
                             cache_p=C.INTERIM / f"v4p_{run_id}.npy")
    tdf = t_pairs[["s1_id", "cand_id"]].assign(p=p)
    tsel = decide.apply(tdf, best)
    del tdf
    gc.collect()
    cand_sets = t_pairs.groupby("s1_id")["cand_id"].apply(set).to_dict()
    del t_pairs
    gc.collect()
    matches = tsel.groupby("s1_id")["cand_id"].apply(set).to_dict()
    out = D.write_outputs(test_ids, {s: matches.get(s, set()) for s in test_ids},
                          {s: cand_sets.get(s, set()) for s in test_ids})
    sing_by, links_by, sing, links = rate_stats(tsel, test_ids, t_country)
    log(f"test: singleton {sing:.4f}  links/entity {links:.2f}")
    for c in sorted(links_by):
        log(f"  {c}: singleton {sing_by[c]:.4f}  links/entity {links_by[c]:.2f}")
    for k, v in out.items():
        log(f"  {k}: {v}")
    summary["test_pred_singleton_rate"] = sing
    summary["test_pred_links_per_entity"] = links
    run.write_summary(**summary)
    run.end()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=C.TRAIN_SAMPLE)
    ap.add_argument("--rounds", type=int, default=4000,
                    help="boosting cap; fold 1 hit the default at 150k, so raise it")
    ap.add_argument("--chunk", type=int, default=1_200_000)
    ap.add_argument("--rerank", default=None)
    ap.add_argument("--band", type=float, nargs=2, default=(0.2, 0.8))
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--run-id", default=None)
    main(ap.parse_args())
