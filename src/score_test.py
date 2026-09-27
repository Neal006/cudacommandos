"""Score the test set from an already-trained run, and write the submission.

run_v2.py has no resume: asked for a submission it retrains stages 1 and 2
from scratch (~60 min at --sample 150000) even when the models it is about to
rebuild are already sitting in runs/<id>/model.pkl. That is an hour of compute
to arrive back where we started, and it gives a *different* model -- LightGBM
is seeded, but mlguard's watch can stop a fold mid-run, so a retrain is not
guaranteed to reproduce the run whose OOF score we reported.

This scores test with the exact models that produced that score:

    python src/score_test.py --run runs/007_v2_full

It reuses run_v2.predict_test_chunked, so the feature path, the chunking and
the stage-2 claim handling are the same code the training run would have used
-- this module only skips the training.

Memory: the candidate frame is ~5.3 GB, the S1 record table ~1.5 GB and the
polars source pool ~1.2 GB, so ~8 GB is held for the duration. Each chunk adds
its own record table and feature matrix on top. --chunk trades that per-chunk
peak against per-chunk overhead (each chunk costs one filter over the 10M-row
source pool). 2M is sized for a 12 GB budget.
"""
import argparse
import gc
import hashlib
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
import data as D
import decide
import ingest
from run_pipeline import cached_candidates
from run_v2 import log, predict_test_chunked, rate_stats, stats_for


def main(a):
    run_dir = Path(a.run)
    bundle = pickle.loads((run_dir / "model.pkl").read_bytes())
    models1, feat1 = bundle["models1"], bundle["feat1"]
    models2, feat2 = bundle.get("models2"), bundle.get("feat2")
    calibrator, best = bundle["calibrator"], bundle["decision"]
    log(f"loaded {run_dir/'model.pkl'}: {len(models1)} stage-1 models, "
        f"{len(models2) if models2 else 0} stage-2, decision {best}")

    needs_rerank = models2 is not None and "rr" in feat2
    if needs_rerank and not a.rerank:
        raise SystemExit("this model was trained with the band reranker; pass --rerank <dir>")
    if a.rerank and not needs_rerank:
        raise SystemExit("--rerank given but this model has no 'rr' feature")
    if needs_rerank:
        log(f"band reranker: {a.rerank}, band {a.band}")

    t1_, t2_, t3_ = ingest.load_split_lean("test")
    test_ids = t1_[C.ID].astype(str).tolist()
    t_country = pd.Series(t1_[C.COUNTRY].astype(str).to_numpy(), index=test_ids)
    _, t_pairs = cached_candidates(t1_, t2_, t3_, "test", None, frame_only=True)
    del t1_, t2_, t3_
    gc.collect()
    t_pairs = t_pairs.reset_index(drop=True)

    # Every Source-1 entity must appear in the output even with no candidates,
    # or the submission is rejected. write_outputs iterates test_ids rather
    # than the prediction frame, so this is a report, not a repair.
    orphans = len(test_ids) - t_pairs["s1_id"].nunique()
    log(f"test: {len(t_pairs):,} pairs over {t_pairs['s1_id'].nunique():,} entities; "
        f"{orphans:,} of {len(test_ids):,} got no candidates and will be written empty")

    t0 = time.time()
    # Cache key: anything that changes the scores must change the filename, or
    # a rerun silently reuses another model's predictions. The model file's
    # size and mtime stand in for hashing 40 MB of pickle on every start.
    mp = run_dir / "model.pkl"
    key = hashlib.sha1(
        f"{run_dir.name}|{mp.stat().st_size}|{int(mp.stat().st_mtime)}|"
        f"{a.rerank}|{tuple(a.band)}|{len(t_pairs)}".encode()
    ).hexdigest()[:12]
    cache_p1 = C.INTERIM / f"testp1_{key}.npy"
    cache_p = C.INTERIM / f"testp_{key}.npy"
    log(f"score cache key {key}  (stage-1 {cache_p1.name}, final {cache_p.name})")

    p = predict_test_chunked(t_pairs, stats_for("test"), models1, feat1,
                             models2, feat2, calibrator, chunk=a.chunk,
                             rerank_dir=a.rerank if needs_rerank else None,
                             band=tuple(a.band),
                             cache_p1=cache_p1, cache_p=cache_p)
    log(f"scored {len(p):,} pairs in {(time.time()-t0)/60:.1f} min")

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
    del cand_sets, matches
    gc.collect()

    # rate_stats returns the PER-COUNTRY dicts first, then the two scalars.
    sing_by, links_by, sing, links = rate_stats(tsel, test_ids, t_country)
    log(f"predicted singleton rate {sing:.4f}  links/entity {links:.2f}")
    for c in sorted(links_by):
        log(f"  {c}: singleton {sing_by[c]:.4f}  links/entity {links_by[c]:.2f}")
    for k, v in out.items():
        log(f"  {k}: {v}")
    if out["entities_with_matches_outside_candidates"]:
        raise SystemExit("matches outside candidates -- submission would be rejected")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir holding model.pkl, e.g. runs/007_v2_full")
    ap.add_argument("--chunk", type=int, default=C.TEST_CHUNK_PAIRS,
                    help="pairs per entity-aligned chunk (lower = less RAM, more overhead)")
    ap.add_argument("--rerank", default=None, help="band reranker dir, if the model was trained with one")
    ap.add_argument("--band", type=float, nargs=2, default=(0.2, 0.8),
                    help="stage-1 band sent to the reranker; must match training")
    main(ap.parse_args())
