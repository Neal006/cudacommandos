"""Stage C: cheap features on the blocking union + prune model -> pruned candidate set.

The union is ~100 candidates per S1 (200M+ pairs), so this stage streams:
  pass 1: cheap features on a training sample of Q (all positives + all hard negatives + 10% of the other
          negatives, weight 10) -> 5 fold models (GroupKFold by S1)
  pass 2: stream the union in S1-aligned chunks, compute cheap features, score
          (Q rows: their own fold's model = out-of-fold; H / test: average of fold models), keep a pair if
          p >= PRUNE_MIN_P (per country if set) and it is in its S1's top PRUNE_K or in the top PRUNE_K_PER_SRC
          of its source (S2 / S3). Then add back each pool record's best S1 (PRUNE_KEEP_CAND_BEST): a pool
          record matches at most one S1, so a crowded S1 should not push out a record's only plausible owner.
The pruned set is exactly what every later model scores, i.e. candidate_pairs.tsv for test.

Diagnostics (train): work/prune_pos_train.parquet (every Q true pair the union contains, with its prune score)
and work/prune_grid.csv (Q recall and pairs per S1 for a grid of K / min-p values, from one scoring pass).
`python prune.py report` reprints them without rescoring.
"""
import sys
import time

import numpy as np
import polars as pl

from config import (N_FOLDS, N_JOBS, PRUNE_CAND_BEST_MIN_P, PRUNE_EXTRA_FEATS, PRUNE_HARD_NEG_RANK, PRUNE_K,
                    PRUNE_K_PER_SRC, PRUNE_KEEP_CAND_BEST, PRUNE_MIN_P, PRUNE_MIN_P_COUNTRY, SEED, wpath)
from features import Ctx, features, features_chunked
from gbdt import cv_train
from splits import load_splits

BLOCK_FEATS = ["s_name", "r_name", "s_na", "r_na", "s_ad", "r_ad", "s_rev", "x", "n_cand"]
NEG_RATE = 0.10
TIER = "cheap_plus" if PRUNE_EXTRA_FEATS else "cheap"
GRID_K = (5, 10, 15, 20, 25, 30, 40)
GRID_P = (0.02, 0.01, 0.005, 0.002, 0.001, 0.0005)
EXTRA_COMPACT = 5_000_000   # rows of pending best-S1 additions before they are compacted


def load_union(split: str) -> pl.DataFrame:
    c = pl.read_parquet(wpath(f"cands_{split}.parquet"))
    return c.with_columns(pl.len().over("s1").cast(pl.Float32).alias("n_cand"),
                          pl.col("x").cast(pl.Float32), pl.col("r_name", "r_na", "r_ad").cast(pl.Float32))


def label(df: pl.DataFrame) -> pl.DataFrame:
    gt = pl.read_parquet(wpath("gt_pairs.parquet")).with_columns(pl.lit(1, pl.Int8).alias("y"))
    return df.join(gt, on=["s1", "cand"], how="left").with_columns(pl.col("y").fill_null(0))


def _feat_cols(df):
    return [c for c in df.columns if c not in ("s1", "cand", "y", "hidden", "fold", "w")]


def _pair_key(s1, cand) -> np.ndarray:
    return (np.asarray(s1, np.int64) << 32) | np.asarray(cand, np.int64)


def train_models():
    qids = load_splits().filter(~pl.col("hidden")).select(pl.col("idx").alias("s1"), "fold")
    q = label(load_union("train").join(qids, on="s1"))
    rng = np.random.default_rng(SEED)
    pos = q["y"].to_numpy() == 1
    # hard negatives (near the top of some blocker) are what the prune model must learn to beat: keep them all
    hard = (q.select(pl.min_horizontal("r_name", "r_na", "r_ad") <= PRUNE_HARD_NEG_RANK).to_series().to_numpy()
            if PRUNE_HARD_NEG_RANK > 0 else np.zeros(q.height, bool))
    keep = pos | hard | (rng.random(q.height) < NEG_RATE)
    sample = q.filter(pl.Series(keep)).with_columns(
        pl.Series("w", np.where(pos | hard, 1.0, 1.0 / NEG_RATE)[keep].astype(np.float32)))
    print(f"prune: Q union {q.height:,} rows, training sample {sample.height:,} ({sample['y'].sum():,} pos, "
          f"{int((hard & ~pos).sum()):,} hard neg), feature tier {TIER}")
    del q
    f = features_chunked(Ctx("train"), sample, TIER)
    feats = _feat_cols(f)
    _, _, models, imp = cv_train(f, feats, params=dict(learning_rate=0.1, num_leaves=127), rounds=1000,
                                 weight_col="w", log="prune")
    print(imp.head(12))
    return models, feats


def score_split(split, models, feats, fold_of=None, chunk=6_000_000):
    u = load_union(split)
    ctx = Ctx(split)
    diag = fold_of is not None
    if diag:
        gt = pl.read_parquet(wpath("gt_pairs.parquet"))
        gt_keys = np.sort(_pair_key(gt["s1"], gt["cand"]))
        grid = np.zeros((len(GRID_K), len(GRID_P), 2), np.int64)   # Q: kept pairs, kept true pairs
        pos_parts = []
    best_p = np.full(ctx.norm.height, -1.0, np.float32)            # pool record -> best p over all S1
    best_kept = np.full(ctx.norm.height, -1.0, np.float32)         # ... over the S1s the per-S1 rule kept
    s1 = u["s1"].to_numpy()
    bounds = [0]
    while bounds[-1] < len(s1):
        j = min(bounds[-1] + chunk, len(s1))
        while j < len(s1) and s1[j] == s1[j - 1]:
            j += 1
        bounds.append(j)
    parts, extra, t0 = [], [], time.time()
    for a, b in zip(bounds[:-1], bounds[1:]):
        c = features(ctx, u.slice(a, b - a), TIER)
        X = c.select(feats).to_numpy().astype(np.float32)
        fo = fold_of[c["s1"].to_numpy()] if diag else np.full(c.height, -1)
        p = np.zeros(c.height, np.float32)
        avg = fo == -1                       # H / test rows: average of all fold models
        for k, m in enumerate(models):
            rows = np.flatnonzero(avg | (fo == k))   # Q rows: only their own (out-of-fold) model
            if rows.size:
                pk = m.predict(X[rows], num_threads=N_JOBS)
                p[rows] += np.where(avg[rows], pk / N_FOLDS, pk)
        min_p = ctx.col("country", c["s1"].to_numpy())
        min_p = (min_p.replace_strict(PRUNE_MIN_P_COUNTRY, default=PRUNE_MIN_P, return_dtype=pl.Float32)
                 if PRUNE_MIN_P_COUNTRY else pl.Series(np.full(c.height, PRUNE_MIN_P, np.float32)))
        c = (c.select(["s1", "cand", "src_b"] + BLOCK_FEATS)
              .with_columns(pl.Series("p_prune", p), min_p.alias("_mp"))
              .with_columns(pl.col("p_prune").rank("ordinal", descending=True).over("s1").cast(pl.Float32)
                            .alias("r_prune"),
                            pl.col("p_prune").rank("ordinal", descending=True).over("s1", "src_b").alias("_rs")))
        keep = ((pl.col("p_prune") >= pl.col("_mp"))
                & ((pl.col("r_prune") <= PRUNE_K) | (pl.col("_rs") <= PRUNE_K_PER_SRC)))
        c = c.with_columns(keep.alias("_keep"))
        if diag:
            q = fo >= 0
            y = np.isin(_pair_key(c["s1"], c["cand"]), gt_keys, assume_unique=False)
            r = c["r_prune"].to_numpy()
            for i, k in enumerate(GRID_K):
                for jj, mp in enumerate(GRID_P):
                    m = q & (r <= k) & (p >= mp)
                    grid[i, jj, 0] += m.sum()
                    grid[i, jj, 1] += (m & y).sum()
            pos_parts.append(c.filter(pl.Series(q & y)).select("s1", "cand", "src_b", "p_prune", "r_prune", "_rs",
                                                                "_mp", "n_cand"))
        if PRUNE_KEEP_CAND_BEST:
            ci = c["cand"].to_numpy()
            kp = c["_keep"].to_numpy()
            np.maximum.at(best_p, ci, p)
            np.maximum.at(best_kept, ci[kp], p[kp])
            # rows the per-S1 rule dropped that are (so far) their pool record's best S1
            c = c.with_columns(pl.Series("_best", p >= best_p[ci]))
            extra.append(c.filter(~pl.col("_keep") & pl.col("_best") & (pl.col("p_prune") >= PRUNE_CAND_BEST_MIN_P))
                          .drop("_best"))
            n_extra = sum(e.height for e in extra)
            if n_extra > EXTRA_COMPACT:          # compact: drop rows that are no longer their record's best
                e = pl.concat(extra)
                extra = [e.filter(pl.Series(e["p_prune"].to_numpy() >= best_p[e["cand"].to_numpy()]))]
            c = c.drop("_best")
        parts.append(c.filter(pl.col("_keep")))
        print(f"  prune-score {split}: {b:,}/{len(s1):,} ({time.time() - t0:.0f}s)", flush=True)
    out = pl.concat(parts)
    n_rule = out.height
    if PRUNE_KEEP_CAND_BEST and extra:
        ex = pl.concat(extra)
        ec = ex["cand"].to_numpy()
        is_best = ex["p_prune"].to_numpy() >= best_p[ec]
        not_kept = best_kept[ec] < best_p[ec]      # the record's best S1 is not already in the kept set
        ex = (ex.filter(pl.Series(is_best & not_kept))
                .sort("cand", "s1").unique("cand", keep="first", maintain_order=True))
        out = pl.concat([out, ex])
    out = out.drop("src_b", "_mp", "_rs", "_keep").sort("s1", "cand")
    out.write_parquet(wpath(f"pruned_{split}.parquet"))
    print(f"{split}: pruned to {out.height:,} pairs ({n_rule:,} by the per-S1 rule, "
          f"{out.height - n_rule:,} added as a pool record's best S1)")
    if diag:
        pl.concat(pos_parts).write_parquet(wpath("prune_pos_train.parquet"))
        n_q = int((fold_of >= 0).sum())
        pl.DataFrame([{"K": k, "min_p": mp, "pairs_per_s1": grid[i, jj, 0] / n_q, "kept_true": int(grid[i, jj, 1])}
                      for i, k in enumerate(GRID_K) for jj, mp in enumerate(GRID_P)]
                     ).write_csv(wpath("prune_grid.csv"))


def main():
    t0 = time.time()
    models, feats = train_models()
    sp = load_splits()
    q = sp.filter(~pl.col("hidden"))
    fold_of = np.full(int(sp["idx"].max()) + 1, -1, np.int64)   # S1 idx -> fold (-1 = hidden)
    fold_of[q["idx"].to_numpy()] = q["fold"].to_numpy()
    score_split("train", models, feats, fold_of)
    score_split("test", models, feats)
    report()
    print(f"prune done in {time.time() - t0:.0f}s")


def _share(df: pl.DataFrame, by, lost="lost") -> dict:
    g = df.group_by(by).agg(pl.len().alias("n"), pl.col(lost).sum().alias("lost")).sort(by)
    return {str(r[by]): f"{r['lost']:,} of {r['n']:,} lost ({r['lost'] / r['n']:.2%})" for r in g.to_dicts()}


def report():
    from normalize import load_norm
    gt = pl.read_parquet(wpath("gt_pairs.parquet"))
    q = load_splits().filter(~pl.col("hidden")).select(pl.col("idx").alias("s1"))
    g = gt.join(q, on="s1")
    p = pl.read_parquet(wpath("pruned_train.parquet"), columns=["s1", "cand", "r_prune"]).join(q, on="s1")
    hit = g.join(p, on=["s1", "cand"], how="left")
    print(f"  pruned recall (Q): {hit['r_prune'].is_not_null().mean():.5f}   pairs per Q S1: {p.height / q.height:.2f}")
    for k in (5, 10, 15, 20):
        print(f"    recall with rank <= {k}: {hit.filter(pl.col('r_prune') <= k).height / g.height:.5f}")

    pos_path = wpath("prune_pos_train.parquet")
    if pos_path.exists():
        pos = pl.read_parquet(pos_path).join(q, on="s1")
        print(f"  blocking recall (Q): {pos.height / g.height:.5f}  -> pruning keeps "
              f"{hit['r_prune'].is_not_null().sum() / max(pos.height, 1):.4%} of the true pairs blocking found")
        kept = p.select("s1", "cand", pl.lit(True).alias("kept"))
        pos = pos.join(kept, on=["s1", "cand"], how="left").with_columns(pl.col("kept").fill_null(False))
        pos = pos.with_columns(
            (~pl.col("kept")).alias("lost"),
            pl.when(pl.col("kept")).then(pl.lit("kept"))
            .when((pl.col("p_prune") < pl.col("_mp")) & (pl.col("r_prune") > PRUNE_K)).then(pl.lit("low p and low rank"))
            .when(pl.col("p_prune") < pl.col("_mp")).then(pl.lit("low p only"))
            .otherwise(pl.lit("low rank only")).alias("reason"))
        lost = pos.filter(pl.col("lost"))
        print(f"  true pairs lost by pruning: {lost.height:,}")
        print("    why:", {r["reason"]: r["len"] for r in lost.group_by("reason").len().sort("len", descending=True).to_dicts()})
        norm = load_norm("train", ["idx", "country", "a_empty", "n_flags"])
        pos = (pos.join(norm.select(pl.col("idx").alias("s1"), "country"), on="s1")
                  .join(norm.select(pl.col("idx").alias("cand"), "a_empty", "n_flags"), on="cand")
                  .with_columns(pl.col("src_b").cast(pl.Int8).alias("source"),
                                pl.when((pl.col("n_flags") & 8) > 0).then(pl.lit("indic"))
                                .when((pl.col("n_flags") & 1) > 0).then(pl.lit("domain"))
                                .when((pl.col("n_flags") & 2) > 0).then(pl.lit("alias"))
                                .when(pl.col("a_empty")).then(pl.lit("empty address"))
                                .otherwise(pl.lit("plain")).alias("record type"),
                                pl.col("n_cand").cut([25, 50, 100, 200]).cast(pl.String).alias("union size")))
        for by in ("country", "source", "record type", "union size"):
            print(f"    by {by}:", _share(pos, by))
    grid_path = wpath("prune_grid.csv")
    if grid_path.exists():
        grid = pl.read_csv(grid_path).with_columns((pl.col("kept_true") / g.height).alias("recall"))
        print("  per-S1 rule only (no per-source / best-S1 additions): Q recall | pairs per S1")
        print(grid.pivot(on="min_p", index="K", values="recall").with_columns(pl.exclude("K").round(5)))
        print(grid.pivot(on="min_p", index="K", values="pairs_per_s1").with_columns(pl.exclude("K").round(2)))

    tp = pl.read_parquet(wpath("pruned_test.parquet"), columns=["s1"]) if wpath("pruned_test.parquet").exists() else None
    if tp is not None:
        c = load_norm("test", ["idx", "src", "country"]).filter(pl.col("src") == 1)
        per = (c.select(pl.col("idx").alias("s1"), "country")
                .join(tp.group_by("s1").len(), on="s1", how="left").with_columns(pl.col("len").fill_null(0))
                .group_by("country").agg(pl.col("len").mean()).sort("country"))
        print("  test pairs per S1 by country:", {r["country"]: round(r["len"], 2) for r in per.to_dicts()})


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        report()
    else:
        main()
