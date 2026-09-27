"""Exact macro F0.5 (per S1, singletons included) + breakdowns.

F0.5 for one S1 = 1.25*tp / (0.25*|T| + |P|); 1.0 when both P and T are empty.
"""
import polars as pl

from config import wpath


def per_entity(pred: pl.DataFrame, gt: pl.DataFrame, s1_ids: pl.DataFrame) -> pl.DataFrame:
    """pred, gt: (s1, cand); s1_ids: (s1,) the evaluation population. Returns s1, np, nt, tp, f."""
    npred = pred.group_by("s1").len("np")
    ntrue = gt.group_by("s1").len("nt")
    tp = pred.join(gt, on=["s1", "cand"]).group_by("s1").len("tp")
    e = (s1_ids.select("s1").join(npred, on="s1", how="left").join(ntrue, on="s1", how="left")
         .join(tp, on="s1", how="left").with_columns(pl.col("np", "nt", "tp").fill_null(0)))
    return e.with_columns(
        pl.when((pl.col("np") == 0) & (pl.col("nt") == 0)).then(1.0)
        .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("nt") + pl.col("np"))).fill_nan(0.0).alias("f"))


def macro_f05(pred, gt, s1_ids) -> float:
    return per_entity(pred, gt, s1_ids)["f"].mean()


def report(pred: pl.DataFrame, gt: pl.DataFrame, s1_ids: pl.DataFrame, title=""):
    from normalize import load_norm
    e = per_entity(pred, gt, s1_ids)
    c = load_norm("train", ["idx", "country"]).rename({"idx": "s1"})
    e = e.join(c, on="s1", how="left")
    tot_tp, tot_p, tot_t = e["tp"].sum(), e["np"].sum(), e["nt"].sum()
    print(f"== {title} macro F0.5 = {e['f'].mean():.5f}   (micro P {tot_tp / max(tot_p, 1):.4f} "
          f"R {tot_tp / max(tot_t, 1):.4f}, {e.height:,} S1)")
    print("   by country:", {r["country"]: round(r["f"], 5) for r in e.group_by("country").agg(pl.col("f").mean()).to_dicts()})
    s = e.with_columns((pl.col("nt") == 0).alias("singleton")).group_by("singleton").agg(
        pl.col("f").mean(), pl.len())
    print("   singleton / matched:", {("singleton" if r["singleton"] else "matched"): (round(r["f"], 5), r["len"])
                                      for r in s.to_dicts()})
    return e


def load_gt() -> pl.DataFrame:
    return pl.read_parquet(wpath("gt_pairs.parquet"))
