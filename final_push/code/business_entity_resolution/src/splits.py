"""Train S1 split assignment: hidden set H (20%) and GBDT folds over the query set Q."""
import numpy as np
import polars as pl

from config import HIDDEN_FRAC, N_FOLDS, SEED, wpath
from io_utils import load_records


def main():
    s1 = load_records("train", ["idx", "src"]).filter(pl.col("src") == 1)
    rng = np.random.default_rng(SEED)
    n = s1.height
    out = s1.select("idx").with_columns(
        pl.Series("hidden", rng.random(n) < HIDDEN_FRAC),
        pl.Series("fold", rng.integers(0, N_FOLDS, n).astype(np.int8)),
    )
    out.write_parquet(wpath("splits.parquet"))
    print(out.group_by("hidden").len().to_dicts())


def load_splits() -> pl.DataFrame:
    return pl.read_parquet(wpath("splits.parquet"))


if __name__ == "__main__":
    main()
