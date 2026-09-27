"""Stage 0: raw TSV -> parquet caches with integer record ids.

records_{split}.parquet : idx (int32), entity_id, src (1/2/3), name, addr, country
gt_pairs.parquet        : s1 (int32 idx), cand (int32 idx)   [train only]
"""
import polars as pl

from config import DATA, SPLITS, wpath


def read_tsv(path) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


def build_records(split: str) -> pl.DataFrame:
    frames = []
    for src in (1, 2, 3):
        df = read_tsv(DATA / split / f"{split}_source{src}.tsv")
        frames.append(df.select(
            pl.col("entity_id"),
            pl.lit(src, pl.Int8).alias("src"),
            pl.col("business_name").fill_null("").alias("name"),
            pl.col("business_address").fill_null("").alias("addr"),
            pl.col("country").fill_null("").alias("country"),
        ))
    rec = pl.concat(frames).with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int32))
    rec.write_parquet(wpath(f"records_{split}.parquet"))
    return rec


def build_gt(rec: pl.DataFrame) -> pl.DataFrame:
    gt = read_tsv(DATA / "train" / "train_ground_truth.tsv")
    ids = rec.select("entity_id", "idx")
    pairs = (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
               .explode("matched_entity_ids", empty_as_null=True)
               .filter(pl.col("matched_entity_ids") != "")
               .join(ids.rename({"entity_id": "source1_entity_id", "idx": "s1"}), on="source1_entity_id")
               .join(ids.rename({"entity_id": "matched_entity_ids", "idx": "cand"}), on="matched_entity_ids")
               .select("s1", "cand"))
    pairs.write_parquet(wpath("gt_pairs.parquet"))
    return pairs


def load_records(split: str, columns=None) -> pl.DataFrame:
    return pl.read_parquet(wpath(f"records_{split}.parquet"), columns=columns)


def main():
    for split in SPLITS:
        rec = build_records(split)
        print(split, rec.height, rec.group_by("src").len().sort("src").to_dicts())
        if split == "train":
            print("gt pairs", build_gt(rec).height)


if __name__ == "__main__":
    main()
