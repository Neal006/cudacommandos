"""Design-driving EDA. Answers the questions in QUESTIONS.md with measured numbers."""
import polars as pl, re, sys, time
sys.stdout.reconfigure(encoding='utf-8')
import os
D = os.environ.get("AMLC_DATASET_DIR", "data")  # folder containing train/ and test/
t0 = time.time()
def rd(p):
    return pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, encoding="utf8-lossy")
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

s1 = rd(f"{D}/train/train_source1.tsv"); s2 = rd(f"{D}/train/train_source2.tsv"); s3 = rd(f"{D}/train/train_source3.tsv")
gt = rd(f"{D}/train/train_ground_truth.tsv")
log("rows", len(s1), len(s2), len(s3), len(gt))
oth = pl.concat([s2, s3])
for n, df in [("s1", s1), ("s2", s2), ("s3", s3)]:
    log(n, "dup ids", len(df) - df["entity_id"].n_unique(), "null country", df["country"].null_count(),
        "countries", df["country"].value_counts().sort("count", descending=True).rows())

pairs = (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
           .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
           .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "o_id"}))
log("positive pairs", len(pairs))

# Q1 exclusivity: does any S2/S3 record match more than one S1?
vc = pairs["o_id"].value_counts()
log("Q1 o_id linked to >1 S1:", vc.filter(pl.col("count") > 1).height, "max", vc["count"].max())
# Q2 distractors
linked = set(pairs["o_id"].to_list())
for n, df in [("s2", s2), ("s3", s3)]:
    k = df["entity_id"].is_in(pl.Series(list(linked)).implode()).sum()
    log(f"Q2 {n} linked {k}/{len(df)} = {k/len(df):.3f}  -> unlinked distractors {1-k/len(df):.3f}")
# Q3 per-source counts per entity
pc = pairs.with_columns(pl.col("o_id").str.slice(0, 2).alias("src")).group_by(["s1_id", "src"]).len()
for s in ["S2", "S3"]:
    log("Q3", s, "links/entity dist", pc.filter(pl.col("src") == s)["len"].value_counts().sort("len").rows())
# Q4 country agreement on positives
P = (pairs.join(s1.select(pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("n1"),
                          pl.col("business_address").alias("a1"), pl.col("country").alias("c1")), on="s1_id")
          .join(oth.select(pl.col("entity_id").alias("o_id"), pl.col("business_name").alias("n2"),
                           pl.col("business_address").alias("a2"), pl.col("country").alias("c2")), on="o_id"))
log("joined positives", len(P), "(missing o_id in sources:", len(pairs) - len(P), ")")
log("Q4 cross-country positives", (P["c1"] != P["c2"]).sum(), P.filter(pl.col("c1") != pl.col("c2")).select("c1", "c2").head(5).rows())
# Q5 ID leakage: numeric id correlation
num = lambda c: pl.col(c).str.slice(3).cast(pl.Int64)
log("Q5 spearman(id1,id2) on positives", P.select(pl.corr(num("s1_id"), num("o_id"), method="spearman")).item())
log("Q5 S2/S3 id value range overlap: s2 ids also in s3?", s2["entity_id"].str.slice(3).is_in(pl.Series(s3["entity_id"].str.slice(3)).implode()).sum())
# Q7 surface agreement on positives
low = lambda c: pl.col(c).fill_null("").str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", "").str.replace_all(r"\s+", " ").str.strip_chars()
Q = P.with_columns(low("n1").alias("l1"), low("n2").alias("l2"), low("a1").alias("m1"), low("a2").alias("m2"))
log("Q7 positives exact norm-name eq", (Q["l1"] == Q["l2"]).mean(), "exact norm-addr eq", (Q["m1"] == Q["m2"]).mean(),
    "o addr empty", (Q["m2"] == "").mean())
nonascii = lambda c: pl.col(c).fill_null("").str.contains(r"[^\x00-\x7F]")
dev = lambda c: pl.col(c).fill_null("").str.contains(r"[\u0900-\u097F]")
for n, df in [("s1", s1), ("s2", s2), ("s3", s3)]:
    log("Q15", n, df.group_by("country").agg(nonascii("business_name").mean().alias("name_nonascii"),
        dev("business_name").mean().alias("name_devanagari"), dev("business_address").mean().alias("addr_devanagari"),
        pl.col("business_name").str.contains(r"\.(com|in|net|org|co)\b").mean().alias("name_domain"),
        pl.col("business_name").str.contains(r"^[^\p{L}\p{N}]").mean().alias("name_junk_prefix"),
        pl.col("business_address").fill_null("").str.contains(r"\b\d{6}\b").mean().alias("addr_6digit"),
        pl.col("business_address").fill_null("").str.contains(r"\b\d{5}\b").mean().alias("addr_5digit"),
        (pl.col("business_address").fill_null("") == "").mean().alias("addr_empty"),
        pl.col("business_name").str.len_chars().mean().alias("name_len")).sort("country").rows())
# Q11 postal agreement on positives (when both sides have one)
pin = lambda c, k: pl.col(c).fill_null("").str.extract(rf"\b(\d{{{k}}})\b", 1)
for cty, k in [("India", 6), ("US", 5)]:
    x = P.filter(pl.col("c1") == cty).with_columns(pin("a1", k).alias("p1"), pin("a2", k).alias("p2"))
    both = x.filter(pl.col("p1").is_not_null() & pl.col("p2").is_not_null())
    log(f"Q11 {cty} positives: s1 has code {x['p1'].is_not_null().mean():.3f}, other has {x['p2'].is_not_null().mean():.3f}, both {len(both)/max(len(x),1):.3f}, agree|both {(both['p1']==both['p2']).mean() if len(both) else None}")
# Q8 non-unique S1 names (chains / hard negatives)
s1n = s1.with_columns(low("business_name").alias("l"))
dn = s1n["l"].value_counts().filter(pl.col("count") > 1)
log("Q8 S1 normalized names shared by >1 S1 entity:", dn["count"].sum(), "of", len(s1), "top", dn.sort("count", descending=True).head(8).rows())
# Q10 singleton profile
sing = gt.filter(pl.col("matched_entity_ids").fill_null("") == "")["source1_entity_id"]
ss = s1.filter(pl.col("entity_id").is_in(pl.Series(sing).implode()))
log("Q10 singleton country share", ss["country"].value_counts().rows(), "overall", s1["country"].value_counts().rows())
# Q6 train/test overlap
t1 = rd(f"{D}/test/test_source1.tsv")
log("Q6 test S1 ids also in train S1:", t1["entity_id"].is_in(pl.Series(s1["entity_id"]).implode()).sum())
t1l = t1.with_columns(low("business_name").alias("l"))
log("Q6 test S1 norm names seen in train S1:", t1l["l"].is_in(pl.Series(s1n["l"]).implode()).mean())
log("Q15 test s1", t1.group_by("country").agg(nonascii("business_name").mean().alias("nonascii"), dev("business_name").mean().alias("dev"),
    pl.col("business_address").fill_null("").str.contains(r"\b\d{5}\b").mean().alias("5digit"),
    pl.col("business_name").str.contains(r"^[^\p{L}\p{N}]").mean().alias("junk")).rows())
# samples
pl.Config.set_fmt_str_lengths(80); pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
for cty in ["US", "India"]:
    ids = P.filter(pl.col("c1") == cty)["s1_id"].unique(maintain_order=True).head(4)
    print(P.filter(pl.col("s1_id").is_in(ids)).select("s1_id", "n1", "a1", "o_id", "n2", "a2").sort("s1_id"))
print(ss.head(6))
print(t1.filter(pl.col("country") == "France").head(8))
t2 = rd(f"{D}/test/test_source2.tsv")
print(t2.filter(pl.col("country") == "France").head(8))
log("test s2 countries", t2["country"].value_counts().rows(), "Q6 test S2 ids in train S2:", t2["entity_id"].is_in(pl.Series(s2["entity_id"]).implode()).sum())
log("done")
