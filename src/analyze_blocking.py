"""Where does blocking recall actually leak?

The headline ceiling (0.9499 @K30) was measured on a random 150k Source-1
sample, which inherits train's 60/40 US/India mix. Test is 38/47/15 US /
India / France. If the misses concentrate in one country — or in records whose
names are written in a non-Latin script, which the normalizer folds accents on
but does not transliterate — then the headline number is optimistic for test
and the fix is a different one entirely.

This measures recall split by country, by the script of the record we failed
to retrieve, and by whether the target had a usable address.

    python src/analyze_blocking.py                  # 15k per country
    python src/analyze_blocking.py --per-country 5000

Read-only: no model, no training, nothing written except a summary.
"""
import argparse
import re
import time
from collections import Counter

import pandas as pd

import config as C
import blocking
import data as D
from normalize import add_blocking_columns
from metrics import parse_id_list

# Unicode blocks for the scripts the problem statement and Neal's analysis
# flag as present in the India slice.
INDIC_RANGES = [
    (0x0900, 0x097F, "devanagari"), (0x0980, 0x09FF, "bengali"),
    (0x0A00, 0x0A7F, "gurmukhi"), (0x0A80, 0x0AFF, "gujarati"),
    (0x0B00, 0x0B7F, "oriya"), (0x0B80, 0x0BFF, "tamil"),
    (0x0C00, 0x0C7F, "telugu"), (0x0C80, 0x0CFF, "kannada"),
    (0x0D00, 0x0D7F, "malayalam"),
]
_DOMAIN = re.compile(r"(www\.|https?://|\.com|\.in\b|\.co\b|\.net|\.org)", re.I)


def classify(text: str) -> str:
    """Coarse bucket for why a record might be hard to retrieve."""
    if not text or not text.strip():
        return "empty"
    for ch in text:
        o = ord(ch)
        for lo, hi, name in INDIC_RANGES:
            if lo <= o <= hi:
                return name
    if _DOMAIN.search(text):
        return "domain"
    if any(ord(ch) > 0x2000 for ch in text):
        return "other-nonlatin"
    return "latin"


def stratified_sample(s1, per_country, seed=C.SEED):
    """Equal-sized sample per country, so one slice cannot hide inside another."""
    parts = []
    for country, grp in s1.groupby(s1[C.COUNTRY].astype(str), observed=True):
        n = min(per_country, len(grp))
        parts.append(grp.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def main(per_country):
    t0 = time.time()

    s1_all = D.read_source(C.TRAIN_S1)
    s1 = add_blocking_columns(stratified_sample(s1_all, per_country))
    del s1_all
    print(f"[{time.time()-t0:6.1f}s] sampled {len(s1):,} entities: "
          f"{dict(Counter(s1[C.COUNTRY].astype(str)))}", flush=True)

    s2 = add_blocking_columns(D.read_source(C.TRAIN_S2))
    s3 = add_blocking_columns(D.read_source(C.TRAIN_S3))
    print(f"[{time.time()-t0:6.1f}s] index: S2={len(s2):,} S3={len(s3):,}", flush=True)

    truth = D.read_ground_truth(C.TRAIN_GT, keep_ids=s1[C.ID])
    cands = blocking.generate_candidates(s1, s2, s3)
    del s2, s3
    print(f"[{time.time()-t0:6.1f}s] blocking done", flush=True)

    country_of = dict(zip(s1[C.ID], s1[C.COUNTRY].astype(str)))

    # ---- recall by country, at several K ----
    print("\n=== pair recall by country ===")
    rows = []
    for k in (10, 20, 30, 50):
        if k > C.TOP_K:
            continue
        trunc = {sid: {c for c, _ in v[:k]} for sid, v in cands.items()}
        per = {}
        for sid, true_ids in truth.items():
            if not true_ids:
                continue
            ctry = country_of[sid]
            hit, tot = per.setdefault(ctry, [0, 0])
            hit += len(true_ids & trunc.get(sid, set()))
            tot += len(true_ids)
            per[ctry] = [hit, tot]
        row = {"K": k}
        for ctry, (hit, tot) in sorted(per.items()):
            row[ctry] = round(hit / tot, 4) if tot else None
        allhit = sum(h for h, _ in per.values())
        alltot = sum(t for _, t in per.values())
        row["ALL"] = round(allhit / alltot, 4)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    # ---- what do the MISSED targets look like? ----
    top = {sid: {c for c, _ in v} for sid, v in cands.items()}
    missed, covered = [], []
    for sid, true_ids in truth.items():
        if not true_ids:
            continue
        got = top.get(sid, set())
        missed.extend(true_ids - got)
        covered.extend(true_ids & got)
    print(f"\nmissed true pairs: {len(missed):,} / {len(missed)+len(covered):,}")

    want = set(missed) | set(covered[:len(missed) * 2])   # comparison baseline
    text = pd.concat(
        [D.load_records_by_id(C.TRAIN_S2, want), D.load_records_by_id(C.TRAIN_S3, want)],
        ignore_index=True,
    )
    name_of = dict(zip(text[C.ID], text[C.NAME]))
    addr_of = dict(zip(text[C.ID], text[C.ADDR]))

    def profile(ids, label):
        cls = Counter(classify(name_of.get(i, "")) for i in ids)
        noaddr = sum(1 for i in ids if not str(addr_of.get(i, "")).strip())
        n = max(len(ids), 1)
        print(f"\n--- {label} (n={len(ids):,}) ---")
        for k, v in cls.most_common(8):
            print(f"  {k:16s} {v:8,}  {v/n:6.1%}")
        print(f"  {'(empty address)':16s} {noaddr:8,}  {noaddr/n:6.1%}")

    profile(missed, "MISSED targets")
    profile(covered[:len(missed) * 2], "COVERED targets (baseline)")
    print(f"\n[{time.time()-t0:6.1f}s] done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-country", type=int, default=15000)
    a = ap.parse_args()
    main(a.per_country)
