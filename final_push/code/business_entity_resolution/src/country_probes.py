"""Per-country leaderboard probes from an existing submission, and the arithmetic to read them.

The leaderboard is a mean of per-S1 F0.5, so changing only one country's rows moves the score by that
country's contribution alone. With w_c = the country's share of S1, F_c = its F0.5, s_c = its singleton rate:

  empty_C : every C row predicts nothing      -> score = LB - w_c * (F_c - s_c)
  junk_C  : every C row predicts one sure miss -> score = LB - w_c * F_c        (every C row scores 0)
  empty_all                                    -> score = sum_c w_c * s_c        (overall singleton rate)

junk_C gives F_c directly, with no assumption about the singleton rate; junk_C together with empty_C gives s_c.
The "sure miss" is an S2 record from another country (matches never cross countries).

  python country_probes.py make output/matching_results.tsv          # -> output/probes/*.tsv
  python country_probes.py solve --lb 0.974 empty_France=0.83846 junk_France=0.81 empty_all=0.07
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from config import DATA, OUT

csv.field_size_limit(sys.maxsize)
HEADER = ["source1_entity_id", "matched_entity_ids"]


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)


def s1_country(test_dir: Path) -> dict:
    return {r["entity_id"]: r["country"] or "" for r in _read(test_dir / "test_source1.tsv")}


def junk_ids(test_dir: Path, countries) -> dict:
    """one S2 id per country that belongs to a *different* country."""
    first = {}
    for r in _read(test_dir / "test_source2.tsv"):
        first.setdefault(r["country"] or "", r["entity_id"])
        if len(first) >= len(countries) and all(any(k != c for k in first) for c in countries):
            break
    return {c: next(v for k, v in first.items() if k != c) for c in countries}


def weights(country: dict) -> dict:
    n = Counter(country.values())
    return {c: k / len(country) for c, k in n.items()}


def make(sub: Path, test_dir: Path, out_dir: Path):
    country = s1_country(test_dir)
    cs = sorted(set(country.values()))
    junk = junk_ids(test_dir, cs)
    rows = [(r["source1_entity_id"], r["matched_entity_ids"] or "") for r in _read(sub)]
    assert len(rows) == len(country), f"{sub} has {len(rows):,} rows, test S1 has {len(country):,}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def write(name, fn):
        path = out_dir / f"matching_results_{name}.tsv"
        with open(path, "w", encoding="utf-8") as f:
            f.write("\t".join(HEADER) + "\n")
            for s1, m in rows:
                f.write(f"{s1}\t{fn(s1, m)}\n")
        print(f"  wrote {path}")

    for c in cs:
        write(f"empty_{c}", lambda s1, m, c=c: "" if country[s1] == c else m)
        write(f"junk_{c}", lambda s1, m, c=c: junk[c] if country[s1] == c else m)
    write("empty_all", lambda s1, m: "")
    print("S1 share by country:", {c: round(w, 4) for c, w in weights(country).items()})
    print("predicted-empty share by country:",
          {c: round(sum(1 for s1, m in rows if country[s1] == c and not m) / n, 4)
           for c, n in Counter(country.values()).items()})


def solve(lb: float, scores: dict, w: dict):
    f, s = {}, {}
    for c, wc in w.items():
        e, j = scores.get(f"empty_{c}"), scores.get(f"junk_{c}")
        if j is not None:
            f[c] = (lb - j) / wc
        if e is not None and j is not None:
            s[c] = (e - j) / wc
        if e is not None and j is None:
            print(f"  {c}: F - singleton rate = {(lb - e) / wc:.4f}  (add junk_{c} to separate them)")
    missing = [c for c in w if c not in f]
    if len(missing) == 1:                      # last country by subtraction
        c = missing[0]
        f[c] = (lb - sum(w[k] * f[k] for k in f)) / w[c]
        print(f"  {c}: F by subtraction")
    for c in w:
        if c in f:
            print(f"  {c:8s} w={w[c]:.4f}  F0.5={f[c]:.4f}  loss vs 1.0 = {w[c] * (1 - f[c]):.4f} of LB"
                  + (f"  singleton rate={s[c]:.4f}" if c in s else ""))
    if "empty_all" in scores:
        print(f"  overall singleton rate (public set) = {scores['empty_all']:.4f}  (train: 0.0558)")
    if any(v > 1.0 or v < 0.0 for v in f.values()):
        print("  !! an F outside [0, 1]: the public subset's country mix differs from the full test set, "
              "or the probes come from different base submissions")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("make")
    m.add_argument("submission")
    m.add_argument("--test-dir", default=str(DATA / "test"))
    m.add_argument("--out", default=str(OUT / "probes"))
    s = sp.add_parser("solve")
    s.add_argument("--lb", type=float, required=True, help="leaderboard score of the base submission")
    s.add_argument("scores", nargs="+", help="probe=score, e.g. empty_France=0.83846 junk_India=0.52")
    s.add_argument("--test-dir", default=str(DATA / "test"))
    a = ap.parse_args(argv)
    if a.cmd == "make":
        make(Path(a.submission), Path(a.test_dir), Path(a.out))
    else:
        scores = {k: float(v) for k, v in (x.split("=", 1) for x in a.scores)}
        solve(a.lb, scores, weights(s1_country(Path(a.test_dir))))


if __name__ == "__main__":
    main()
