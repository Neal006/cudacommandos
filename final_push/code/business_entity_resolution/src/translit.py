"""Indic -> Latin transliteration.

Primary: a token dictionary learned from train GT pairs (Indic names are word-by-word
renderings of the S1 English names, token counts align ~100%).
Address components in Indic script (mostly state names) map to the S1 state they co-occur with.
Fallback for unseen tokens: indic_transliteration (MIT) -> ITRANS, snapped to a known Latin word by
consonant skeleton when possible.
"""
import pickle
import re
from collections import Counter, defaultdict

import polars as pl

from config import wpath

INDIC_RE = re.compile(r"[ऀ-ൿ]")
ASCII_PUNCT = re.compile(r"[!-/:-@\[-`{-~]")
_BLOCKS = [(0x0900, "DEVANAGARI"), (0x0980, "BENGALI"), (0x0A00, "GURMUKHI"), (0x0A80, "GUJARATI"),
           (0x0B00, "ORIYA"), (0x0B80, "TAMIL"), (0x0C00, "TELUGU"), (0x0C80, "KANNADA"),
           (0x0D00, "MALAYALAM")]


def has_indic(s: str) -> bool:
    return bool(INDIC_RE.search(s))


def script_of(tok: str):
    for ch in tok:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x0D7F:
            return _BLOCKS[(cp - 0x0900) // 0x80][1]
    return None


def skeleton(w: str) -> str:
    w = re.sub(r"[aeiouhy]", "", w.lower())
    return re.sub(r"(.)\1+", r"\1", w)


def _latin_canon(tok: str) -> str:
    # canonicalise legal-suffix words so pvt/private etc. do not split the vote
    from normalize import SUFFIX_CANON
    t = tok.lower()
    return SUFFIX_CANON.get(t, t)


def build():
    from io_utils import load_records
    rec = load_records("train", ["idx", "name", "addr"])
    gt = pl.read_parquet(wpath("gt_pairs.parquet"))
    pairs = (gt.join(rec.rename({"idx": "s1", "name": "n1", "addr": "a1"}), on="s1")
               .join(rec.rename({"idx": "cand", "name": "n2", "addr": "a2"}), on="cand"))
    nm = pairs.filter(pl.col("n2").str.contains(r"[ऀ-ൿ]"))
    counts = defaultdict(Counter)
    for a, b in zip(nm["n1"].to_list(), nm["n2"].to_list()):
        ta = ASCII_PUNCT.sub(" ", a).split()
        tb = ASCII_PUNCT.sub(" ", b).split()
        if len(ta) != len(tb):
            continue
        for x, y in zip(ta, tb):
            if has_indic(y):
                counts[y][_latin_canon(x)] += 1
    name_dict = {k: v.most_common(1)[0][0] for k, v in counts.items()}

    # address components in Indic script -> S1 last component (the state)
    ad = pairs.filter(pl.col("a2").str.contains(r"[ऀ-ൿ]"))
    acounts = defaultdict(Counter)
    for a, b in zip(ad["a1"].to_list(), ad["a2"].to_list()):
        last = a.split(",")[-1].strip().lower()
        for comp in b.split(","):
            comp = comp.strip()
            if has_indic(comp):
                acounts[comp][last] += 1
    addr_dict = {}
    for k, v in acounts.items():
        (w, c), tot = v.most_common(1)[0], sum(v.values())
        if c / tot >= 0.5:
            addr_dict[k] = w

    # skeleton -> most frequent Latin word, for snapping fallback output
    vocab = Counter(name_dict.values())
    skel = {}
    for w, _ in vocab.most_common():
        skel.setdefault(skeleton(w), w)
    with open(wpath("translit.pkl"), "wb") as f:
        pickle.dump({"name": name_dict, "addr": addr_dict, "skel": skel}, f)
    print("name dict", len(name_dict), "addr dict", len(addr_dict), "skeletons", len(skel))


_T = None


def _load():
    global _T
    if _T is None:
        with open(wpath("translit.pkl"), "rb") as f:
            _T = pickle.load(f)
    return _T


def _fallback(tok: str):
    from indic_transliteration import sanscript
    sc = script_of(tok)
    try:
        out = sanscript.transliterate(tok, getattr(sanscript, sc), sanscript.ITRANS)
    except Exception:
        return tok
    out = re.sub(r"[^a-z0-9]", "", out.lower())
    return _load()["skel"].get(skeleton(out), out)


def translit_name(s: str):
    """Returns (latin_text, n_indic_tokens, n_oov_tokens)."""
    if not has_indic(s):
        return s, 0, 0
    d = _load()["name"]
    out, n_ind, n_oov = [], 0, 0
    for tok in s.split():
        if not has_indic(tok):
            out.append(tok)
            continue
        n_ind += 1
        core = ASCII_PUNCT.sub("", tok)
        if core in d:
            out.append(d[core])
        else:
            n_oov += 1
            out.append(_fallback(core))
    return " ".join(out), n_ind, n_oov


def translit_addr(s: str) -> str:
    if not has_indic(s):
        return s
    d = _load()["addr"]
    comps = []
    for comp in s.split(","):
        c = comp.strip()
        if has_indic(c):
            c = d.get(c) or translit_name(c)[0]
        comps.append(c)
    return ", ".join(comps)


if __name__ == "__main__":
    build()
