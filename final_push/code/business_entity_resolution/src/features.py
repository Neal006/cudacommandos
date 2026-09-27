"""Stage D: pairwise features for (S1, candidate) pairs.

Tiers:
  cheap / cheap_plus -> prune model on the full blocking union
  full               -> stage-1 model on the pruned candidate set
Set features use a numba merge over sorted token-id arrays (CSR); string similarities use rapidfuzz cpdist.
"""
import math
import time

import numpy as np
import polars as pl
from numba import njit, prange
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from config import N_JOBS, wpath
from normalize import load_norm

SET_FIELDS = {"nm": ("n_core", " "), "st": ("a_street", " "), "at": ("a_tok", " "), "nu": ("a_nums", " "),
              "loc": ("a_loc", "|")}
STR_COLS = ["n_core", "n_full", "n_concat", "n_alt", "n_key", "n_suffix", "a_street", "a_tok", "a_hn", "a_hnd",
            "a_hns", "a_unit", "a_city", "a_state", "a_pc"]
NUM_COLS = ["src", "n_flags", "n_oov", "n_ntok", "a_empty", "a_ncomp", "a_land"]


# ----------------------------------------------------------------------------------------------
# token sets
# ----------------------------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _sort_unique(off, tok):
    n = len(off) - 1
    newlen = np.zeros(n, np.int64)
    for i in prange(n):
        a, b = off[i], off[i + 1]
        if b - a > 1:
            seg = np.sort(tok[a:b])
            k = 1
            for j in range(1, b - a):
                if seg[j] != seg[k - 1]:
                    seg[k] = seg[j]
                    k += 1
            tok[a:a + k] = seg[:k]
            newlen[i] = k
        else:
            newlen[i] = b - a
    noff = np.zeros(n + 1, np.int64)
    for i in range(n):
        noff[i + 1] = noff[i] + newlen[i]
    ntok = np.empty(noff[n], np.int32)
    for i in prange(n):
        ntok[noff[i]:noff[i + 1]] = tok[off[i]:off[i] + newlen[i]]
    return noff, ntok


@njit(parallel=True, cache=True)
def _pair_sets(off, tok, idf, ia, ib, out):
    for i in prange(len(ia)):
        p, a1 = off[ia[i]], off[ia[i] + 1]
        q, b1 = off[ib[i]], off[ib[i] + 1]
        na, nb = a1 - p, b1 - q
        inter = 0
        wi = wa = wb = mua = mub = 0.0
        while p < a1 and q < b1:
            ta, tb = tok[p], tok[q]
            if ta == tb:
                w = idf[ta]
                inter += 1
                wi += w
                wa += w
                wb += w
                p += 1
                q += 1
            elif ta < tb:
                w = idf[ta]
                wa += w
                mua = max(mua, w)
                p += 1
            else:
                w = idf[tb]
                wb += w
                mub = max(mub, w)
                q += 1
        while p < a1:
            w = idf[tok[p]]
            wa += w
            mua = max(mua, w)
            p += 1
        while q < b1:
            w = idf[tok[q]]
            wb += w
            mub = max(mub, w)
            q += 1
        out[i, 0] = inter
        out[i, 1] = na
        out[i, 2] = nb
        out[i, 3] = wi
        out[i, 4] = wa
        out[i, 5] = wb
        out[i, 6] = mua
        out[i, 7] = mub


def build_sets(split: str = None):
    """Token CSR arrays per split with ONE vocabulary / IDF computed over train + test together, so a word
    has the same weight in both splits (per-split IDF made train and test features drift apart)."""
    norms = {sp: load_norm(sp, [c for c, _ in SET_FIELDS.values()]) for sp in ("train", "test")}
    n_all = sum(d.height for d in norms.values())
    for key, (col, sep) in SET_FIELDS.items():
        flats, lens = {}, {}
        for sp, norm in norms.items():
            lst = norm[col].str.split(sep).list.eval(pl.element().filter(pl.element() != ""))
            lens[sp] = lst.list.len().to_numpy().astype(np.int64)
            flats[sp] = lst.explode(empty_as_null=True).drop_nulls().alias("t")
        vocab = pl.concat([f.unique() for f in flats.values()]).unique().to_frame().with_row_index("id")
        df_count = np.zeros(vocab.height, np.int64)
        arrays = {}
        for sp, flat in flats.items():
            codes = (flat.to_frame().join(vocab, on="t", how="left", maintain_order="left")["id"]
                     .to_numpy().astype(np.int32))
            off = np.zeros(norms[sp].height + 1, np.int64)
            np.cumsum(lens[sp], out=off[1:])
            off, tok = _sort_unique(off, codes.copy())
            df_count += np.bincount(tok, minlength=vocab.height)
            arrays[sp] = (off, tok)
        idf = np.log((n_all + 1) / (df_count + 1)).astype(np.float32)
        for sp, (off, tok) in arrays.items():
            np.savez(wpath(f"sets_{sp}_{key}.npz"), off=off, tok=tok, idf=idf)
        print(f"  sets {key}: joint vocab {vocab.height:,}")


# ----------------------------------------------------------------------------------------------
# context
# ----------------------------------------------------------------------------------------------
class Ctx:
    def __init__(self, split: str):
        self.split = split
        t0 = time.time()
        self.norm = load_norm(split, ["idx", "country"] + STR_COLS + NUM_COLS)
        assert (self.norm["idx"].to_numpy() == np.arange(self.norm.height)).all()
        self.num = {c: self.norm[c].cast(pl.Float32).to_numpy() for c in NUM_COLS}
        self.sets = {}
        for key in SET_FIELDS:
            z = np.load(wpath(f"sets_{split}_{key}.npz"))
            self.sets[key] = (z["off"], z["tok"], z["idf"])
        self._freq()
        print(f"  ctx {split} loaded in {time.time() - t0:.0f}s")

    def _freq(self):
        """name / address ambiguity: how many S1 (or pool) records share this name key / exact address."""
        cols = ["idx", "src", "country", "n_key", "a_city", "a_street", "a_hnd"]
        d = self.norm.select(cols)
        other = load_norm("test" if self.split == "train" else "train", cols)
        both = pl.concat([d, other])            # counts over train + test: same scale in both splits
        s1 = both.filter(pl.col("src") == 1)
        pool = both.filter(pl.col("src") != 1)
        addr = ["country", "a_city", "a_street", "a_hnd"]
        k1 = s1.group_by("country", "n_key").len().rename({"len": "f1"})
        k2 = s1.group_by("country", "a_city", "n_key").len().rename({"len": "f2"})
        k3 = pool.group_by("country", "a_city", "n_key").len().rename({"len": "f3"})
        k4 = s1.filter(pl.col("a_street") != "").group_by(addr).len().rename({"len": "f4"})
        k5 = s1.filter(pl.col("a_hnd") != "").group_by("country", "a_city", "a_hnd").len().rename({"len": "f5"})
        f = (d.join(k1, on=["country", "n_key"], how="left").join(k2, on=["country", "a_city", "n_key"], how="left")
              .join(k3, on=["country", "a_city", "n_key"], how="left").join(k4, on=addr, how="left")
              .join(k5, on=["country", "a_city", "a_hnd"], how="left").sort("idx"))
        self.freq = {c: np.log1p(f[c].fill_null(0).to_numpy().astype(np.float32))
                     for c in ("f1", "f2", "f3", "f4", "f5")}

    def col(self, name, idx):
        return self.norm[name].gather(idx)


def _setfeats(ctx, key, ia, ib, prefix, full=True):
    off, tok, idf = ctx.sets[key]
    out = np.zeros((len(ia), 8), np.float32)
    _pair_sets(off, tok, idf, ia, ib, out)
    inter, na, nb, wi, wa, wb, mua, mub = out.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uni = na + nb - inter
        f = {f"{prefix}_jacc": np.where(uni > 0, inter / uni, np.nan),
             f"{prefix}_inter": inter}
        if full:
            wu = wa + wb - wi
            f.update({f"{prefix}_contA": np.where(na > 0, inter / na, np.nan),
                      f"{prefix}_contB": np.where(nb > 0, inter / nb, np.nan),
                      f"{prefix}_idf_jacc": np.where(wu > 0, wi / wu, np.nan),
                      f"{prefix}_unm_a": mua, f"{prefix}_unm_b": mub, f"{prefix}_unm_sum_b": wb - wi,
                      f"{prefix}_na": na, f"{prefix}_nb": nb})
    return {k: v.astype(np.float32) for k, v in f.items()}


def _cp(a, b, scorer, **kw):
    return cpdist(a, b, scorer=scorer, workers=N_JOBS, dtype=np.float32, **kw)


def _eq(a: pl.Series, b: pl.Series, missing=-1.0):
    """1 equal, 0 different, `missing` if either side empty."""
    both = (a != "") & (b != "")
    return np.where(both.to_numpy(), (a == b).to_numpy().astype(np.float32), missing).astype(np.float32)


def _best_variant(ctx, ia, ib, base):
    """max token_set_ratio over name variants (main core + alias / raw / domain alternatives)."""
    alt_a, alt_b = ctx.col("n_alt", ia), ctx.col("n_alt", ib)
    sel = np.flatnonzero(((alt_a != "") | (alt_b != "")).to_numpy())
    best = base.copy()
    if sel.size == 0:
        return best
    va = (ctx.col("n_core", ia[sel]) + "|" + alt_a.gather(sel)).str.strip_chars("|").str.split("|")
    vb = (ctx.col("n_core", ib[sel]) + "|" + alt_b.gather(sel)).str.strip_chars("|").str.split("|")
    fr = pl.DataFrame({"i": sel, "a": va, "b": vb}).explode("a").explode("b")
    sc = _cp(fr["a"].to_list(), fr["b"].to_list(), fuzz.token_set_ratio)
    np.maximum.at(best, fr["i"].to_numpy(), sc)
    return best


def _prune_extra(ctx, ia, ib, ha, hb):
    """Vectorised subset of the full-tier signals for the prune model: flags that explain a low name score
    (domain / alias / Indic / OOV tokens), name ambiguity, house-number distance and postcode."""
    h = pl.DataFrame({"a": ha, "b": hb}).select(
        pl.when((pl.col("a") != "") & (pl.col("b") != ""))
        .then(((pl.col("a") != pl.col("b"))
               & (pl.col("a").str.starts_with(pl.col("b")) | pl.col("b").str.starts_with(pl.col("a"))))
              .cast(pl.Float32))
        .otherwise(-1.0).alias("pre"),
        pl.col("a").str.slice(0, 9).cast(pl.Int64, strict=False).alias("ai"),
        pl.col("b").str.slice(0, 9).cast(pl.Int64, strict=False).alias("bi"))
    ai, bi = h["ai"].to_numpy().astype(np.float64), h["bi"].to_numpy().astype(np.float64)
    return {"flags_a": ctx.num["n_flags"][ia], "flags_b": ctx.num["n_flags"][ib], "oov_b": ctx.num["n_oov"][ib],
            "nf_key_s1": ctx.freq["f1"][ia], "nf_key_city_s1": ctx.freq["f2"][ia],
            "nf_key_s1_cand": ctx.freq["f1"][ib], "af_s1_hn_cand": ctx.freq["f5"][ib],
            "hn_prefix": h["pre"].to_numpy(), "hn_absdiff": np.log1p(np.abs(ai - bi)).astype(np.float32),
            "pc_eq": _eq(ctx.col("a_pc", ia), ctx.col("a_pc", ib))}


def features(ctx: Ctx, pairs: pl.DataFrame, tier: str) -> pl.DataFrame:
    """pairs must have s1, cand plus blocking columns; returns pairs + feature columns.
    tier: 'cheap' | 'cheap_plus' (cheap + a few vectorised full-tier signals, for the prune model) | 'full'."""
    full = tier == "full"
    ia = pairs["s1"].to_numpy().astype(np.int64)
    ib = pairs["cand"].to_numpy().astype(np.int64)
    f = {}
    # names
    na, nb = ctx.col("n_core", ia).to_list(), ctx.col("n_core", ib).to_list()
    f["nm_tset"] = _cp(na, nb, fuzz.token_set_ratio)
    f.update(_setfeats(ctx, "nm", ia, ib, "nm", full))
    f["nm_key_eq"] = (ctx.col("n_key", ia) == ctx.col("n_key", ib)).to_numpy().astype(np.float32)
    # address
    f.update(_setfeats(ctx, "st", ia, ib, "st", full))
    f.update(_setfeats(ctx, "at", ia, ib, "at", full))
    f.update(_setfeats(ctx, "nu", ia, ib, "nu", False))
    f.update(_setfeats(ctx, "loc", ia, ib, "loc", False))
    ha, hb = ctx.col("a_hnd", ia), ctx.col("a_hnd", ib)
    f["hnd_eq"] = _eq(ha, hb)
    f["hn_a_has"] = (ha != "").to_numpy().astype(np.float32)
    f["hn_b_has"] = (hb != "").to_numpy().astype(np.float32)
    f["state_eq"] = _eq(ctx.col("a_state", ia), ctx.col("a_state", ib))
    f["city_eq"] = _eq(ctx.col("a_city", ia), ctx.col("a_city", ib))
    f["ad_empty_b"] = ctx.num["a_empty"][ib]
    f["src_b"] = ctx.num["src"][ib]
    if tier == "cheap_plus":
        f.update(_prune_extra(ctx, ia, ib, ha, hb))
    if full:
        fa, fb = ctx.col("n_full", ia).to_list(), ctx.col("n_full", ib).to_list()
        f["nm_tsort"] = _cp(fa, fb, fuzz.token_sort_ratio)
        f["nm_full_tset"] = _cp(fa, fb, fuzz.token_set_ratio)
        f["nm_partial"] = _cp(na, nb, fuzz.partial_ratio)
        f["nm_ratio"] = _cp(na, nb, fuzz.ratio)
        f["nm_jw"] = _cp(na, nb, JaroWinkler.normalized_similarity)
        ca, cb = ctx.col("n_concat", ia).to_list(), ctx.col("n_concat", ib).to_list()
        f["nm_concat_lev"] = _cp(ca, cb, Levenshtein.normalized_similarity)
        f["nm_concat_partial"] = _cp(ca, cb, fuzz.partial_ratio)
        ska = ctx.col("n_concat", ia).str.replace_all(r"[aeiouhy]", "").to_list()
        skb = ctx.col("n_concat", ib).str.replace_all(r"[aeiouhy]", "").to_list()
        f["nm_skel"] = _cp(ska, skb, fuzz.ratio)
        f["nm_best_tset"] = _best_variant(ctx, ia, ib, f["nm_tset"])
        sa, sb = ctx.col("n_suffix", ia), ctx.col("n_suffix", ib)
        f["nm_suffix_rel"] = np.select(
            [((sa == "") & (sb == "")).to_numpy(), (sa == sb).to_numpy(), ((sa == "") | (sb == "")).to_numpy()],
            [0, 1, 2], 3).astype(np.float32)
        f["flags_b"] = ctx.num["n_flags"][ib]
        f["flags_a"] = ctx.num["n_flags"][ia]
        f["oov_b"] = ctx.num["n_oov"][ib]
        f["ntok_a"] = ctx.num["n_ntok"][ia]
        f["ntok_b"] = ctx.num["n_ntok"][ib]
        f["nf_key_s1"] = ctx.freq["f1"][ia]
        f["nf_key_city_s1"] = ctx.freq["f2"][ia]
        f["nf_key_city_pool"] = ctx.freq["f3"][ia]
        f["nf_key_s1_cand"] = ctx.freq["f1"][ib]
        f["af_s1_addr"] = ctx.freq["f4"][ia]        # co-located S1 businesses at the S1 address
        f["af_s1_addr_cand"] = ctx.freq["f4"][ib]   # ... at the candidate's address
        f["af_s1_hn_cand"] = ctx.freq["f5"][ib]     # S1 sharing the candidate's (city, house number)
        # house number / unit / postcode
        f["hn_eq"] = _eq(ctx.col("a_hn", ia), ctx.col("a_hn", ib))
        both = ((ha != "") & (hb != "")).to_numpy()
        hal, hbl = ha.to_list(), hb.to_list()
        pre = np.array([(x != y) and bool(x) and bool(y) and (x.startswith(y) or y.startswith(x))
                        for x, y in zip(hal, hbl)], np.float32)
        f["hn_prefix"] = np.where(both, pre, -1).astype(np.float32)
        ia_i = ha.str.slice(0, 9).cast(pl.Int64, strict=False).to_numpy().astype(np.float64)
        ib_i = hb.str.slice(0, 9).cast(pl.Int64, strict=False).to_numpy().astype(np.float64)
        f["hn_absdiff"] = np.log1p(np.abs(ia_i - ib_i)).astype(np.float32)
        f["hn_reldiff"] = (np.abs(ia_i - ib_i) / np.maximum(np.maximum(ia_i, ib_i), 1)).astype(np.float32)
        f["hn_diglev"] = np.where(both, _cp(hal, hbl, Levenshtein.distance), np.nan).astype(np.float32)
        f["hns_eq"] = _eq(ctx.col("a_hns", ia), ctx.col("a_hns", ib))
        f["unit_eq"] = _eq(ctx.col("a_unit", ia), ctx.col("a_unit", ib))
        f["pc_eq"] = _eq(ctx.col("a_pc", ia), ctx.col("a_pc", ib))
        sta, stb = ctx.col("a_street", ia).to_list(), ctx.col("a_street", ib).to_list()
        f["st_tset"] = _cp(sta, stb, fuzz.token_set_ratio)
        f["st_jw"] = _cp(sta, stb, JaroWinkler.normalized_similarity)
        f["at_tset"] = _cp(ctx.col("a_tok", ia).to_list(), ctx.col("a_tok", ib).to_list(), fuzz.token_set_ratio)
        f["land_a"] = ctx.num["a_land"][ia]
        f["land_b"] = ctx.num["a_land"][ib]
        f["ncomp_a"] = ctx.num["a_ncomp"][ia]
        f["ncomp_b"] = ctx.num["a_ncomp"][ib]
        f["x_nm_ad"] = f["nm_tset"] / 100 * np.nan_to_num(f["at_jacc"])
        f["x_hn_st"] = np.clip(f["hnd_eq"], 0, 1) * np.nan_to_num(f["st_jacc"])
    return pairs.with_columns([pl.Series(k, np.asarray(v, dtype=np.float32)) for k, v in f.items()])


def features_chunked(ctx: Ctx, pairs: pl.DataFrame, tier: str, chunk=3_000_000) -> pl.DataFrame:
    parts = []
    t0 = time.time()
    for i in range(0, pairs.height, chunk):
        parts.append(features(ctx, pairs.slice(i, chunk), tier))
        print(f"  {tier} feats {ctx.split}: {min(i + chunk, pairs.height):,}/{pairs.height:,} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return pl.concat(parts)


if __name__ == "__main__":
    build_sets()
