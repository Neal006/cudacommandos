"""Stage A: name and address normalisation for every record (train + test).

Writes norm_{split}.parquet, one row per record (same idx as records_{split}.parquet).
Transliteration always runs before accent stripping: NFKD + combining-mark removal would
otherwise delete Devanagari viramas / nuktas.
"""
import math
import pickle
import re
import sys
import unicodedata
from collections import Counter
from multiprocessing import get_context

import polars as pl

from config import N_JOBS, SPLITS, wpath

# ----------------------------------------------------------------------------------------------
# lexicons
# ----------------------------------------------------------------------------------------------
SUFFIX_CANON = {
    "corporation": "corp", "corp": "corp", "incorporated": "inc", "inc": "inc",
    "company": "co", "co": "co", "compagnie": "co", "cie": "co",
    "limited": "ltd", "ltd": "ltd", "private": "pvt", "pvt": "pvt", "pte": "pvt",
    "llc": "llc", "pllc": "pllc", "llp": "llp", "lp": "lp", "plc": "plc", "pc": "pc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "sci": "sci", "snc": "snc", "eurl": "eurl",
}
LEGAL = set(SUFFIX_CANON.values())
STOP = {"and", "of", "the", "de", "des", "du", "la", "le", "les", "d", "l", "et", "a"}
# measured on train true pairs: 0/1/5/6/8 replace o/l/s/g/b 99.8%+ of the time; 3/4/7/9 are real digits
LEET = str.maketrans({"0": "o", "1": "l", "5": "s", "6": "g", "8": "b"})
ORDINAL = re.compile(r"^\d+(st|nd|rd|th)$")

JUNK = re.compile(r"^[\W_]+")
MS = re.compile(r"^\s*m/s\.?\s+", re.I)
ALIAS = re.compile(r"\s+(?:d\.?\s?b\.?\s?a\.?|a\.?k\.?a\.?|t/a|trading\s+as)\s+", re.I)
DOMAIN = re.compile(r"^(?:www\.)?([a-z0-9\-]+)\.com$")
URL_SUFFIX = re.compile(r"\s*\|\s*(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.[a-z]{2,4}\S*\s*$", re.I)  # "Name | www.x.com"
APOS = re.compile(r"['’`´]")
NONWORD = re.compile(r"[^\w\s]")

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
    "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee", "tx": "texas",
    "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia", "pr": "puerto rico",
}
IN_STATES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar", "cg": "chhattisgarh",
    "ct": "chhattisgarh", "ga": "goa", "gj": "gujarat", "hr": "haryana", "hp": "himachal pradesh",
    "jk": "jammu and kashmir", "jh": "jharkhand", "ka": "karnataka", "kl": "kerala", "mp": "madhya pradesh",
    "mh": "maharashtra", "mn": "manipur", "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland", "od": "odisha",
    "or": "odisha", "pb": "punjab", "rj": "rajasthan", "sk": "sikkim", "tn": "tamil nadu", "ts": "telangana",
    "tg": "telangana", "tr": "tripura", "up": "uttar pradesh", "uk": "uttarakhand", "ut": "uttarakhand",
    "wb": "west bengal", "dl": "delhi", "ch": "chandigarh", "py": "puducherry", "la": "ladakh",
    "an": "andaman and nicobar islands", "dn": "dadra and nagar haveli", "dd": "daman and diu", "ld": "lakshadweep",
}
IN_ALIASES = {"orissa": "odisha", "nct of delhi": "delhi", "new delhi": None, "pondicherry": "puducherry",
              "uttaranchal": "uttarakhand", "j and k": "jammu and kashmir", "jammu kashmir": "jammu and kashmir"}
FR_REGIONS = {
    "hauts de france": ["nord", "pas de calais", "somme", "aisne", "oise"],
    "nouvelle aquitaine": ["gironde", "landes", "pyrenees atlantiques", "dordogne", "lot et garonne",
                           "charente", "charente maritime", "vienne", "haute vienne", "deux sevres",
                           "creuse", "correze"],
    "pays de la loire": ["loire atlantique", "maine et loire", "vendee", "sarthe", "mayenne"],
    "ile de france": ["paris", "seine saint denis", "hauts de seine", "val de marne", "essonne", "yvelines",
                      "val d oise", "seine et marne"],
    "bretagne": ["ille et vilaine", "finistere", "morbihan", "cotes d armor"],
    "auvergne rhone alpes": ["rhone", "isere", "loire", "ain", "savoie", "haute savoie", "drome", "ardeche"],
    "provence alpes cote d azur": ["bouches du rhone", "var", "alpes maritimes", "vaucluse"],
    "occitanie": ["haute garonne", "herault", "gard"],
    "grand est": ["bas rhin", "haut rhin", "moselle", "marne"],
    "normandie": ["seine maritime", "calvados", "manche", "eure", "orne"],
    "centre val de loire": ["loiret", "indre et loire", "cher", "eure et loir", "loir et cher", "indre"],
    "bourgogne franche comte": ["cote d or", "doubs", "saone et loire", "yonne"],
    "corse": ["corse du sud", "haute corse"],
}
FR_DEPT_NUM = {"59": "hauts de france", "62": "hauts de france", "33": "nouvelle aquitaine",
               "44": "pays de la loire", "75": "ile de france"}


def _state_lex():
    lex = {"US": {}, "India": {}, "France": {}}
    for k, v in US_STATES.items():
        lex["US"][k] = k
        lex["US"][v] = k
    for k, v in IN_STATES.items():
        lex["India"][k] = v
        lex["India"][v] = v
    for k, v in IN_ALIASES.items():
        if v:
            lex["India"][k] = v
    # S2/S3 label Telangana cities as Andhra Pradesh (~29K train pairs): treat them as one state
    for k, v in list(lex["India"].items()):
        if v in ("telangana", "andhra pradesh"):
            lex["India"][k] = "andhra pradesh telangana"
    for reg, depts in FR_REGIONS.items():
        lex["France"][reg] = reg
        for d in depts:
            lex["France"][d] = reg
    return lex


STATE_LEX = _state_lex()
POSTCODE = {"US": re.compile(r"^\d{5}(?:-\d{4})?$"), "India": re.compile(r"^\d{6}$|^\d{3} \d{3}$"),
            "France": re.compile(r"^\d{5}$")}
POSTCODE_ANY = re.compile(r"^\d{4,6}$")

STREET_US = {"st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "dr": "drive",
             "blvd": "boulevard", "ln": "lane", "ct": "court", "pl": "place", "cir": "circle", "pkwy": "parkway",
             "hwy": "highway", "trl": "trail", "ter": "terrace", "terr": "terrace", "sq": "square", "cv": "cove",
             "pt": "point", "xing": "crossing", "hts": "heights", "mt": "mount", "ft": "fort", "cres": "crescent",
             "expy": "expressway", "fwy": "freeway", "tpke": "turnpike", "aly": "alley", "plz": "plaza",
             "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
             "se": "southeast", "sw": "southwest", "co": "county", "cr": "county road"}
STREET_IN = {"rd": "road", "st": "street", "opp": "opposite", "nr": "near", "sec": "sector", "ngr": "nagar",
             "apt": "apartment", "apts": "apartments", "bldg": "building", "flr": "floor", "fl": "floor",
             "mkt": "market", "extn": "extension", "ext": "extension", "colony": "colony", "clny": "colony"}
STREET_FR = {"r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "boul": "boulevard",
             "bld": "boulevard", "bvd": "boulevard", "imp": "impasse", "all": "allee", "ch": "chemin",
             "chem": "chemin", "pl": "place", "rte": "route", "crs": "cours", "qu": "quai", "st": "saint",
             "ste": "sainte", "sq": "square", "fg": "faubourg", "fbg": "faubourg", "res": "residence",
             "pass": "passage", "prom": "promenade", "sent": "sentier", "lot": "lotissement"}
STREET_EXP = {"US": STREET_US, "India": STREET_IN, "France": STREET_FR}
STREET_WORDS = {
    "street", "road", "avenue", "drive", "boulevard", "lane", "court", "place", "circle", "parkway", "highway",
    "trail", "terrace", "way", "square", "cove", "point", "crossing", "heights", "plaza", "alley", "loop", "pike",
    "row", "run", "path", "walk", "expressway", "freeway", "turnpike", "crescent", "route", "unit", "apt",
    "apartment", "suite", "floor", "box", "po", "rue", "impasse", "allee", "chemin", "cours", "quai",
    "faubourg", "residence", "passage", "promenade", "sentier", "lotissement", "near", "opposite", "behind",
    "beside", "sector", "block", "cross", "main", "marg", "flat", "plot", "house", "door", "shop", "office",
    "building", "tower", "complex", "floorsant", "phase", "stage", "gali", "bis", "ter",
}
LANDMARK = {"near", "opposite", "behind", "beside", "opp", "nr", "next"}
HN_MARK = re.compile(
    r"^(?:#|no\.?|n°|nº|h\.?\s?no\.?|hno|house\s*no\.?|door\s*no\.?|door|d\.?\s?no\.?|plot\s*no\.?|plot|"
    r"flat\s*no\.?|kh\s*no\.?|khasra\s*no\.?|shop\s*no\.?|office\s*no\.?)\s*[-:.#]*\s*")
UNIT_COMP = re.compile(r"^(?:unit|apt|apartment|suite|ste|fl|floor|room|rm|bldg|po box|p o box|box)\b")
UNIT_VAL = re.compile(r"\b(?:unit|apt|apartment|suite|fl|floor|room|rm)\s*[#:.-]*\s*([a-z]?\d+[a-z]?|[a-z])\b")
HN_TOK = re.compile(r"[a-z]{0,3}[-/]?\d[\w/-]*")
CITY_PRE = re.compile(r"^(?:city of|town of|village of|township of|borough of|cdp of)\s+")
CITY_SUF = re.compile(r"\s+(?:city|town|cdp|township|village|borough|hq region|region|dcp)$")
LEAD_ZEROS = re.compile(r"(?<!\d)0+(?=\d)")


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def collapse_single_letters(toks):
    out, run = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run) if len(run) > 1 else run[0])
            run = []
        out.append(t)
    if run:
        out.append("".join(run) if len(run) > 1 else run[0])
    return out


def leet(t: str) -> str:
    if ORDINAL.match(t) or t.isdigit() or t.isalpha():
        return t
    return t.translate(LEET)


def basic_tokens(s: str):
    """lower-cased, accent-free, punctuation-free tokens (text must already be Latin)."""
    s = strip_accents(s).lower().replace("&", " and ").replace("+", " and ")
    s = APOS.sub("", s)
    s = NONWORD.sub(" ", s).replace("_", " ")
    return collapse_single_letters(s.split())


_SEG = None


def _seg_vocab():
    global _SEG
    if _SEG is None:
        with open(wpath("seg_vocab.pkl"), "rb") as f:
            _SEG = pickle.load(f)
    return _SEG


def segment(s: str):
    """Word-break a concatenated domain base with unigram costs learnt from S1 names."""
    voc = _seg_vocab()
    n = len(s)
    if n == 0:
        return []
    unk = 12.0
    best = [0.0] + [math.inf] * n
    back = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(max(0, i - 20), i):
            w = s[j:i]
            c = voc.get(w)
            cost = best[j] + (c if c is not None else unk * (i - j))
            if cost < best[i]:
                best[i], back[i] = cost, j
    out, i = [], n
    while i > 0:
        out.append(s[back[i]:i])
        i = back[i]
    return collapse_single_letters(out[::-1])


# ----------------------------------------------------------------------------------------------
# names
# ----------------------------------------------------------------------------------------------
def _variant(text: str):
    """Return dict for one name variant; text is Latin."""
    low = strip_accents(text).lower().strip()
    m = DOMAIN.match(low.replace(" ", ""))
    is_dom = bool(m)
    if is_dom:
        base = m.group(1).replace("-", "")
        base = "".join(leet(t) for t in [base])
        toks = segment(base)
    else:
        toks = [leet(t) for t in basic_tokens(text)]
    toks = [SUFFIX_CANON.get(t, t) for t in toks]
    core = [t for t in toks if t not in LEGAL and t not in STOP] or [t for t in toks if t not in STOP] or toks
    return toks, core, is_dom


def norm_name(raw: str):
    from translit import translit_name
    s = unicodedata.normalize("NFKC", raw).strip()
    s = MS.sub("", JUNK.sub("", s)).strip()
    lat, n_ind, n_oov = translit_name(s)
    site = URL_SUFFIX.search(lat)
    if site:                                    # appended website: name = left part, site = extra domain variant
        lat = lat[:site.start()]
    parts = [p for p in ALIAS.split(lat) if p.strip()]
    texts = [lat] + (parts if len(parts) > 1 else []) + ([f"{site.group(1)}.com"] if site else [])
    flags = 0
    if len(parts) > 1:
        flags |= 2
    if n_ind:
        flags |= 8
        if n_ind < len(s.split()):
            flags |= 16
    variants = []
    main = None
    for i, t in enumerate(texts):
        toks, core, is_dom = _variant(t)
        if i == 0:
            main = (toks, core, is_dom)
        variants.append(" ".join(core))
        # raw (non-leet) form as extra variant when leet decoding changed something
        if not is_dom:
            raw_core = [SUFFIX_CANON.get(x, x) for x in basic_tokens(t)]
            raw_core = [x for x in raw_core if x not in LEGAL and x not in STOP] or raw_core
            rc = " ".join(raw_core)
            if rc != variants[-1]:
                flags |= 4
                variants.append(rc)
    toks, core, is_dom = main
    if is_dom:
        flags |= 1
    core_s = " ".join(core)
    alt = [v for v in dict.fromkeys(variants) if v and v != core_s]
    low = strip_accents(s).lower()
    ce = f"{lat.lower()} ({s})" if n_ind else low
    return (core_s, " ".join(toks), " ".join(sorted(set(core))), "".join(core),
            " ".join(sorted({t for t in toks if t in LEGAL})), "|".join(alt), flags,
            (n_oov / n_ind) if n_ind else 0.0, len(core), ce)


NAME_COLS = ["n_core", "n_full", "n_key", "n_concat", "n_suffix", "n_alt", "n_flags", "n_oov", "n_ntok", "n_ce"]


# ----------------------------------------------------------------------------------------------
# addresses
# ----------------------------------------------------------------------------------------------
def _clean_comp(c: str) -> str:
    c = strip_accents(c).lower()
    c = APOS.sub(" ", c)
    c = re.sub(r"[^\w\s/#°-]", " ", c).replace("_", " ")
    return re.sub(r"\s+", " ", c).strip()


def _lex_key(c: str) -> str:
    return re.sub(r"\s+", " ", c.replace("-", " ").replace("/", " ")).strip()


def clean_city(c: str) -> str:
    c = _lex_key(c)
    c = CITY_PRE.sub("", c)
    c = CITY_SUF.sub("", c)
    return c.strip()


def _expand(tokens, country):
    exp = STREET_EXP.get(country, {})
    out = []
    for t in tokens:
        m = ORDINAL.match(t)
        if m:
            t = t[: -2]
        out.append(exp.get(t, t))
    return out


def norm_addr(raw: str, country: str):
    from translit import translit_addr
    a = translit_addr(unicodedata.normalize("NFKC", raw))
    comps = [_clean_comp(p) for p in a.split(",")]
    comps = [c for c in comps if c and c != "null"]
    comps = [re.sub(r"\bnull\b", "", c).strip() for c in comps]
    comps = [c for c in comps if c]
    slex = STATE_LEX.get(country, {})
    pcre = POSTCODE.get(country, POSTCODE_ANY)
    state = pc = state_comp = ""
    state_is_code = False
    locs, streets = [], []
    for c in comps:
        k = _lex_key(c.replace("#", " ").replace("°", " "))
        if k in slex:
            is_code = len(k) <= 3
            if not state or (is_code and not state_is_code):
                if state and state_comp:
                    locs.append(clean_city(state_comp))      # the displaced full name is a city
                state, state_is_code, state_comp = slex[k], is_code, k
            else:
                locs.append(clean_city(k))
            continue
        if pcre.match(k):
            pc = k
            continue
        toks = k.split()
        # "59000 lille" / "oh 43004" / "lille 59000"
        pcs = [t for t in toks if pcre.match(t)]
        if pcs and len(toks) <= 4:
            rest = " ".join(t for t in toks if t not in pcs)
            if rest in slex:
                pc, state = pcs[0], slex[rest]
                continue
            if (rest and len(rest.split()) <= 3 and not any(ch.isdigit() for ch in rest)
                    and not (set(_expand(rest.split(), country)) & STREET_WORDS)):
                pc = pcs[0]
                locs.append(clean_city(rest))
                continue
        exp = _expand(toks, country)
        if any(ch.isdigit() for ch in k) or (set(exp) & STREET_WORDS) or (set(toks) & LANDMARK):
            streets.append(c)
        else:
            cc = clean_city(k)
            if cc:
                locs.append(cc)
    # house number: first street component that is not a unit/floor component
    hn = hnd = hns = unit = ""
    street_toks, nums = [], set()
    for c in streets:
        u = UNIT_VAL.search(c) if country != "France" else None
        if u and not unit:
            unit = LEAD_ZEROS.sub("", u.group(1))
        is_unit = bool(UNIT_COMP.match(c)) and country != "India"
        rest = HN_MARK.sub("", c)
        if not hn and not is_unit:
            m = HN_TOK.search(rest)
            if m and m.start() <= 3:
                tok = m.group(0).strip("-/")
                hn = LEAD_ZEROS.sub("", tok)
                d = re.search(r"\d+", tok)
                hnd = d.group(0).lstrip("0") or "0"
                sm = re.search(r"\d([a-z]+)$", tok) or re.match(r"\s*(bis|ter)\b", rest[m.end():])
                hns = sm.group(1) if sm else ""
                rest = rest[:m.start()] + " " + rest[m.end():]
        toks = re.sub(r"[#°/-]", " ", rest).split()
        if not is_unit:
            street_toks.extend(t for t in _expand(toks, country) if t not in LANDMARK and t not in ("bis", "ter"))
    for d in re.findall(r"\d+", " ".join(comps)):
        nums.add(d.lstrip("0") or "0")
    land = any(set(c.split()) & LANDMARK for c in comps)
    loc_toks = [t for loc in locs for t in loc.split()]
    all_toks = sorted(set(street_toks) | set(loc_toks) | ({state} if state else set()))
    clean = ", ".join(comps)
    return (state, "|".join(dict.fromkeys(locs)), (locs[-1] if locs else ""), pc, hn, hnd, hns, unit,
            " ".join(street_toks), " ".join(sorted(nums, key=lambda x: (len(x), x))), " ".join(all_toks),
            len(comps) == 0, len(comps), land, clean)


ADDR_COLS = ["a_state", "a_loc", "a_city", "a_pc", "a_hn", "a_hnd", "a_hns", "a_unit", "a_street", "a_nums",
             "a_tok", "a_empty", "a_ncomp", "a_land", "a_ce"]


# ----------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------
def _work(chunk):
    idx, names, addrs, countries = chunk
    nrows = [norm_name(n) for n in names]
    arows = [norm_addr(a, c) for a, c in zip(addrs, countries)]
    df = pl.DataFrame({"idx": idx, **{c: list(v) for c, v in zip(NAME_COLS, zip(*nrows))},
                       **{c: list(v) for c, v in zip(ADDR_COLS, zip(*arows))}})
    return df.with_columns(pl.col("n_flags").cast(pl.Int8), pl.col("n_oov").cast(pl.Float32),
                           pl.col("n_ntok").cast(pl.Int16), pl.col("a_ncomp").cast(pl.Int8))


def build_seg_vocab():
    from io_utils import load_records
    cnt = Counter()
    for split in SPLITS:
        s1 = load_records(split, ["src", "name"]).filter(pl.col("src") == 1)["name"].to_list()
        for n in s1:
            for t in basic_tokens(n):
                if t.isalpha():
                    cnt[t] += 1
    tot = sum(cnt.values())
    voc = {w: -math.log(c / tot) for w, c in cnt.items() if c >= 2 and len(w) >= 2 or w in ("a",)}
    with open(wpath("seg_vocab.pkl"), "wb") as f:
        pickle.dump(voc, f)
    print("seg vocab", len(voc))


def normalize_split(split: str, chunk=20000):
    from io_utils import load_records
    rec = load_records(split, ["idx", "src", "name", "addr", "country"])
    n = rec.height

    def chunks():
        for i in range(0, n, chunk):
            c = rec.slice(i, chunk)
            yield (c["idx"].to_list(), c["name"].to_list(), c["addr"].to_list(), c["country"].to_list())

    parts = []
    # "spawn", not the Linux default "fork": by this point polars has started its
    # rayon threads, and forking a process that holds live threads gives the children
    # mutexes no thread owns -- the pool then hangs at 0% CPU instead of failing.
    with get_context("spawn").Pool(N_JOBS) as pool:
        for k, part in enumerate(pool.imap(_work, chunks())):
            parts.append(part)
            if k % 100 == 0:
                print(f"  {split}: {min((k + 1) * chunk, n)}/{n}", flush=True)
    df = rec.select("idx", "src", "country").join(pl.concat(parts), on="idx").sort("idx")
    return df


def fill_state(dfs):
    """Fill missing a_state from the locality -> state majority map learnt on S1 (train + test)."""
    s1 = pl.concat([d.filter((pl.col("src") == 1) & (pl.col("a_state") != ""))
                    .select("country", "a_loc", "a_state") for d in dfs])
    m = (s1.with_columns(pl.col("a_loc").str.split("|")).explode("a_loc").filter(pl.col("a_loc") != "")
           .group_by("country", "a_loc", "a_state").len()
           .with_columns(pl.col("len").sum().over("country", "a_loc").alias("tot"))
           .filter((pl.col("len") >= 5) & (pl.col("len") / pl.col("tot") >= 0.9))
           .select("country", "a_loc", pl.col("a_state").alias("st_fill")))
    out = []
    for d in dfs:
        miss = (d.filter(pl.col("a_state") == "").select("idx", "country", "a_loc")
                 .with_columns(pl.col("a_loc").str.split("|").alias("l")).explode("l")
                 .join(m.rename({"a_loc": "l"}), on=["country", "l"])
                 .group_by("idx").agg(pl.col("st_fill").first()))
        d = (d.join(miss, on="idx", how="left")
              .with_columns(pl.when(pl.col("a_state") == "").then(pl.col("st_fill").fill_null(""))
                            .otherwise(pl.col("a_state")).alias("a_state"))
              .drop("st_fill").sort("idx"))
        out.append(d)
    return out


def main():
    import translit
    translit.build()
    build_seg_vocab()
    dfs = [normalize_split(s) for s in SPLITS]
    dfs = fill_state(dfs)
    for s, d in zip(SPLITS, dfs):
        d.write_parquet(wpath(f"norm_{s}.parquet"))
        print(s, d.height, "state missing:",
              d.group_by("src").agg((pl.col("a_state") == "").mean()).sort("src").to_dicts())


def load_norm(split: str, columns=None) -> pl.DataFrame:
    return pl.read_parquet(wpath(f"norm_{split}.parquet"), columns=columns)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "vocab":
        build_seg_vocab()
    else:
        main()
