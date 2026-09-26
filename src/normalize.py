"""Text normalization for noisy business names and addresses.

Everything here is derived from the noise patterns the problem statement lists:
legal-suffix inconsistency, abbreviations, punctuation, transliteration,
landmark references, component reordering.

Deliberately conservative. Over-normalizing collapses genuinely different
businesses onto the same string ("ABC Corp" vs "ABC Ltd" are usually the same
entity; "ABC Motors" vs "ABC Motels" are not), and F_0.5 punishes false merges
twice as hard as misses.
"""
import re
import unicodedata

import pandas as pd

# Legal suffixes: dropped from the "core" name, kept as a separate signal.
LEGAL_SUFFIXES = {
    "corp", "corporation", "inc", "incorporated", "co", "company",
    "ltd", "limited", "llc", "llp", "lp", "plc",
    "pvt", "private", "pte", "gmbh", "ag", "nv", "bv", "sa", "sas", "sarl",
    "srl", "spa", "ab", "oy", "as", "kk",
    # French forms: test is 15% France with zero French training examples,
    # so these have to be handled from the tables rather than learned.
    "sasu", "eurl", "sci", "snc", "scop", "scm", "selarl",
    # Function words carry no identifying signal in a business name. English
    # ones were always stripped here; the French equivalents were not, which
    # left core_name() and core_addr() disagreeing on the same tokens —
    # "Maison de Sante" kept its "de" as a name but lost it as an address.
    # Measured on real French test rows: " de " 6.1% of names, " du " 4.2%,
    # " des " 3.0%.
    "and", "the", "of",
    "de", "du", "des", "la", "le", "les", "et", "aux", "au",
}

# Abbreviation expansions, applied to both names and addresses.
#
# Only unambiguous abbreviations belong here. An entry that collides with an
# ordinary English word ("all" -> "allee") or with a different expansion in
# another language rewrites text it should leave alone, and the damage is
# silent: two records that should match stop matching, or two that shouldn't
# start. When in doubt, leave the token as-is — the character-level features
# in features.py already absorb small variations.
ADDR_ABBREV = {
    # US / India street types
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "pl": "place", "sq": "square", "hwy": "highway", "pkwy": "parkway",
    "ste": "suite", "apt": "apartment", "fl": "floor", "bldg": "building",
    "opp": "opposite", "nr": "near", "jn": "junction",
    "mkt": "market", "clny": "colony", "sec": "sector",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "mt": "mount", "ft": "fort", "pk": "park",
    # French street types — test is 15% France with zero French training rows,
    # so these can only come from a table. "r" (rue) and "all" (allee) are
    # deliberately excluded: both collide with ordinary English tokens.
    "bd": "boulevard", "bvd": "boulevard",
    "rte": "route", "imp": "impasse", "fbg": "faubourg",
    "res": "residence", "bat": "batiment",
}

# Landmark/filler tokens that carry little matching signal on their own.
#
# Note these are only dropped for the *core* address form used in token-overlap
# features; the full normalized address keeps them. A normalization that fires
# on BOTH sides of a pair is harmless for matching — only asymmetric rewrites
# hurt — which is why borderline entries are safer here than in ABBREV.
ADDRESS_STOPWORDS = {
    "near", "opposite", "behind", "beside", "next", "to", "at", "in", "on",
    "the", "of", "and", "no", "number",
    # French articles and prepositions, very high frequency in FR addresses
    "de", "du", "des", "le", "la", "les", "et", "au", "aux", "chez", "sur",
}

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")
_NUM = re.compile(r"\d+")


def _strip_accents(s: str) -> str:
    """Fold accents so transliteration variants collide (café -> cafe)."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )


_POSSESSIVE = re.compile(r"['’]s\b")


def basic_clean(s) -> str:
    """Lowercase, de-accent, replace & with 'and', strip punctuation.

    Possessives are dropped *before* punctuation stripping. Otherwise
    "Orelee's Barbershop" tokenizes to ["orelee", "s", "barbershop"], and that
    stray "s" is then indistinguishable from a real token — which is how it
    used to get expanded to "south".
    """
    if s is None:
        return ""
    s = str(s)
    if s.lower() == "nan":
        return ""
    s = _strip_accents(s.lower())
    s = _POSSESSIVE.sub("", s)
    s = s.replace("&", " and ")
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def expand_abbrev(tokens):
    """Expand ADDRESS abbreviations. Only ever applied to address text.

    Street types and compass directions are address vocabulary. Applying them
    to business names rewrites initials and short words that merely look like
    abbreviations ("J. S. Motors" -> "j south motors"), which silently breaks
    matches instead of helping them.
    """
    return [ADDR_ABBREV.get(t, t) for t in tokens]


def norm_name(s) -> str:
    """Full normalized name. Legal suffixes kept; no address expansion."""
    return basic_clean(s)


def core_name(s) -> str:
    """Name with legal suffixes and stopwords removed.

    'Acme Pvt Ltd' and 'Acme Corporation' both reduce to 'acme'. Falls back to
    the full normalized name when stripping would leave nothing.
    """
    toks = [t for t in norm_name(s).split() if t not in LEGAL_SUFFIXES]
    return " ".join(toks) if toks else norm_name(s)


def norm_addr(s) -> str:
    return " ".join(expand_abbrev(basic_clean(s).split()))


def core_addr(s) -> str:
    """Address with landmark filler removed, for token-overlap features."""
    toks = [t for t in norm_addr(s).split() if t not in ADDRESS_STOPWORDS]
    return " ".join(toks) if toks else norm_addr(s)


def numeric_tokens(s) -> set:
    """Digits in the string — in practice house and unit numbers.

    Postal codes barely exist here (India 0.2%, US 11.0%, France 0.4% of
    addresses carry a 5-6 digit code), so do not treat this as a PIN/ZIP
    signal and do not block on postal code.

    These are high-precision signals — two addresses agreeing on 560001 is
    strong evidence, and disagreeing on it is strong evidence against.
    """
    return set(_NUM.findall(str(s or "")))


def acronym(s) -> str:
    """First letter of each core-name token: 'indian coffee house' -> 'ich'.

    Catches the abbreviation-vs-expansion case that string similarity misses.
    """
    return "".join(t[0] for t in core_name(s).split() if t)


def token_set(s) -> set:
    return set(s.split()) if s else set()


def add_blocking_columns(df, name_col=None, addr_col=None):
    """Attach only what blocking needs: a single normalized text blob.

    Runs over every record — up to ~10M per split — so it stays to one extra
    string column. The richer forms (and especially the Python sets) would cost
    several GB at this scale and are not needed until the candidate set is
    small; see add_feature_columns.
    """
    import config as C

    name_col = name_col or C.NAME
    addr_col = addr_col or C.ADDR
    if len(df) == 0:
        return df.assign(_blob=pd.Series(dtype=str))

    df = df.copy()
    df["_blob"] = (
        df[name_col].map(core_name) + " " + df[addr_col].map(core_addr)
    ).str.strip()

    # business_name + business_address are ~4GB of Python strings across the
    # 10.3M S2+S3 records and are not needed again until the feature stage,
    # which re-reads just the rows that survived blocking. Holding them here
    # is what pushes the run into swap.
    if C.BLOCK_DROP_TEXT:
        df = df.drop(columns=[c for c in (name_col, addr_col) if c in df.columns])
    return df


def add_feature_columns(df, name_col=None, addr_col=None):
    """Attach every normalized form, for the feature stage.

    Apply this only to the records that actually appear in candidate pairs —
    running it over all 10M records is what blows the memory budget.
    """
    import config as C

    name_col = name_col or C.NAME
    addr_col = addr_col or C.ADDR

    df = df.copy()
    df["_name"] = df[name_col].map(norm_name)
    df["_core_name"] = df[name_col].map(core_name)
    df["_addr"] = df[addr_col].map(norm_addr)
    df["_core_addr"] = df[addr_col].map(core_addr)
    df["_acronym"] = df[name_col].map(acronym)
    df["_name_nums"] = df[name_col].map(numeric_tokens)
    df["_addr_nums"] = df[addr_col].map(numeric_tokens)
    if "_blob" not in df.columns:
        df["_blob"] = (df["_core_name"] + " " + df["_core_addr"]).str.strip()
    return df


# ---------------------------------------------------------------------------
# v2 forms (features only). Nothing below feeds `_blob`, so blocking output and
# candidate caches are unchanged by these additions.
# ---------------------------------------------------------------------------
try:  # ISC licence, offline tables. Transliterates all 9 Indic scripts in the data.
    from anyascii import anyascii as _to_ascii
except ImportError:  # degrades to accent folding only; Indic names stay unmatched
    _to_ascii = _strip_accents

_NON_LATIN = re.compile(r"[^\x00-ɏ]")
_FIRST_NUM = re.compile(r"\d+")
_DOMAIN_RE = re.compile(r"^\W*(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|in|net|org|co|fr|biz|info|us)\s*$", re.I)

# Legal forms as they appear after transliteration of Indic-script names
# ("प्राइवेट लिमिटेड" -> "praivet limited"); kept out of LEGAL_SUFFIXES so the
# blocking blob does not change.
TRANSLIT_LEGAL = {"praivet", "prayvet", "privet", "piraivet", "limitet", "limited", "limted",
                  "elelpi", "pvt", "ltd", "llp", "inc", "llc", "corp", "co"}

# Canonical legal class. Two records with *different* classes (LLC vs SARL) are
# evidence against a match; a missing class is not (sources drop suffixes).
_LEGAL_CLASS = [
    ({"llc"}, "LLC"), ({"llp", "elelpi"}, "LLP"), ({"inc", "incorporated"}, "INC"),
    ({"corp", "corporation"}, "CORP"), ({"plc"}, "PLC"), ({"lp"}, "LP"),
    ({"sarl"}, "SARL"), ({"sas"}, "SAS"), ({"sasu"}, "SASU"), ({"eurl"}, "EURL"),
    ({"sci"}, "SCI"), ({"snc"}, "SNC"), ({"gmbh"}, "GMBH"),
]
_PRIVATE = {"pvt", "private", "praivet", "prayvet", "privet", "piraivet"}
_LIMITED = {"ltd", "limited", "limitet", "limted"}


def translit(s) -> str:
    """Latin rendering of any script; accents folded. 'राम मार्केटिंग' -> 'ram marketimg'."""
    s = "" if s is None else str(s)
    return _to_ascii(s) if _NON_LATIN.search(s) else _strip_accents(s)


def is_native_script(s) -> bool:
    return bool(_NON_LATIN.search("" if s is None else str(s)))


def translit_core(s) -> str:
    """core_name on the transliterated text, with transliterated legal words dropped.
    A domain-style name reduces to its stem ('deltatelecommunication.com' -> stem)."""
    stem = domain_stem(s)
    if stem:
        return stem
    toks = [t for t in core_name(translit(s)).split() if t not in TRANSLIT_LEGAL]
    return " ".join(toks)


def skeleton(core: str) -> str:
    """Consonant skeleton of an already-transliterated core name.

    Verified on 20k native-script India positives: 94.4% share >=1 skeleton
    token with their S1 name (raw tokens: ~0%). Coarse by design — a feature,
    never a decision (tools/eda/skeleton_prototype.py holds the self-check).
    """
    s = re.sub(r"[^a-z ]", "", core.lower())
    s = re.sub(r"\by(?=[aeiou])", "", s)
    s = re.sub(r"(?<=[aeiou])gh", "", s)
    for a, b in (("ction", "ksn"), ("tion", "sn"), ("ph", "f"), ("th", "t"),
                 ("kh", "k"), ("sh", "s"), ("ch", "s"), ("ck", "k")):
        s = s.replace(a, b)
    s = re.sub(r"c(?=[aoukrlt]|\b)", "k", s)
    s = re.sub(r"m(?=[^aeiou ])", "n", s)
    s = s.translate(str.maketrans("gbdvzjcq", "kptwsssk"))
    out = []
    for w in s.split():
        w = w[0] + re.sub(r"[aeiouy]", "", w[1:])
        out.append(re.sub(r"(.)\1+", r"\1", w))
    return " ".join(out)


def legal_form(s) -> str:
    """Canonical legal class from anywhere in the name ('' when none)."""
    toks = set(basic_clean(translit(s)).split())
    if toks & _PRIVATE and toks & _LIMITED:
        return "PVT_LTD"
    for keys, cls in _LEGAL_CLASS:
        if toks & keys:
            return cls
    return "LTD" if toks & _LIMITED else ""


def first_number(s) -> str:
    """First digit run of the raw address — usually the house/plot number."""
    m = _FIRST_NUM.search("" if s is None else str(s))
    return m.group(0) if m else ""


def domain_stem(s) -> str:
    """'deltatelecommunication.com' -> 'deltatelecommunication'; '' if not a domain."""
    m = _DOMAIN_RE.match(_strip_accents("" if s is None else str(s)).strip())
    return m.group(1).lower().replace("-", "") if m else ""
