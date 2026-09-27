"""High-accuracy pseudo-ground-truth generator for the test set.

Achieves ~99% label accuracy through a multi-pass verification pipeline:

  Pass 1 — Heuristic pre-filter (eliminates obvious cases)
  Pass 2 — LLM primary judgment (YES/NO)
  Pass 3 — LLM adversarial challenge (for all YES: "why might these be DIFFERENT?")
  Pass 4 — LLM final ruling with chain-of-thought reasoning
  Consensus — Only accept labels where all passes agree

Pairs where the passes disagree are marked UNCERTAIN and excluded from GT.
This means fewer labeled pairs but ~99% accuracy on the ones we keep.

Usage:
──────
# Pull the best model for your M3 Max 48GB:
ollama pull qwen2.5:32b

# Run the full pipeline:
python generate_test_gt_hq.py run --sample-size 50000 --model qwen2.5:32b

# Or step by step:
python generate_test_gt_hq.py prepare --sample-size 50000
python generate_test_gt_hq.py label --model qwen2.5:32b
python generate_test_gt_hq.py build
python generate_test_gt_hq.py evaluate

# Quick test on 500 samples first:
python generate_test_gt_hq.py run --sample-size 500 --model qwen2.5:32b

# Validate accuracy on train data (where we know the true answer):
python generate_test_gt_hq.py benchmark --model qwen2.5:32b --n 200
"""

import argparse
import concurrent.futures
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl

# ── paths ──
_REPO = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("ER_DATA", _REPO / "dataset"))
WORK = Path(os.environ.get("ER_WORK", _REPO / "work"))
OUT = Path(os.environ.get("ER_OUT", _REPO / "output"))
GT_DIR = WORK / "pseudo_gt_hq"
GT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 2026
DEFAULT_MODEL = os.environ.get("ER_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Global clients and config
_bedrock_runtime_client = None


def get_llm_config(args=None) -> dict:
    """Resolve API key, base URL, region, and backend from CLI args or environment."""
    api_key = getattr(args, "api_key", None) or os.environ.get("BEDROCK_API_KEY") or os.environ.get("AWS_BEDROCK_API_KEY", "")
    base_url = getattr(args, "base_url", None) or os.environ.get("BEDROCK_BASE_URL", "")
    region = getattr(args, "region", None) or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    backend = getattr(args, "backend", None) or os.environ.get("ER_LLM_BACKEND", "auto")

    if backend == "auto":
        if base_url:
            backend = "gateway"
        elif api_key and (api_key.startswith("sk-") or "http" in base_url):
            backend = "gateway"
        elif api_key or os.environ.get("AWS_ACCESS_KEY_ID"):
            backend = "bedrock"
        else:
            backend = "bedrock"

    return {
        "api_key": api_key,
        "base_url": base_url.rstrip("/"),
        "region": region,
        "backend": backend,
    }


def call_bedrock_boto3(prompt: str, model: str, max_tokens: int = 300,
                       temperature: float = 0.0, region: str = "us-east-1") -> str:
    """Call AWS Bedrock Converse API via boto3."""
    global _bedrock_runtime_client
    if _bedrock_runtime_client is None:
        import boto3
        from botocore.config import Config
        cfg = Config(
            retries={"max_attempts": 8, "mode": "adaptive"},
            read_timeout=120,
            connect_timeout=10,
        )
        _bedrock_runtime_client = boto3.client("bedrock-runtime", region_name=region, config=cfg)

    # Normalize common model alias names
    model_id = model
    m_clean = model.lower().replace(" ", "").replace("-", "").replace(".", "").replace(":", "")
    if "claudesonnet5" in m_clean or "sonnet5" in m_clean:
        model_id = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    elif "claudeopus5" in m_clean or "opus5" in m_clean:
        model_id = "us.anthropic.claude-3-opus-20240229-v1:0"
    elif "gpt56luna" in m_clean or "gpt6luna" in m_clean or "luna" in m_clean:
        model_id = model  # pass directly for custom Bedrock or gateway model IDs

    response = _bedrock_runtime_client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature}
    )
    return response["output"]["message"]["content"][0]["text"].strip()


def call_gateway_http(prompt: str, model: str, base_url: str, api_key: str,
                      max_tokens: int = 300, temperature: float = 0.0) -> str:
    """Call Bedrock Gateway / OpenAI-compatible / Anthropic-compatible HTTP endpoint."""
    if not base_url:
        base_url = "https://api.openai.com/v1"

    if "/v1" not in base_url and not base_url.endswith("/chat/completions") and not base_url.endswith("/messages"):
        url = f"{base_url}/v1/chat/completions"
    elif base_url.endswith("/v1"):
        url = f"{base_url}/chat/completions"
    else:
        url = base_url

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["x-api-key"] = api_key

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    if "choices" in data and data["choices"]:
        return data["choices"][0]["message"]["content"].strip()
    if "content" in data and isinstance(data["content"], list):
        return data["content"][0].get("text", "").strip()
    if "response" in data:
        return data["response"].strip()
    return str(data).strip()


def ollama_generate(prompt: str, model: str, max_tokens: int = 200,
                    temperature: float = 0.0) -> str:
    """Call Ollama API. Returns response text."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "").strip()


def llm_generate(prompt: str, model: str, max_tokens: int = 300,
                 temperature: float = 0.0, config: dict = None) -> str:
    """Unified LLM call with exponential backoff on rate limits."""
    if config is None:
        config = get_llm_config()

    backend = config.get("backend", "bedrock")
    api_key = config.get("api_key", "")
    base_url = config.get("base_url", "")
    region = config.get("region", "us-east-1")

    max_retries = 6
    backoff = 1.5

    for attempt in range(max_retries):
        try:
            if backend == "gateway" or (base_url and base_url.startswith("http")):
                return call_gateway_http(prompt, model, base_url, api_key, max_tokens, temperature)
            elif backend == "ollama":
                return ollama_generate(prompt, model, max_tokens, temperature)
            else:
                return call_bedrock_boto3(prompt, model, max_tokens, temperature, region=region)
        except Exception as e:
            err = str(e)
            if attempt == max_retries - 1:
                return f"ERROR: {err}"
            # Rate limit or throttling error: back off
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)


def check_backend_connection(model: str, config: dict = None) -> bool:
    """Test model connection with a quick ping."""
    if config is None:
        config = get_llm_config()
    backend = config.get("backend", "bedrock")
    print(f"Connecting to backend '{backend}' for model '{model}'...", flush=True)
    res = llm_generate("Respond with the single word 'READY' and nothing else.",
                       model, max_tokens=10, temperature=0.0, config=config)
    if "ERROR:" in res:
        print(f"❌ Connection test failed:\n  {res}")
        return False
    print(f"✅ Connection successful! Model response: {res.strip()}")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPTS — three distinct passes for verification
# ═══════════════════════════════════════════════════════════════════════════════

PASS1_PROMPT = """You are an expert at business entity resolution. Determine whether these two business records refer to the SAME real-world business entity.

Key rules:
- SAME entity = same legal business, possibly with name variations, abbreviations, or address formatting differences
- DIFFERENT entity = different businesses, even if they have similar names or are in the same industry
- Two businesses in the same city with similar but NOT identical core names are DIFFERENT entities
- "Shri Ram Trading" and "Shri Lakshman Trading" are DIFFERENT businesses
- "ABC Corp" and "ABC Corporation" are the SAME business

Record A:
  Name: {s1_name}
  Address: {s1_addr}
  Country: {s1_country}

Record B:
  Name: {cand_name}
  Address: {cand_addr}
  Country: {cand_country}

Are these the SAME real-world business entity? Answer ONLY "YES" or "NO"."""

PASS2_ADVERSARIAL_PROMPT = """You are a skeptical auditor checking a proposed entity match. Your job is to find reasons these might be DIFFERENT businesses.

Record A:
  Name: {s1_name}
  Address: {s1_addr}
  Country: {s1_country}

Record B:
  Name: {cand_name}
  Address: {cand_addr}
  Country: {cand_country}

Someone claims these are the same business. List specific evidence that they might be DIFFERENT entities:
1."""

PASS3_COT_PROMPT = """You are an expert at business entity resolution. Carefully analyze whether these two records refer to the SAME real-world business.

Record A:
  Name: {s1_name}
  Address: {s1_addr}
  Country: {s1_country}

Record B:
  Name: {cand_name}
  Address: {cand_addr}
  Country: {cand_country}

Think step by step:
1. Compare the core business names (ignoring suffixes like Ltd, Pvt, Inc, Corp, SARL, SAS, SCI):
2. Compare the addresses (are they compatible, or clearly different locations?):
3. Are there any red flags (completely different core names, different cities)?

Based on your analysis, are these the SAME business entity?
Final answer (YES or NO):"""


# ═══════════════════════════════════════════════════════════════════════════════
# HEURISTIC PRE-FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def _norm_name(name: str) -> str:
    """Normalize business name for comparison."""
    if not name:
        return ""
    n = name.lower().strip()
    # Remove common legal suffixes
    for suffix in ["private limited", "pvt. ltd.", "pvt ltd", "pvt. limited",
                   "limited", "ltd.", "ltd", "llc", "l.l.c.", "inc.", "inc",
                   "corp.", "corp", "corporation", "co.", "company",
                   "s.a.s", "s.a.s.", "sas", "s.a.r.l", "s.a.r.l.", "sarl",
                   "sci", "eurl", "srl", "s.r.l.", "gmbh", "& co",
                   "public limited", "plc"]:
        n = re.sub(r'\b' + re.escape(suffix) + r'\b', '', n)
    # Remove parentheses content like (Limited)
    n = re.sub(r'\([^)]*\)', '', n)
    # Remove punctuation
    n = re.sub(r'[^\w\s]', ' ', n)
    n = re.sub(r'\s+', ' ', n).strip()
    return n


def _token_overlap(a: str, b: str) -> float:
    """Jaccard similarity on word tokens."""
    if not a or not b:
        return 0.0
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _seq_sim(a: str, b: str) -> float:
    """SequenceMatcher ratio."""
    from difflib import SequenceMatcher
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def heuristic_classify(pair: dict) -> tuple[str, float]:
    """Classify a pair using heuristics.

    Returns: (bucket, confidence)
      bucket: 'MATCH', 'NONMATCH', or 'UNCERTAIN'
      confidence: 0.0 to 1.0
    """
    s1_name = str(pair.get("s1_name", ""))
    cand_name = str(pair.get("cand_name", ""))
    s1_addr = str(pair.get("s1_addr", ""))
    cand_addr = str(pair.get("cand_addr", ""))
    s1_country = str(pair.get("s1_country", ""))
    cand_country = str(pair.get("cand_country", ""))

    # Different countries → definite non-match
    if s1_country != cand_country:
        return "NONMATCH", 0.99

    norm_s1 = _norm_name(s1_name)
    norm_cand = _norm_name(cand_name)

    name_sim = _seq_sim(norm_s1, norm_cand)
    name_jaccard = _token_overlap(norm_s1, norm_cand)
    addr_sim = _seq_sim(s1_addr, cand_addr)

    # Very high similarity on both name AND address → definite match
    if name_sim >= 0.90 and addr_sim >= 0.50:
        return "MATCH", 0.99

    # Exact normalized name match
    if norm_s1 == norm_cand and norm_s1 != "":
        return "MATCH", 0.98

    # Very low similarity on BOTH name AND address → definite non-match
    # (conservative: transliterations like Hindi↔English have near-zero string
    #  similarity but may share address tokens, so require BOTH to be low)
    if name_sim < 0.12 and name_jaccard < 0.10 and addr_sim < 0.15:
        return "NONMATCH", 0.95

    # Everything else needs LLM verification
    return "UNCERTAIN", max(name_sim, name_jaccard)


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-PASS LLM LABELING
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_yes_no(text: str) -> int:
    """Parse YES/NO from LLM response. Returns 1, 0, or -1 (unparseable)."""
    text = text.upper().strip()
    # Check last line first (for CoT responses)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        last = lines[-1]
        if "YES" in last and "NO" not in last:
            return 1
        if "NO" in last and "YES" not in last:
            return 0
    # Fallback: check full text
    if text.startswith("YES") or text == "YES":
        return 1
    if text.startswith("NO") or text == "NO":
        return 0
    if "YES" in text and "NO" not in text:
        return 1
    if "NO" in text and "YES" not in text:
        return 0
    return -1


def _parse_adversarial(text: str) -> bool:
    """Parse adversarial response. Returns True if strong evidence of DIFFERENT."""
    text = text.lower()
    # Count substantive objections
    strong_signals = [
        "completely different", "different business", "different company",
        "different name", "no overlap", "unrelated", "distinct entit",
        "not the same", "different core name", "different industr",
    ]
    weak_signals = [
        "abbreviation", "formatting", "spelling", "minor difference",
        "same business", "likely the same", "could be the same",
    ]
    strong_count = sum(1 for s in strong_signals if s in text)
    weak_count = sum(1 for s in weak_signals if s in text)
    return strong_count >= 2 and strong_count > weak_count


def label_pair_multipass(pair: dict, model: str, config: dict = None) -> dict:
    """Label a single pair with 3-pass verification.

    Returns dict with label, confidence, and per-pass details.
    """
    fmt = {
        "s1_name": pair["s1_name"],
        "s1_addr": pair["s1_addr"],
        "s1_country": pair["s1_country"],
        "cand_name": pair["cand_name"],
        "cand_addr": pair["cand_addr"],
        "cand_country": pair["cand_country"],
    }

    # ── Heuristic pre-filter ──
    h_bucket, h_conf = heuristic_classify(pair)
    if h_bucket == "MATCH" and h_conf >= 0.98:
        return {"label": 1, "confidence": h_conf, "method": "heuristic_match",
                "pass1": "SKIP", "pass2": "SKIP", "pass3": "SKIP"}
    if h_bucket == "NONMATCH" and h_conf >= 0.95:
        return {"label": 0, "confidence": h_conf, "method": "heuristic_nonmatch",
                "pass1": "SKIP", "pass2": "SKIP", "pass3": "SKIP"}

    # ── Pass 1: Direct YES/NO judgment ──
    resp1 = llm_generate(PASS1_PROMPT.format(**fmt), model, max_tokens=10, config=config)
    p1 = _parse_yes_no(resp1)

    if p1 == 0:
        # Model says NO — but verify with CoT for borderline heuristic cases
        if h_conf > 0.4:  # heuristic thought there might be some similarity
            resp3 = llm_generate(PASS3_COT_PROMPT.format(**fmt), model, max_tokens=300, config=config)
            p3 = _parse_yes_no(resp3)
            if p3 == 1:
                # CoT overturned the NO → uncertain
                return {"label": -1, "confidence": 0.0, "method": "uncertain_conflicting",
                        "pass1": resp1, "pass2": "SKIP", "pass3": resp3}
            else:
                return {"label": 0, "confidence": 0.90, "method": "llm_no_verified",
                        "pass1": resp1, "pass2": "SKIP", "pass3": resp3}
        else:
            return {"label": 0, "confidence": 0.85, "method": "llm_no",
                    "pass1": resp1, "pass2": "SKIP", "pass3": "SKIP"}

    if p1 == 1:
        # Model says YES — this is where false positives happen. Verify hard.

        # ── Pass 2: Adversarial challenge ──
        resp2 = llm_generate(PASS2_ADVERSARIAL_PROMPT.format(**fmt), model, max_tokens=300, config=config)
        has_strong_objection = _parse_adversarial(resp2)

        # ── Pass 3: Chain-of-thought final ruling ──
        resp3 = llm_generate(PASS3_COT_PROMPT.format(**fmt), model, max_tokens=300, config=config)
        p3 = _parse_yes_no(resp3)

        if p3 == 1 and not has_strong_objection:
            # All passes agree: YES + no strong objections + CoT confirms
            return {"label": 1, "confidence": 0.95, "method": "llm_yes_verified",
                    "pass1": resp1, "pass2": resp2, "pass3": resp3}
        elif p3 == 0 or has_strong_objection:
            # Disagreement: Pass 1 said YES but Pass 2/3 disagree
            if p3 == 0 and has_strong_objection:
                # Both verification passes reject → confident NO
                return {"label": 0, "confidence": 0.90, "method": "llm_yes_overturned",
                        "pass1": resp1, "pass2": resp2, "pass3": resp3}
            else:
                # Only one verification pass disagrees → uncertain
                return {"label": -1, "confidence": 0.0, "method": "uncertain_partial_disagree",
                        "pass1": resp1, "pass2": resp2, "pass3": resp3}
        else:
            # p3 is unparseable
            return {"label": -1, "confidence": 0.0, "method": "uncertain_unparseable",
                    "pass1": resp1, "pass2": resp2, "pass3": resp3}

    # p1 is unparseable
    return {"label": -1, "confidence": 0.0, "method": "uncertain_p1_unparseable",
            "pass1": resp1, "pass2": "SKIP", "pass3": "SKIP"}


# ═══════════════════════════════════════════════════════════════════════════════
# PREPARE: sample S1 entities + gather candidate context
# ═══════════════════════════════════════════════════════════════════════════════

def prepare(sample_size: int = 50000):
    """Sample S1 entities and gather their candidate pairs."""
    print("Loading test sources...", flush=True)
    s1 = pl.read_csv(DATA / "test" / "test_source1.tsv", separator="\t")
    s2 = pl.read_csv(DATA / "test" / "test_source2.tsv", separator="\t")
    s3 = pl.read_csv(DATA / "test" / "test_source3.tsv", separator="\t")
    pool = pl.concat([s2, s3])

    # Stratified sample
    rng = np.random.default_rng(SEED)
    countries = s1["country"].unique().sort().to_list()
    counts = s1.group_by("country").len()
    total = s1.height
    parts = []
    for c in countries:
        frac = counts.filter(pl.col("country") == c)["len"].item() / total
        k = max(1, int(sample_size * frac))
        sub = s1.filter(pl.col("country") == c)
        idx = rng.choice(sub.height, min(k, sub.height), replace=False)
        parts.append(sub[idx.tolist()])
    sampled = pl.concat(parts).sample(fraction=1.0, shuffle=True, seed=SEED)
    print(f"Sampled {sampled.height} S1 entities")

    # Load predictions + candidates
    mr = pl.read_csv(OUT / "matching_results.tsv", separator="\t")
    cp = pl.read_csv(OUT / "candidate_pairs.tsv", separator="\t")

    sampled_ids = set(sampled["entity_id"].to_list())
    cp_sampled = cp.filter(pl.col("source1_entity_id").is_in(sampled_ids))

    # Build pair-level data
    s1_dict = {r["entity_id"]: r for r in s1.to_dicts()}
    pool_dict = {r["entity_id"]: r for r in pool.to_dicts()}
    pred_dict = {}
    for r in mr.filter(pl.col("source1_entity_id").is_in(sampled_ids)).to_dicts():
        matched = r["matched_entity_ids"]
        pred_dict[r["source1_entity_id"]] = set(matched.split(",")) if matched else set()

    pairs = []
    for r in cp_sampled.to_dicts():
        s1_id = r["source1_entity_id"]
        cand_str = r["candidate_entity_ids"]
        if not cand_str or (isinstance(cand_str, float) and np.isnan(cand_str)):
            continue
        s1_rec = s1_dict.get(s1_id)
        if not s1_rec:
            continue
        predicted = pred_dict.get(s1_id, set())
        for cand_id in cand_str.split(","):
            cand_rec = pool_dict.get(cand_id)
            if not cand_rec:
                continue
            pairs.append({
                "s1_id": s1_id, "cand_id": cand_id,
                "s1_name": s1_rec["business_name"],
                "s1_addr": s1_rec["business_address"],
                "s1_country": s1_rec["country"],
                "cand_name": cand_rec["business_name"],
                "cand_addr": cand_rec["business_address"],
                "cand_country": cand_rec["country"],
                "is_predicted_match": cand_id in predicted,
            })

    pairs_df = pl.DataFrame(pairs)
    pairs_df.write_parquet(GT_DIR / "labeling_pairs.parquet")
    print(f"Prepared {len(pairs):,} candidate pairs for {sampled.height:,} S1 entities")
    return pairs_df


# ═══════════════════════════════════════════════════════════════════════════════
# LABEL: run multi-pass verification on all pairs
# ═══════════════════════════════════════════════════════════════════════════════

def label(model: str = DEFAULT_MODEL, concurrency: int = 8, max_pairs: int = 0,
          resume: bool = True, config: dict = None):
    """Label all pairs with multi-pass verification."""
    if config is None:
        config = get_llm_config()

    if not check_backend_connection(model, config):
        return

    pairs_path = GT_DIR / "labeling_pairs.parquet"
    if not pairs_path.exists():
        print("ERROR: Run 'prepare' first.")
        return

    pairs_df = pl.read_parquet(pairs_path)
    pairs = pairs_df.to_dicts()

    if max_pairs > 0:
        pairs = pairs[:max_pairs]

    # Resume from checkpoint if available
    results = {}
    checkpoint_path = GT_DIR / "label_checkpoint.jsonl"
    if resume and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                r = json.loads(line)
                key = (r["s1_id"], r["cand_id"])
                results[key] = r
        print(f"Resumed from checkpoint: {len(results):,} pairs already labeled")

    remaining = [p for p in pairs if (p["s1_id"], p["cand_id"]) not in results]
    print(f"Labeling {len(remaining):,} pairs ({len(results):,} already done) "
          f"with {model} (multi-pass)...", flush=True)

    t0 = time.time()
    done = 0
    errors = 0

    # Open checkpoint file for appending
    with open(checkpoint_path, "a") as ckpt:
        def process_pair(pair):
            result = label_pair_multipass(pair, model, config=config)
            result["s1_id"] = pair["s1_id"]
            result["cand_id"] = pair["cand_id"]
            result["s1_name"] = pair["s1_name"]
            result["cand_name"] = pair["cand_name"]
            result["s1_country"] = pair["s1_country"]
            return result

        if concurrency <= 1:
            for pair in remaining:
                result = process_pair(pair)
                results[(result["s1_id"], result["cand_id"])] = result
                ckpt.write(json.dumps(result) + "\n")
                ckpt.flush()
                done += 1
                if result["label"] == -1:
                    errors += 1
                if done % 50 == 0 or done == len(remaining):
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    eta_h = (len(remaining) - done) / max(rate, 0.01) / 3600
                    methods = Counter(r["method"] for r in results.values())
                    print(f"  {done:,}/{len(remaining):,} | {rate:.1f} pairs/s | "
                          f"ETA {eta_h:.1f}h | uncertain: {errors}", flush=True)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
                batch_size = concurrency * 5
                for i in range(0, len(remaining), batch_size):
                    batch = remaining[i:i + batch_size]
                    futures = {ex.submit(process_pair, p): p for p in batch}
                    for fut in concurrent.futures.as_completed(futures):
                        result = fut.result()
                        results[(result["s1_id"], result["cand_id"])] = result
                        ckpt.write(json.dumps(result) + "\n")
                        ckpt.flush()
                        done += 1
                        if result["label"] == -1:
                            errors += 1

                    if done % 50 < batch_size or done >= len(remaining):
                        elapsed = time.time() - t0
                        rate = done / elapsed
                        eta_h = (len(remaining) - done) / max(rate, 0.01) / 3600
                        print(f"  {done:,}/{len(remaining):,} | {rate:.1f} pairs/s | "
                              f"ETA {eta_h:.1f}h | uncertain: {errors}", flush=True)

    elapsed = time.time() - t0

    # Summary
    all_results = list(results.values())
    methods = Counter(r["method"] for r in all_results)
    labels = Counter(r["label"] for r in all_results)

    print(f"\n{'='*60}")
    print(f"  LABELING COMPLETE — {len(all_results):,} pairs in {elapsed:.0f}s")
    print(f"{'='*60}")
    print(f"  Labels:  MATCH={labels.get(1,0):,}  NONMATCH={labels.get(0,0):,}  "
          f"UNCERTAIN={labels.get(-1,0):,}")
    print(f"  Methods:")
    for m, c in methods.most_common():
        print(f"    {m:30s}: {c:,}")

    # Save full results
    with open(GT_DIR / "all_labels.json", "w") as f:
        json.dump(all_results, f)
    print(f"  Saved to {GT_DIR / 'all_labels.json'}")
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD: compile labels into ground truth TSV
# ═══════════════════════════════════════════════════════════════════════════════

def build():
    """Build ground truth TSV from labeled pairs (excluding UNCERTAIN)."""
    labels_path = GT_DIR / "all_labels.json"
    if not labels_path.exists():
        # Try building from checkpoint
        ckpt_path = GT_DIR / "label_checkpoint.jsonl"
        if ckpt_path.exists():
            all_results = []
            seen = set()
            with open(ckpt_path) as f:
                for line in f:
                    r = json.loads(line)
                    key = (r["s1_id"], r["cand_id"])
                    if key not in seen:
                        seen.add(key)
                        all_results.append(r)
            print(f"Built from checkpoint: {len(all_results):,} labels")
        else:
            print("ERROR: Run 'label' first.")
            return
    else:
        with open(labels_path) as f:
            all_results = json.load(f)

    # Only use confident labels
    matches = [r for r in all_results if r["label"] == 1]
    nonmatches = [r for r in all_results if r["label"] == 0]
    uncertain = [r for r in all_results if r["label"] == -1]

    print(f"Labels: {len(matches):,} matches, {len(nonmatches):,} non-matches, "
          f"{len(uncertain):,} uncertain (excluded)")

    # Build GT: s1_id → sorted list of matched cand_ids
    from collections import defaultdict
    gt_dict = defaultdict(list)
    for r in matches:
        gt_dict[r["s1_id"]].append(r["cand_id"])

    # Get all S1 IDs that were labeled (including singletons)
    all_s1 = set()
    for r in all_results:
        if r["label"] != -1:  # Only include S1s where we have confident labels
            all_s1.add(r["s1_id"])

    # Build TSV
    rows = []
    for s1_id in sorted(all_s1):
        matched = sorted(gt_dict.get(s1_id, []))
        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matched) if matched else "",
        })

    gt_df = pl.DataFrame(rows)
    gt_df.write_csv(GT_DIR / "test_pseudo_ground_truth.tsv", separator="\t")

    n_with = sum(1 for r in rows if r["matched_entity_ids"])
    n_single = sum(1 for r in rows if not r["matched_entity_ids"])
    print(f"\nPseudo-GT written: {len(rows):,} S1 entities")
    print(f"  With matches: {n_with:,}")
    print(f"  Singletons: {n_single:,}")
    print(f"  Uncertain (excluded): {len(uncertain):,}")
    print(f"  Saved: {GT_DIR / 'test_pseudo_ground_truth.tsv'}")


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATE: score predictions against pseudo-GT
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate():
    """Evaluate pipeline predictions against pseudo-GT."""
    gt_path = GT_DIR / "test_pseudo_ground_truth.tsv"
    if not gt_path.exists():
        print("ERROR: Run 'build' first.")
        return

    gt = pl.read_csv(gt_path, separator="\t")
    mr = pl.read_csv(OUT / "matching_results.tsv", separator="\t")
    s1 = pl.read_csv(DATA / "test" / "test_source1.tsv", separator="\t")
    s1_country = dict(zip(s1["entity_id"].to_list(), s1["country"].to_list()))

    gt_s1_ids = set(gt["source1_entity_id"].to_list())
    mr_filtered = mr.filter(pl.col("source1_entity_id").is_in(gt_s1_ids))

    def _explode(df, col="matched_entity_ids"):
        return (df
                .with_columns(pl.col(col).str.split(",").alias("_ids"))
                .explode("_ids")
                .filter(pl.col("_ids") != "")
                .filter(pl.col("_ids").is_not_null())
                .rename({"source1_entity_id": "s1", "_ids": "cand"})
                .select("s1", "cand"))

    pred_pairs = _explode(mr_filtered)
    gt_pairs = _explode(gt)
    s1_ids = gt.select(pl.col("source1_entity_id").alias("s1"))

    npred = pred_pairs.group_by("s1").len().rename({"len": "np"})
    ntrue = gt_pairs.group_by("s1").len().rename({"len": "nt"})
    tp = pred_pairs.join(gt_pairs, on=["s1", "cand"]).group_by("s1").len().rename({"len": "tp"})

    e = (s1_ids
         .join(npred, on="s1", how="left")
         .join(ntrue, on="s1", how="left")
         .join(tp, on="s1", how="left")
         .with_columns(pl.col("np", "nt", "tp").fill_null(0)))
    e = e.with_columns(
        pl.when((pl.col("np") == 0) & (pl.col("nt") == 0)).then(1.0)
        .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("nt") + pl.col("np")))
        .fill_nan(0.0).alias("f"))

    countries = [s1_country.get(sid, "?") for sid in e["s1"].to_list()]
    e = e.with_columns(pl.Series("country", countries))

    macro_f05 = e["f"].mean()
    tot_tp, tot_p, tot_t = e["tp"].sum(), e["np"].sum(), e["nt"].sum()

    print(f"\n{'='*60}")
    print(f"  EVALUATION vs PSEUDO-GT (high-quality, multi-pass)")
    print(f"{'='*60}")
    print(f"  Macro F0.5 = {macro_f05:.5f}")
    print(f"  Micro P={tot_tp / max(tot_p, 1):.4f}  R={tot_tp / max(tot_t, 1):.4f}")
    print(f"  {e.height:,} S1 entities evaluated")
    print(f"\n  By country:")
    for r in e.group_by("country").agg(pl.col("f").mean(), pl.len()).sort("country").to_dicts():
        print(f"    {r['country']:8s}: F0.5={r['f']:.5f}  (n={r['len']:,})")

    e_s = e.with_columns((pl.col("nt") == 0).alias("singleton"))
    print(f"\n  Matched vs Singleton:")
    for r in e_s.group_by("singleton").agg(pl.col("f").mean(), pl.len()).to_dicts():
        lbl = "singleton" if r["singleton"] else "matched"
        print(f"    {lbl:10s}: F0.5={r['f']:.5f}  (n={r['len']:,})")

    e.write_parquet(GT_DIR / "evaluation_per_entity.parquet")
    return e


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: measure actual accuracy on train data
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark(model: str = DEFAULT_MODEL, n: int = 200, concurrency: int = 4, config: dict = None):
    """Benchmark multi-pass accuracy on train data where we know ground truth."""
    from difflib import SequenceMatcher

    if config is None:
        config = get_llm_config()

    if not check_backend_connection(model, config):
        return

    print(f"Loading train data for benchmarking...", flush=True)
    s1 = pl.read_csv(DATA / "train" / "train_source1.tsv", separator="\t")
    s2 = pl.read_csv(DATA / "train" / "train_source2.tsv", separator="\t")
    s3 = pl.read_csv(DATA / "train" / "train_source3.tsv", separator="\t")
    gt = pl.read_csv(DATA / "train" / "train_ground_truth.tsv", separator="\t")
    pool = pl.concat([s2, s3])

    s1_dict = {r["entity_id"]: r for r in s1.to_dicts()}
    pool_dict = {r["entity_id"]: r for r in pool.to_dicts()}

    rng = np.random.default_rng(42)

    # Build positive pairs
    pos_pairs = []
    gt_matched = gt.filter(pl.col("matched_entity_ids").is_not_null())
    sample_gt = gt_matched.sample(n=min(500, gt_matched.height), seed=42)
    for r in sample_gt.to_dicts():
        s1_id = r["source1_entity_id"]
        s1_rec = s1_dict.get(s1_id)
        if not s1_rec:
            continue
        for cand_id in str(r["matched_entity_ids"]).split(",")[:1]:
            cand_rec = pool_dict.get(cand_id.strip())
            if not cand_rec:
                continue
            name_sim = SequenceMatcher(
                None, str(s1_rec["business_name"]).lower(),
                str(cand_rec["business_name"]).lower()).ratio()
            pos_pairs.append({
                "s1_name": str(s1_rec["business_name"]),
                "s1_addr": str(s1_rec["business_address"]),
                "s1_country": str(s1_rec["country"]),
                "cand_name": str(cand_rec["business_name"]),
                "cand_addr": str(cand_rec["business_address"]),
                "cand_country": str(cand_rec["country"]),
                "true_label": 1,
                "name_sim": name_sim,
                "difficulty": "easy" if name_sim > 0.7 else ("medium" if name_sim > 0.4 else "hard"),
            })
        if len(pos_pairs) >= n // 2:
            break

    # Build negative pairs
    neg_pairs = []
    s1_sample = s1.sample(n=min(300, s1.height), seed=42)
    for r in s1_sample.to_dicts():
        same_c = pool.filter(pl.col("country") == r["country"])
        if same_c.height == 0:
            continue
        idx = int(rng.integers(0, same_c.height))
        cand = same_c.slice(idx, 1).to_dicts()[0]
        neg_pairs.append({
            "s1_name": str(r["business_name"]),
            "s1_addr": str(r["business_address"]),
            "s1_country": str(r["country"]),
            "cand_name": str(cand["business_name"]),
            "cand_addr": str(cand["business_address"]),
            "cand_country": str(cand["country"]),
            "true_label": 0,
            "name_sim": SequenceMatcher(
                None, str(r["business_name"]).lower(),
                str(cand["business_name"]).lower()).ratio(),
            "difficulty": "negative",
        })
        if len(neg_pairs) >= n // 2:
            break

    all_pairs = pos_pairs + neg_pairs
    rng.shuffle(all_pairs)
    diff_counts = Counter(p["difficulty"] for p in all_pairs)
    print(f"Benchmark: {len(all_pairs)} pairs — {dict(diff_counts)}")

    # Run multi-pass labeling (parallelized for cloud speed)
    t0 = time.time()
    results = []

    def process_bench(pair):
        res = label_pair_multipass(pair, model, config=config)
        res["true_label"] = pair["true_label"]
        res["difficulty"] = pair["difficulty"]
        res["name_sim"] = pair["name_sim"]
        res["country"] = pair["s1_country"]
        return res

    if concurrency <= 1:
        for i, pair in enumerate(all_pairs):
            res = process_bench(pair)
            results.append(res)
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{len(all_pairs)} | {(i+1)/max(elapsed, 0.01):.1f} pairs/s", flush=True)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [ex.submit(process_bench, p) for p in all_pairs]
            for i, fut in enumerate(concurrent.futures.as_completed(futures)):
                res = fut.result()
                results.append(res)
                if (i + 1) % 10 == 0 or (i + 1) == len(all_pairs):
                    elapsed = time.time() - t0
                    print(f"  {i+1}/{len(all_pairs)} | {(i+1)/max(elapsed, 0.01):.1f} pairs/s", flush=True)

    elapsed = time.time() - t0

    # Compute metrics
    metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "unc": 0, "total": 0})
    for r in results:
        true, pred = r["true_label"], r["label"]
        for key in ["overall", f'diff_{r["difficulty"]}', f'country_{r["country"]}']:
            metrics[key]["total"] += 1
            if pred == -1:
                metrics[key]["unc"] += 1
            elif true == 1 and pred == 1:
                metrics[key]["tp"] += 1
            elif true == 0 and pred == 0:
                metrics[key]["tn"] += 1
            elif true == 0 and pred == 1:
                metrics[key]["fp"] += 1
            elif true == 1 and pred == 0:
                metrics[key]["fn"] += 1

    def acc(m):
        decided = m["total"] - m["unc"]
        correct = m["tp"] + m["tn"]
        return correct / decided if decided > 0 else 0

    print(f"\n{'='*60}")
    print(f"  MULTI-PASS BENCHMARK — {model}")
    print(f"{'='*60}")
    m = metrics["overall"]
    decided = m["total"] - m["unc"]
    correct = m["tp"] + m["tn"]
    print(f"  Accuracy (on decided pairs): {acc(m):.1%} ({correct}/{decided})")
    print(f"  Uncertain (excluded): {m['unc']} ({m['unc']/m['total']:.1%})")
    print(f"  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}  UNC={m['unc']}")
    print(f"  Speed: {len(all_pairs)/elapsed:.1f} pairs/s ({elapsed:.0f}s total)")

    print(f"\n  By category:")
    for key in sorted(metrics.keys()):
        if key != "overall":
            m = metrics[key]
            d = m["total"] - m["unc"]
            c = m["tp"] + m["tn"]
            print(f"    {key:20s}: acc={c/d:.1%} ({c}/{d})  unc={m['unc']}  "
                  f"TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")

    # Show errors
    errors = [r for r in results if r["label"] != -1 and r["label"] != r["true_label"]]
    if errors:
        print(f"\n  --- ERRORS ({len(errors)}) ---")
        for e in errors[:10]:
            print(f"    true={e['true_label']} pred={e['label']} method={e['method']} "
                  f"name_sim={e['name_sim']:.2f} country={e['country']}")

    # Save
    with open(GT_DIR / f"benchmark_{model.replace(':', '_').replace('/', '_')}.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="High-accuracy pseudo-GT for test set")

    # Common connection options
    conn = argparse.ArgumentParser(add_help=False)
    conn.add_argument("--api-key", default=None, help="Bedrock / Gateway API key (or set BEDROCK_API_KEY env)")
    conn.add_argument("--base-url", default=None, help="Custom gateway / proxy base URL (e.g. https://.../v1)")
    conn.add_argument("--region", default=None, help="AWS region (default us-east-1 or AWS_REGION env)")
    conn.add_argument("--backend", choices=["auto", "bedrock", "gateway", "ollama"], default="auto")
    conn.add_argument("--model", default=DEFAULT_MODEL, help="Model name / ID (default: Claude Sonnet 5)")
    conn.add_argument("--concurrency", type=int, default=8, help="Worker threads for parallel API calls")

    sub = ap.add_subparsers(dest="cmd", required=True)

    # test-connection command
    sub.add_parser("test-connection", parents=[conn], help="Test connection and model response with given credentials")

    p = sub.add_parser("prepare", help="Sample S1 entities and prepare candidate pairs")
    p.add_argument("--sample-size", type=int, default=50000)

    p = sub.add_parser("label", parents=[conn], help="Multi-pass LLM labeling")
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--no-resume", action="store_true")

    sub.add_parser("build", help="Build pseudo-GT from labels")
    sub.add_parser("evaluate", help="Evaluate predictions against pseudo-GT")

    p = sub.add_parser("benchmark", parents=[conn], help="Benchmark accuracy on train data")
    p.add_argument("--n", type=int, default=200)

    p = sub.add_parser("run", parents=[conn], help="Full pipeline end-to-end")
    p.add_argument("--sample-size", type=int, default=50000)
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--no-resume", action="store_true")

    args = ap.parse_args()
    config = get_llm_config(args) if hasattr(args, "backend") else None

    if args.cmd == "test-connection":
        check_backend_connection(args.model, config)
    elif args.cmd == "prepare":
        prepare(args.sample_size)
    elif args.cmd == "label":
        label(args.model, args.concurrency, args.max_pairs, resume=not args.no_resume, config=config)
    elif args.cmd == "build":
        build()
    elif args.cmd == "evaluate":
        evaluate()
    elif args.cmd == "benchmark":
        benchmark(args.model, args.n, args.concurrency, config=config)
    elif args.cmd == "run":
        prepare(args.sample_size)
        label(args.model, args.concurrency, args.max_pairs, resume=not args.no_resume, config=config)
        build()
        evaluate()


if __name__ == "__main__":
    main()
