"""Pull every finished cloud submission, validate it, and rank them on one screen.

There will be three or four candidates today (hopeso alone, hopeso+e5,
hopeso+bge, and the non-hopeso v4 from the laptop) and only one gets uploaded.
Comparing them by hand invites picking the wrong one, so this prints the single
table the decision needs and refuses to rank anything the validator rejects.

    python tools/pick_submission.py --list
    python tools/pick_submission.py --fetch amlc-a-p98b --fetch amlc-e5-p98c
    python tools/pick_submission.py --local runs/011_v4_contention

The ranking key is `holdout_score`, not `oof_score`. OOF is measured with one
fold model per entity; test is scored by the mean of five, and the holdout is
the only number computed the way test actually is. Where a run has no holdout
it is listed but marked, never silently compared against one that has.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import boto3

BUCKET = "sagemaker-ap-south-1-654479364872"
PREFIX = "amlc/out"
LOCAL = Path("cloud_out")


def s3():
    return boto3.Session(profile_name="amlc", region_name="ap-south-1").client("s3")


def listing():
    c, out = s3(), []
    for p in c.get_paginator("list_objects_v2").paginate(
            Bucket=BUCKET, Prefix=PREFIX + "/", Delimiter="/").search("CommonPrefixes"):
        if p:
            out.append(p["Prefix"].rstrip("/").split("/")[-1])
    return out


def fetch(job):
    c, dst = s3(), LOCAL / job
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for page in c.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"{PREFIX}/{job}/"):
        for o in page.get("Contents", []):
            rel = o["Key"][len(f"{PREFIX}/{job}/"):]
            if not rel:
                continue
            f = dst / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            if not f.exists() or f.stat().st_size != o["Size"]:
                c.download_file(BUCKET, o["Key"], str(f))
            n += 1
    print(f"[{job}] {n} objects -> {dst}")
    return dst


def summarise(d: Path, name: str):
    s = next(d.rglob("summary.json"), None)
    mr = next(d.rglob("matching_results.tsv"), None)
    cp = next(d.rglob("candidate_pairs.tsv"), None)
    row = {"name": name, "holdout": None, "oof": None, "india": None, "us": None,
           "singleton": None, "links": None, "valid": "no outputs", "dir": d}
    if s:
        j = json.loads(s.read_text())
        pc = j.get("per_country") or {}
        row.update(holdout=j.get("holdout_score"), oof=j.get("oof_score"),
                   india=pc.get("India"), us=pc.get("US"),
                   singleton=j.get("oof_pred_singleton_rate"),
                   links=j.get("oof_pred_links_per_entity"))
    if mr and cp:
        test_dir = Path("D:/amlc_data/dataset/test")
        r = subprocess.run([sys.executable, "validate_submission.py", "--matching", str(mr),
                            "--candidate", str(cp), "--test-dir", str(test_dir)],
                           capture_output=True, text=True)
        row["valid"] = "PASS" if r.returncode == 0 else f"FAIL {r.stdout[-120:].strip()}"
    return row


def fmt(v, w=8, p=4):
    return " " * w if v is None else f"{v:{w}.{p}f}"


def main(a):
    if a.list:
        for j in listing():
            print(" ", j)
        return
    rows = [summarise(fetch(j), j) for j in a.fetch]
    rows += [summarise(Path(p), p) for p in a.local]
    rows.sort(key=lambda r: (r["holdout"] is None, -(r["holdout"] or r["oof"] or 0)))

    print(f"\n{'run':26s} {'holdout':>8s} {'oof':>8s} {'India':>8s} {'US':>8s} "
          f"{'singl':>7s} {'links':>6s}  validator")
    print("-" * 96)
    for r in rows:
        star = "*" if r["holdout"] is not None else " "
        print(f"{r['name'][:26]:26s} {fmt(r['holdout'])} {fmt(r['oof'])} {fmt(r['india'])} "
              f"{fmt(r['us'])} {fmt(r['singleton'],7,3)} {fmt(r['links'],6,2)}  {r['valid']}{star}")
    print("\n* ranked on holdout_score. Rows without one are NOT comparable to rows "
          "with one -- OOF sits ~0.010 above the leaderboard, the holdout does not.")
    ok = [r for r in rows if r["valid"] == "PASS" and r["holdout"] is not None]
    if ok:
        print(f"\nupload: {ok[0]['name']}  (holdout {ok[0]['holdout']:.4f})  from {ok[0]['dir']}")
    else:
        print("\nnothing both validates and has a holdout score yet.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--fetch", action="append", default=[])
    ap.add_argument("--local", action="append", default=[])
    main(ap.parse_args())
