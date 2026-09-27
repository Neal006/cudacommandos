"""Leaderboard probes (plan §H.2): variants that change ONLY one country's decision rule.

The leaderboard is a mean over S1 entities, so the score difference between a probe and the baseline
submission is caused entirely by that country's rows. France has no labels, so this is the only way
to tune its decision parameters.

  python make_probes.py France            -> output/probes/matching_results_France_<a>_<b>.tsv + params json
"""
import json
import sys

from config import OUT
from decide import load_params
import write_output

GRID = [(1.0, -1.0), (1.0, -0.5), (1.0, 0.5), (1.0, 1.0), (1.25, 0.0), (0.8, 0.0)]


def main(country="France"):
    base = load_params()
    probe_dir = OUT / "probes"
    probe_dir.mkdir(exist_ok=True)
    for a, b in GRID:
        prm = json.loads(json.dumps(base))
        prm[country] = {"a": a, "b": b}
        tag = f"{country}_a{a}_b{b}"
        path = probe_dir / f"params_{tag}.json"
        path.write_text(json.dumps(prm, indent=2))
        write_output.main(["--params", str(path), "--tag", tag])
        print(f"probe {tag}: submit output/matching_results_{tag}.tsv; compare with the baseline LB score")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["France"]))
