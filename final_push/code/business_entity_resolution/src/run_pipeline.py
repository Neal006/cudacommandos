"""End-to-end orchestrator (CPU side).

  python run_pipeline.py                     # everything: data -> blocking -> models -> output
  python run_pipeline.py --from prune        # resume from a stage (earlier stages' parquet files reused)
  python run_pipeline.py --only output --params my.json --tag probe1

Stages: io, splits, normalize, sets, blocking, prune, stage1, context, output
Cross-encoder (GPU) steps are separate; see README.md. When work/ce_train.parquet and
work/ce_test.parquet exist, the context stage uses them automatically.
"""
import argparse
import time

STAGES = ["io", "splits", "normalize", "sets", "blocking", "prune", "stage1", "context", "output"]


def run(stage, args):
    if stage == "io":
        import io_utils
        io_utils.main()
    elif stage == "splits":
        import splits
        splits.main()
    elif stage == "normalize":
        import normalize
        normalize.main()
    elif stage == "sets":
        import features
        features.build_sets()
    elif stage == "blocking":
        import blocking
        blocking.block_split("train")
        blocking.recall_report(__import__("polars").read_parquet(blocking.wpath("cands_train.parquet")))
        blocking.block_split("test")
    elif stage == "prune":
        import prune
        prune.main()
    elif stage == "stage1":
        import stage1
        stage1.main()
    elif stage == "context":
        import context
        context.main()
    elif stage == "output":
        import write_output
        extra = (["--params", args.params] if args.params else []) + (["--tag", args.tag] if args.tag else [])
        rc = write_output.main(extra)
        if rc:
            raise SystemExit(f"validator failed ({rc})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=STAGES[0], choices=STAGES)
    ap.add_argument("--to", dest="end", default=STAGES[-1], choices=STAGES)
    ap.add_argument("--only", default=None, choices=STAGES)
    ap.add_argument("--params", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    todo = [args.only] if args.only else STAGES[STAGES.index(args.start):STAGES.index(args.end) + 1]
    for st in todo:
        t0 = time.time()
        print(f"===== {st} =====", flush=True)
        run(st, args)
        print(f"===== {st} done in {time.time() - t0:.0f}s =====", flush=True)


if __name__ == "__main__":
    main()
