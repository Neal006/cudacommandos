"""Rebuild <TEAM>_submission.zip. Python port of make_submission_zip.sh, for boxes
with neither rsync nor zip. Same structure, same members, same exclusions.

The zip itself is not in git (117 MiB, over GitHub's 100 MiB limit). Only the two
gzipped outputs are tracked; this script decompresses them as needed and rebuilds.

  python make_submission_zip.py . CUDA_COMMANDOS      # run from final_push/
"""
import gzip
import shutil
import sys
import zipfile
from pathlib import Path


def ensure_tsv(p: Path) -> Path:
    """Tracked form is the .gz; decompress once if the plain .tsv is absent."""
    if p.exists():
        return p
    gz = p.with_suffix(".tsv.gz")
    if not gz.exists():
        raise SystemExit(f"missing both {p} and {gz}")
    print(f"decompressing {gz.name} -> {p.name}")
    with gzip.open(gz, "rb") as fi, open(p, "wb") as fo:
        shutil.copyfileobj(fi, fo, 1 << 22)
    return p


def main(root: Path, team: str):
    out = root / f"{team}_submission.zip"
    members = []

    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        members.append((ensure_tsv(root / "output" / name), f"output/{name}"))

    # tool caches and editor droppings must never reach the package
    skip_dirs = {"__pycache__", ".ruff_cache", ".mypy_cache", ".pytest_cache", ".ipynb_checkpoints",
                 ".git", ".venv", ".idea", ".vscode"}
    skip_suffix = {".pyc", ".pyo", ".log", ".tmp", ".swp"}

    code = root / "code" / "business_entity_resolution"
    for p in sorted(code.rglob("*")):
        if not p.is_file() or skip_dirs & set(p.parts) or p.suffix in skip_suffix or p.name == ".DS_Store":
            continue
        members.append((p, f"code/business_entity_resolution/{p.relative_to(code).as_posix()}"))

    doc = root / "Documentation_template.md"
    members.append((doc, "Documentation_template.md"))

    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for src, arc in members:
            z.write(src, arc)

    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB, {len(members)} files)")
    with zipfile.ZipFile(out) as z:
        for i in sorted(z.namelist()):
            print(f"  {z.getinfo(i).file_size:>12,}  {i}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
