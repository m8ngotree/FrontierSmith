"""
Fast smoke-test seed downloader.

Reads the first N rows from the first HardTests parquet shard (already cached
by HuggingFace hub) using pyarrow directly — no datasets Arrow conversion,
no full dataset scan.

Writes:
  data/problems/hardtest/<pid>/statement.txt   (one dir per problem)
  data/sample_lists/smoke_seeds.json           (seed manifest for the pipeline)

Usage:
    python scripts/download_smoke_seeds.py [--limit N]

The smoke preset in synthesis/config.py points at smoke_seeds.json, so after
running this script you can immediately do:
    python -m synthesis.pipeline --preset smoke --fresh
"""

import argparse
import glob
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "data" / "problems" / "hardtest"
SMOKE_LIST   = PROJECT_ROOT / "data" / "sample_lists" / "smoke_seeds.json"

_HF_SHARD = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--sigcp--hardtests_problems"
    "/snapshots/*/data/train-00000-of-00012.parquet"
)


def main():
    parser = argparse.ArgumentParser(
        description="Download N smoke-test seed problems from the HF parquet cache."
    )
    parser.add_argument("--limit", type=int, default=20,
                        help="Number of problems to download (default: 20)")
    args = parser.parse_args()

    import pyarrow.parquet as pq

    matches = glob.glob(_HF_SHARD)
    if not matches:
        raise FileNotFoundError(
            "First parquet shard not found in HuggingFace hub cache.\n"
            "Run the full downloader first:\n"
            "  python scripts/download_hardtest.py --limit 1"
        )

    fpath = matches[0]
    print(f"Reading first {args.limit} rows from {os.path.basename(fpath)} ...")

    pf    = pq.ParquetFile(fpath)
    batch = next(pf.iter_batches(batch_size=args.limit,
                                 columns=["pid", "question_content"]))
    pids     = batch.column("pid").to_pylist()
    contents = batch.column("question_content").to_pylist()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    written = []
    for pid, content in zip(pids, contents):
        content = (content or "").strip()
        if not content:
            print(f"  SKIP (empty): {pid}")
            continue
        prob_dir = OUTPUT_DIR / pid
        prob_dir.mkdir(exist_ok=True)
        (prob_dir / "statement.txt").write_text(content, encoding="utf-8")
        written.append(pid)

    print(f"Wrote {len(written)} problems → {OUTPUT_DIR}")

    seed_list = {
        "source": "hardtest_parquet_first_shard",
        "valid_problems": [
            {"problem_id": pid, "tier": "hard", "folder_name": pid}
            for pid in written
        ],
    }
    SMOKE_LIST.parent.mkdir(parents=True, exist_ok=True)
    SMOKE_LIST.write_text(json.dumps(seed_list, indent=2))
    print(f"Wrote seed manifest → {SMOKE_LIST}")


if __name__ == "__main__":
    main()
