#!/usr/bin/env python3
"""Download the PixelProse RedCaps caption shards that the pipeline reads.

These parquet files hold image URLs plus the original and VLM captions. They do
NOT contain images -- generate_prompts.ipynb fetches those from the URLs at run
time.

PixelProse is gated: accept the terms once at
https://huggingface.co/datasets/tomg-group-umd/pixelprose
then `huggingface-cli login`, or pass --token / set HF_TOKEN.

    python scripts/download_data.py --shards 0 1
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "tomg-group-umd/pixelprose"
FILENAME = "vlm_captions_redcaps_{:02d}.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", type=int, nargs="+", default=[0, 1],
                        help="shard indices to fetch (default: 0 1)")
    parser.add_argument("--out", type=Path, default=Path("data"),
                        help="destination directory (default: data)")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN") or None,
                        help="HF token; defaults to $HF_TOKEN or a cached login")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import GatedRepoError, EntryNotFoundError
    except ImportError:
        print("huggingface-hub is not installed.  pip install huggingface-hub", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    # The pipeline also writes into these; create them now so a fresh clone is runnable.
    for sub in ("images", "prompts", "results", "masks"):
        (args.out / sub).mkdir(exist_ok=True)

    failed = []
    for shard in args.shards:
        name = FILENAME.format(shard)
        target = args.out / name
        if target.exists():
            print(f"{name}: already present, skipping")
            continue

        print(f"{name}: downloading...")
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=name,
                repo_type="dataset",
                local_dir=args.out,
                token=args.token,
            )
            size_mb = Path(path).stat().st_size / 1e6
            print(f"{name}: done ({size_mb:.0f} MB)")
        except GatedRepoError:
            print(f"{name}: access denied. Accept the dataset terms at\n"
                  f"  https://huggingface.co/datasets/{REPO_ID}\n"
                  f"then run `huggingface-cli login` (or pass --token).", file=sys.stderr)
            return 1
        except EntryNotFoundError:
            print(f"{name}: not found in the repo -- check the shard index.", file=sys.stderr)
            failed.append(name)
        except Exception as exc:
            print(f"{name}: failed -- {exc}", file=sys.stderr)
            failed.append(name)

    if failed:
        print(f"\n{len(failed)} shard(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1

    print(f"\nReady. Data directories created under {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
