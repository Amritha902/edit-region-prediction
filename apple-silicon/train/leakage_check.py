"""Does any dev source image also appear in the 51 train shards?

At 14 shards this was never checked. MagicBrush is multi-turn -- one source
image spawns several instructions -- so if the official split partitions by turn
rather than by image, scaling the train set 3.9x could pull dev images into
training and inflate the dev IoU we report. Hash the raw encoded bytes of every
source and target image on both sides and intersect.
"""
import glob, hashlib, json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def hashes(pat):
    h = {}
    for f in sorted(glob.glob(str(ROOT / "data/magicbrush/data" / pat))):
        df = pd.read_parquet(f, columns=["source_img", "target_img"])
        for i in range(len(df)):
            for col in ("source_img", "target_img"):
                d = hashlib.sha1(df[col].iloc[i]["bytes"]).hexdigest()
                h.setdefault(d, []).append((Path(f).name, i, col))
        print(f"  {Path(f).name}: {len(df)} rows, {len(h)} distinct so far", flush=True)
    return h


if __name__ == "__main__":
    print("dev:"); dv = hashes("dev-*.parquet")
    print("train:"); tr = hashes("train-*.parquet")
    both = set(dv) & set(tr)
    print(f"\ndev distinct images   {len(dv)}")
    print(f"train distinct images {len(tr)}")
    print(f"overlapping images    {len(both)}")
    if both:
        print("  LEAKAGE — examples:")
        for d in list(both)[:5]:
            print(f"    {d[:12]}  dev {dv[d][0]}  train {tr[d][0]}")
    else:
        print("  CLEAN — the splits share no image.")
    json.dump(dict(n_dev=len(dv), n_train=len(tr), n_overlap=len(both),
                   overlap=sorted(both)[:100]),
              open(Path(__file__).parent / "leakage_check.json", "w"), indent=2)
