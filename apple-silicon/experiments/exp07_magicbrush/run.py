"""Experiment 07 - how coarse is the supervision the base paper trains on?

AdaptEdit's MaskPredictor is supervised on MagicBrush human-annotated edit
regions. This compares every one of those annotations against the region that
actually changed between source and target, recovered by CIE76 deltaE.

Note on polarity: MagicBrush masks are bright = PRESERVE. The edit region is
the inverse. Getting this backwards inverts every number - the first run of
this script reported recall 0.3% before the convention was checked.

    python run.py
"""
import glob, io, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT/"datagen"))

import pandas as pd, numpy as np, cv2
from PIL import Image
import mask_extraction as ME

SHARDS = str(ROOT/"data/magicbrush/data/dev-*.parquet")

def kind_of(ins):
    ins = ins.strip().lower()
    if ins.startswith(("put ", "add ")):      return "insert"
    if ins.startswith(("remove", "delete")):  return "remove"
    return "modify"

def main():
    rows = []
    for f in sorted(glob.glob(SHARDS)):
        df = pd.read_parquet(f)
        for i in range(len(df)):
            src = Image.open(io.BytesIO(df.source_img.iloc[i]["bytes"])).convert("RGB")
            tgt = Image.open(io.BytesIO(df.target_img.iloc[i]["bytes"])).convert("RGB").resize(src.size)
            mk  = ~(np.array(Image.open(io.BytesIO(df.mask_img.iloc[i]["bytes"]))
                    .convert("L").resize(src.size, Image.NEAREST)) > 127)
            o = cv2.cvtColor(np.array(src), cv2.COLOR_RGB2BGR)
            e = cv2.cvtColor(np.array(tgt), cv2.COLOR_RGB2BGR)
            ch = ME.clean_mask(((ME.delta_e(o, e) > ME.DELTA_E_THRESHOLD)*255).astype(np.uint8)) > 0
            if mk.sum() == 0 or ch.sum() == 0: continue
            inter = (mk & ch).sum()
            rows.append(dict(kind=kind_of(df.instruction.iloc[i]),
                             instruction=df.instruction.iloc[i],
                             mask=float(mk.mean()), changed=float(ch.mean()),
                             recall=float(inter/ch.sum()), precision=float(inter/mk.sum()),
                             iou=float(inter/(mk|ch).sum()),
                             ratio=float(mk.mean()/max(ch.mean(), 1e-6))))
    d = pd.DataFrame(rows)
    print(f"n = {len(d)}  MagicBrush dev turns\n")
    hdr = f"{'kind':8s} {'n':>4} {'human mask':>11} {'real change':>12} {'ratio':>7} {'recall':>7} {'prec':>6} {'IoU':>6}"
    print(hdr)
    for k, g in d.groupby("kind"):
        print(f"{k:8s} {len(g):>4} {g['mask'].mean():>10.1%} {g['changed'].mean():>11.1%} "
              f"{g['ratio'].median():>6.1f}x {g['recall'].mean():>6.1%} "
              f"{g['precision'].mean():>5.1%} {g['iou'].mean():>5.2f}")
    print(f"{'ALL':8s} {len(d):>4} {d['mask'].mean():>10.1%} {d['changed'].mean():>11.1%} "
          f"{d['ratio'].median():>6.1f}x {d['recall'].mean():>6.1%} "
          f"{d['precision'].mean():>5.1%} {d['iou'].mean():>5.2f}")
    d.to_csv(Path(__file__).parent/"magicbrush_mask_precision.csv", index=False)

if __name__ == "__main__":
    main()
