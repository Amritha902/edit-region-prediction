"""Cache the frozen features once, so training touches only the 1.4M head.

The backbone and CLIP encoder never change, so re-running them every epoch is
pure waste: measured 130 ms/image on the M5, which is ~52 min across 10 epochs
for 2400 samples. Precomputing costs one pass (~5 min) and makes each epoch a
matter of seconds.

Cached per sample: prototypes (32,160,160) fp16 = 1.6 MB, image context 640-d,
CLIP text embedding 512-d, and the ΔE target mask (160,160) as bits.

    python train/precompute.py --split train --limit 3000
"""
import argparse, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import warnings; warnings.filterwarnings("ignore")

import numpy as np, torch
from src.device import pick_device
from dataset import MagicBrushEdits
from model import EditRegionModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    dev = pick_device("auto")
    ds = MagicBrushEdits(a.split, limit=a.limit, cache=False)
    print(f"device {dev} | {len(ds)} rows in {a.split}")
    model = EditRegionModel("best.pt", device=dev)

    P, C, T, M, K = [], [], [], [], []
    t0 = time.time(); kept = skipped = 0
    for s in range(0, len(ds), a.batch):
        items = [ds[i] for i in range(s, min(s+a.batch, len(ds)))]
        use = [it for it in items if it["usable"]]
        skipped += len(items) - len(use)
        if not use: continue
        img = torch.stack([it["image"] for it in use]).to(dev)
        with torch.no_grad():
            proto, ctx = model.backbone(img)
            temb = model.encode_text([it["text"] for it in use])
        P.append(proto.to(torch.float16).cpu())
        C.append(ctx.to(torch.float16).cpu())
        T.append(temb.to(torch.float16).cpu())
        M.append(torch.stack([it["mask"] for it in use]).to(torch.uint8))
        K += [it["kind"] for it in use]
        kept += len(use)
        if (s // a.batch) % 20 == 0 and s:
            el = time.time()-t0
            print(f"  {s}/{len(ds)}  kept {kept} skipped {skipped}  "
                  f"{el:.0f}s  eta {(len(ds)-s)*el/max(s,1)/60:.1f} min", flush=True)

    out = dict(proto=torch.cat(P), ctx=torch.cat(C), text=torch.cat(T),
               mask=torch.cat(M), kind=K)
    path = HERE/f"cache_{a.split}.pt"
    torch.save(out, path)
    gb = sum(v.numel()*v.element_size() for v in out.values() if torch.is_tensor(v))/1e9
    print(f"\nkept {kept}, skipped {skipped} degenerate")
    print(f"wrote {path}  ({gb:.2f} GB)  in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
