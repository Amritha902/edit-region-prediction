"""Cache frozen features for the full 51-shard MagicBrush train split.

Same computation as precompute.py, streamed into memmaps instead of accumulated
in a list, so peak RSS is a batch rather than the whole 11.8 GB. Writes progress
after every batch and can be resumed if the run is interrupted.

    python train/precompute_full.py                # start or resume
    python train/precompute_full.py --restart      # discard and start over
"""
import argparse, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import warnings; warnings.filterwarnings("ignore")

import numpy as np, torch
from src.device import pick_device
from dataset import MagicBrushEdits
from model import EditRegionModel
from cachemm import FIELDS, open_maps, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    out = Path(a.out or HERE / f"cache_full_{a.split}")
    out.mkdir(parents=True, exist_ok=True)
    meta_p = out / "meta.json"

    dev = pick_device("auto")
    ds = MagicBrushEdits(a.split, limit=a.limit, cache=False)
    alloc = len(ds)

    start, kept, kinds = 0, 0, []
    if meta_p.exists() and not a.restart:
        m = json.loads(meta_p.read_text())
        if m.get("alloc") == alloc and not m.get("complete"):
            start, kept, kinds = m["rows_done"], m["n"], m["kind"]
            print(f"resuming at row {start}/{alloc} ({kept} kept)")
        elif m.get("complete"):
            print(f"already complete: {m['n']} samples. --restart to redo.")
            return

    mode = "r+" if start else "w+"
    mm = open_maps(out, alloc, mode=mode)
    model = EditRegionModel("best.pt", device=dev)
    print(f"device {dev} | {alloc} rows in {a.split} | out {out}", flush=True)

    t0 = time.time()
    for s in range(start, alloc, a.batch):
        items = [ds[i] for i in range(s, min(s + a.batch, alloc))]
        use = [it for it in items if it["usable"]]
        if use:
            img = torch.stack([it["image"] for it in use]).to(dev)
            with torch.no_grad():
                proto, ctx = model.backbone(img)
                temb = model.encode_text([it["text"] for it in use])
            k = len(use)
            mm["proto"][kept:kept+k] = proto.to(torch.float16).cpu().numpy()
            mm["ctx"][kept:kept+k]   = ctx.to(torch.float16).cpu().numpy()
            mm["text"][kept:kept+k]  = temb.to(torch.float16).cpu().numpy()
            mm["mask"][kept:kept+k]  = torch.stack(
                [it["mask"] for it in use]).to(torch.uint8).numpy()
            kinds += [it["kind"] for it in use]
            kept += k
        done = min(s + a.batch, alloc)
        meta_p.write_text(json.dumps(
            dict(alloc=alloc, rows_done=done, n=kept, kind=kinds, complete=False)))
        if (s // a.batch) % 25 == 0 and s > start:
            el = time.time() - t0
            rate = (done - start) / el
            print(f"  {done}/{alloc}  kept {kept}  {el/60:.1f} min  "
                  f"eta {(alloc-done)/rate/60:.1f} min", flush=True)

    for v in mm.values():
        v.flush()
    del mm

    # Trim the tail we allocated but did not fill. The layout is C-ordered, so
    # the first `kept` rows are exactly the first kept*itemsize bytes.
    for k, (shape, dt) in FIELDS.items():
        item = int(np.prod(shape)) * np.dtype(dt).itemsize
        os.truncate(paths(out)[k], kept * item)

    meta_p.write_text(json.dumps(
        dict(alloc=kept, rows_done=alloc, n=kept, kind=kinds, complete=True)))
    from collections import Counter
    gb = sum(int(np.prod(s)) * np.dtype(d).itemsize for s, d in FIELDS.values()) * kept / 1e9
    print(f"\nkept {kept} of {alloc} ({alloc-kept} degenerate)")
    print(f"  kinds {dict(Counter(kinds))}")
    print(f"  wrote {out}  ({gb:.1f} GB)  in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
