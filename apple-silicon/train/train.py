"""Train the text-conditioned edit-region head.

    python train/train.py --epochs 8 --batch 8 --limit 2000

Frozen: FastSAM backbone + prototypes (71.75M), CLIP text encoder.
Trainable: 1.4M fusion head.

Baselines it must beat, both reported at the end:
  no-text  - the same head with the instruction embedding zeroed, which
             isolates whether language does any work at all
  mean     - predict the training-set mean mask everywhere
"""
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import warnings; warnings.filterwarnings("ignore")

import numpy as np, torch
from torch.utils.data import DataLoader, random_split
from src.device import pick_device
from dataset import MagicBrushEdits, collate
from model import EditRegionModel, dice_bce_loss, mask_metrics


def evaluate(model, loader, dev, zero_text=False):
    model.head.eval()
    agg = dict(iou=[], dice=[], pred=[], true=[]); byk = {}
    with torch.no_grad():
        for b in loader:
            img = b["image"].to(dev); tgt = b["mask"].to(dev)
            proto, ctx = model.backbone(img)
            temb = model.encode_text(b["text"])
            if zero_text: temb = torch.zeros_like(temb)
            coef, bias = model.head(temb, ctx)
            lg = (torch.einsum("bc,bchw->bhw", coef, proto) + bias[:, :, None]).unsqueeze(1)
            m = mask_metrics(lg, tgt)
            agg["iou"].append(m["iou"]); agg["dice"].append(m["dice"])
            agg["pred"].append(m["pred_frac"]); agg["true"].append(m["true_frac"])
            for k in set(b["kind"]):
                idx = [i for i, kk in enumerate(b["kind"]) if kk == k]
                mk = mask_metrics(lg[idx], tgt[idx])
                byk.setdefault(k, []).append(mk["iou"])
    model.head.train()
    return (dict(iou=float(np.mean(agg["iou"])), dice=float(np.mean(agg["dice"])),
                 pred_frac=float(np.mean(agg["pred"])), true_frac=float(np.mean(agg["true"]))),
            {k: float(np.mean(v)) for k, v in byk.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.15)
    a = ap.parse_args()

    dev = pick_device("auto")
    print(f"device {dev}")
    ds = MagicBrushEdits("train", limit=a.limit)
    n_val = max(1, int(len(ds)*a.val_frac)); n_tr = len(ds)-n_val
    tr, va = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(1368))
    print(f"train {n_tr} / val {n_val}")
    ltr = DataLoader(tr, batch_size=a.batch, shuffle=True, collate_fn=collate, num_workers=0)
    lva = DataLoader(va, batch_size=a.batch, shuffle=False, collate_fn=collate, num_workers=0)

    model = EditRegionModel("best.pt", device=dev)
    ntr = sum(p.numel() for p in model.trainable_parameters())
    print(f"trainable {ntr:,} | frozen backbone 71.75M + CLIP\n")
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs*max(1,len(ltr)))

    hist, best = [], -1.0
    for ep in range(1, a.epochs+1):
        t0 = time.time(); tot = n = 0
        for b in ltr:
            img = b["image"].to(dev); tgt = b["mask"].to(dev)
            lg = model(img, b["text"])
            loss, bce, dice = dice_bce_loss(lg, tgt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(loss); n += 1
        v, byk = evaluate(model, lva, dev)
        hist.append(dict(epoch=ep, loss=tot/max(n,1), **v, by_kind=byk))
        print(f"  ep{ep:>2} loss {tot/max(n,1):.4f} | val IoU {v['iou']:.4f} "
              f"Dice {v['dice']:.4f} | pred {v['pred_frac']:.1%} true {v['true_frac']:.1%} "
              f"| {' '.join(f'{k}:{x:.3f}' for k,x in sorted(byk.items()))} "
              f"| {time.time()-t0:.0f}s", flush=True)
        if v["iou"] > best:
            best = v["iou"]
            torch.save(dict(head=model.head.state_dict(), epoch=ep, iou=best),
                       HERE/"edit_region_head.pt")

    print("\n--- ablations ---")
    zt, _ = evaluate(model, lva, dev, zero_text=True)
    print(f"  no-text (instruction zeroed): IoU {zt['iou']:.4f}  Dice {zt['dice']:.4f}")
    print(f"  trained                     : IoU {best:.4f}")
    delta = best - zt["iou"]
    print(f"  language contributes        : {delta:+.4f} IoU")
    if delta < 0.01:
        print("  WARNING: the instruction is contributing almost nothing.")
    (HERE/"train_log.json").write_text(json.dumps(
        dict(history=hist, best_iou=best, no_text_iou=zt["iou"], args=vars(a)), indent=2))
    print(f"\nsaved {HERE/'edit_region_head.pt'}")


if __name__ == "__main__":
    main()
