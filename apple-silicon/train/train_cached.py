"""Train the 1.4M head on cached frozen features.

Everything expensive was done once by precompute.py, so an epoch here is a few
seconds on the M5 and we can afford large batches and many epochs.

    python train/precompute.py --split train
    python train/train_cached.py --epochs 60 --batch 64
"""
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import warnings; warnings.filterwarnings("ignore")

import numpy as np, torch
from src.device import pick_device
from model import TextCoeffHead, dice_bce_loss, mask_metrics


def batches(n, bs, shuffle, gen=None):
    idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i+bs]


def run_eval(head, D, idx, dev, zero_text=False, bs=64):
    head.eval(); ious=[]; dices=[]; pf=[]; tf=[]; byk={}
    with torch.no_grad():
        for sl in batches(len(idx), bs, False):
            j = idx[sl]
            proto = D["proto"][j].to(dev).float()
            ctx   = D["ctx"][j].to(dev).float()
            temb  = D["text"][j].to(dev).float()
            if zero_text: temb = torch.zeros_like(temb)
            tgt   = D["mask"][j].to(dev).float()
            coef, bias = head(temb, ctx)
            lg = (torch.einsum("bc,bchw->bhw", coef, proto) + bias[:, :, None]).unsqueeze(1)
            m = mask_metrics(lg, tgt)
            ious.append(m["iou"]); dices.append(m["dice"])
            pf.append(m["pred_frac"]); tf.append(m["true_frac"])
            kinds = [D["kind"][k] for k in j.tolist()]
            for kk in set(kinds):
                sel = [t for t,x in enumerate(kinds) if x==kk]
                byk.setdefault(kk, []).append(mask_metrics(lg[sel], tgt[sel])["iou"])
    head.train()
    return (dict(iou=float(np.mean(ious)), dice=float(np.mean(dices)),
                 pred_frac=float(np.mean(pf)), true_frac=float(np.mean(tf))),
            {k: float(np.mean(v)) for k,v in byk.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--wdice", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1368)
    a = ap.parse_args()

    # Seed EVERYTHING. Seeding only the data permutation left weight init
    # unseeded, and two identical commands gave IoU 0.1215 and 0.1073.
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    import random as _r; _r.seed(a.seed)
    if torch.backends.mps.is_available(): torch.mps.manual_seed(a.seed)

    dev = pick_device("auto")
    D = torch.load(HERE/"cache_train.pt", map_location="cpu")
    n = D["proto"].shape[0]
    g = torch.Generator().manual_seed(a.seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(8, int(n*a.val_frac))
    va_i, tr_i = perm[:n_val], perm[n_val:]
    print(f"device {dev} | cached {n} samples -> train {len(tr_i)} val {len(va_i)}")
    from collections import Counter
    print("  kinds:", dict(Counter(D["kind"])))

    head = TextCoeffHead(hidden=a.hidden, dropout=a.dropout).to(dev)
    print(f"  trainable {sum(p.numel() for p in head.parameters()):,}\n")
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    hist=[]; best=-1.0; best_delta=-9.0; best_ep=0; best_state=None; t0=time.time()
    for ep in range(1, a.epochs+1):
        tot=0.0; nb=0
        for sl in batches(len(tr_i), a.batch, True, g):
            j = tr_i[sl]
            proto = D["proto"][j].to(dev).float()
            ctx   = D["ctx"][j].to(dev).float()
            temb  = D["text"][j].to(dev).float()
            tgt   = D["mask"][j].to(dev).float()
            coef, bias = head(temb, ctx)
            lg = (torch.einsum("bc,bchw->bhw", coef, proto) + bias[:, :, None]).unsqueeze(1)
            loss,_,_ = dice_bce_loss(lg, tgt, w_dice=a.wdice)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step(); tot += float(loss); nb += 1
        sch.step()
        v, byk = run_eval(head, D, va_i, dev)
        # Select on DELTA, not IoU. Selecting on IoU picks epoch-1 blob
        # predictors: they score ~0.14 by covering 20% of the frame and score
        # exactly the same with the instruction zeroed, i.e. they have learned
        # nothing about language. Delta is the quantity we actually care about.
        nt, _ = run_eval(head, D, va_i, dev, zero_text=True)
        v["no_text"] = nt["iou"]; v["delta"] = v["iou"] - nt["iou"]
        hist.append(dict(epoch=ep, loss=tot/max(nb,1), **v, by_kind=byk))
        if ep % 5 == 0 or ep == 1 or ep == a.epochs:
            print(f"  ep{ep:>3} loss {tot/max(nb,1):.4f} | IoU {v['iou']:.4f} notext {v['no_text']:.4f} "
                  f"delta {v['delta']:+.4f} | pred {v['pred_frac']:.1%} true {v['true_frac']:.1%} "
                  f"| {' '.join(f'{k}:{x:.3f}' for k,x in sorted(byk.items()))}", flush=True)
        if v["delta"] > best_delta:
            best_delta = v["delta"]; best = v["iou"]; best_ep = ep
            best_state = {k: t.clone() for k, t in head.state_dict().items()}
            torch.save(dict(head=head.state_dict(), epoch=ep, iou=best,
                            delta=best_delta, cfg=vars(a)), HERE/"edit_region_head.pt")

    if best_state is not None: head.load_state_dict(best_state)
    zt,_ = run_eval(head, D, va_i, dev, zero_text=True)
    fin, fbyk = run_eval(head, D, va_i, dev)
    delta = best - zt["iou"]
    print(f"\n--- ablation ---")
    print(f"  trained (best-delta ep{best_ep})   IoU {best:.4f}")
    print(f"  no-text (instruction zeroed) IoU {zt['iou']:.4f}")
    print(f"  language contributes        {delta:+.4f}")
    print(f"  {'PASS' if delta>=0.01 else 'FAIL — instruction contributes almost nothing'}")
    print(f"  by kind: {' '.join(f'{k}:{v:.4f}' for k,v in sorted(fbyk.items()))}")
    print(f"  total {(time.time()-t0)/60:.1f} min")
    (HERE/"train_log.json").write_text(json.dumps(
        dict(history=hist, best_iou=best, no_text_iou=zt["iou"],
             delta=delta, n=n, args=vars(a)), indent=2))


if __name__ == "__main__":
    main()
