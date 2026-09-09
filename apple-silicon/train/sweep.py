"""Hyperparameter sweep. Each run is seconds on cached features, so search properly.

Also runs the no-text ablation for every configuration - a config that scores
well only because it ignores the instruction is worse than useless here.
"""
import itertools, json, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(ROOT))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from src.device import pick_device
from model import TextCoeffHead, dice_bce_loss, mask_metrics
from train_cached import batches, run_eval

def train_one(D, tr_i, va_i, dev, lr, hidden, dropout, wd, epochs, wdice, seed=1368):
    torch.manual_seed(seed)
    head=TextCoeffHead(hidden=hidden, dropout=dropout).to(dev)
    opt=torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g=torch.Generator().manual_seed(seed)
    best=-1; best_ep=0; best_state=None
    for ep in range(1,epochs+1):
        for sl in batches(len(tr_i),64,True,g):
            j=tr_i[sl]
            proto=D["proto"][j].to(dev).float(); ctx=D["ctx"][j].to(dev).float()
            temb=D["text"][j].to(dev).float();  tgt=D["mask"][j].to(dev).float()
            c,b=head(temb,ctx)
            lg=(torch.einsum("bc,bchw->bhw",c,proto)+b[:,:,None]).unsqueeze(1)
            loss,_,_=dice_bce_loss(lg,tgt,w_dice=wdice)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.0); opt.step()
        sch.step()
        v,_=run_eval(head,D,va_i,dev)
        if v["iou"]>best: best, best_ep = v["iou"], ep; best_state={k:t.clone() for k,t in head.state_dict().items()}
    head.load_state_dict(best_state)
    zt,_=run_eval(head,D,va_i,dev,zero_text=True)
    byk=run_eval(head,D,va_i,dev)[1]
    return best, best_ep, zt["iou"], byk, head

def main():
    dev=pick_device("auto")
    D=torch.load(HERE/"cache_train.pt",map_location="cpu")
    n=D["proto"].shape[0]; g=torch.Generator().manual_seed(1368)
    perm=torch.randperm(n,generator=g); nv=max(8,int(n*0.15))
    va_i,tr_i=perm[:nv],perm[nv:]
    print(f"{n} samples -> train {len(tr_i)} val {len(va_i)}\n")
    grid=dict(lr=[3e-4,1e-3,3e-3], hidden=[256,512,1024],
              dropout=[0.0,0.1,0.3], wd=[0.01,0.1], wdice=[1.0,2.0])
    keys=list(grid); combos=list(itertools.product(*[grid[k] for k in keys]))
    print(f"{len(combos)} configurations\n")
    print(f"{'lr':>7} {'hid':>5} {'drop':>5} {'wd':>5} {'wdice':>6} {'IoU':>7} {'ep':>4} {'notext':>7} {'delta':>7}")
    res=[]; t0=time.time()
    for i,c in enumerate(combos,1):
        cfg=dict(zip(keys,c))
        iou,ep,nt,byk,_=train_one(D,tr_i,va_i,dev,epochs=40,**cfg)
        res.append(dict(**cfg,iou=iou,best_epoch=ep,no_text=nt,delta=iou-nt,by_kind=byk))
        print(f"{cfg['lr']:>7.0e} {cfg['hidden']:>5} {cfg['dropout']:>5.1f} {cfg['wd']:>5.2f} "
              f"{cfg['wdice']:>6.1f} {iou:>7.4f} {ep:>4} {nt:>7.4f} {iou-nt:>+7.4f}", flush=True)
    res.sort(key=lambda r:-r["iou"])
    print(f"\n--- top 5 of {len(res)}  ({(time.time()-t0)/60:.1f} min) ---")
    for r in res[:5]:
        print(f"  IoU {r['iou']:.4f} (ep{r['best_epoch']}) delta {r['delta']:+.4f} | "
              f"lr {r['lr']:.0e} hid {r['hidden']} drop {r['dropout']} wd {r['wd']} wdice {r['wdice']} | "
              f"{' '.join(f'{k}:{v:.3f}' for k,v in sorted(r['by_kind'].items()))}")
    (HERE/"sweep_results.json").write_text(json.dumps(res,indent=2))
    print(f"\nbest: IoU {res[0]['iou']:.4f}  (baseline run was 0.1207)")

if __name__=="__main__": main()
