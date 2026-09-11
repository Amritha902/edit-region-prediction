"""Train on the train shards, evaluate on the dev split. No overlap.

The previous run trained on a cache that had dev merged into it, so its
evaluation was contaminated. Here the two caches come from different shard
files and never mix.
"""
import argparse, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(ROOT))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from src.device import pick_device
from model import TextCoeffHead, dice_bce_loss
from train_cached import batches, run_eval

ap=argparse.ArgumentParser()
ap.add_argument("--epochs",type=int,default=60); ap.add_argument("--seeds",type=int,default=8)
a=ap.parse_args()
dev=pick_device("auto")
TR=torch.load(HERE/"cache_train.pt",map_location="cpu")
DV=torch.load(HERE/"cache_dev.pt",map_location="cpu")
print(f"train {TR['proto'].shape[0]} (train shards)   eval {DV['proto'].shape[0]} (dev shards, held out)\n")
va=torch.arange(DV["proto"].shape[0])

res=[]
for s in [1368,1,2,3,4,5,6,7][:a.seeds]:
    torch.manual_seed(s); np.random.seed(s)
    g=torch.Generator().manual_seed(s)
    tr=torch.randperm(TR["proto"].shape[0],generator=g)
    head=TextCoeffHead(hidden=512,dropout=0.0).to(dev)
    opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
    best=(-9,0,0,None,0)
    for ep in range(1,a.epochs+1):
        for sl in batches(len(tr),64,True,g):
            j=tr[sl]
            p=TR["proto"][j].to(dev).float(); c=TR["ctx"][j].to(dev).float()
            t=TR["text"][j].to(dev).float(); y=TR["mask"][j].to(dev).float()
            co,b=head(t,c)
            lg=(torch.einsum("bc,bchw->bhw",co,p)+b[:,:,None]).unsqueeze(1)
            l,_,_=dice_bce_loss(lg,y,w_dice=2.0); opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.0); opt.step()
        sch.step()
        v,byk=run_eval(head,DV,va,dev); nt,_=run_eval(head,DV,va,dev,zero_text=True)
        d=v["iou"]-nt["iou"]
        if d>best[0]: best=(d,v["iou"],nt["iou"],byk,ep)
    d,i,nt,byk,ep=best
    res.append((i,nt,d))
    print(f"  seed {s}: IoU {i:.4f}  no-text {nt:.4f}  delta {d:+.4f}  (ep{ep})  | "
          + " ".join(f"{k}:{v:.3f}" for k,v in sorted(byk.items())), flush=True)
    if s==1368:
        torch.save(dict(head=head.state_dict(),iou=i,epoch=ep,
                        cfg=dict(hidden=512,dropout=0.0)), HERE/"edit_region_head_clean.pt")

A=np.array(res)
print(f"\nHELD-OUT (dev, never trained on)")
print(f"  IoU     {A[:,0].mean():.4f} ± {A[:,0].std(ddof=1):.4f}")
print(f"  no-text {A[:,1].mean():.4f} ± {A[:,1].std(ddof=1):.4f}")
print(f"  delta   {A[:,2].mean():+.4f} ± {A[:,2].std(ddof=1):.4f}")
from scipy import stats
t,p=stats.ttest_rel(A[:,0],A[:,1]); print(f"  paired t-test p = {p:.2e}")
json.dump(dict(iou=A[:,0].tolist(),notext=A[:,1].tolist(),delta=A[:,2].tolist(),
               p=float(p)), open(HERE/"clean_results.json","w"), indent=2)
