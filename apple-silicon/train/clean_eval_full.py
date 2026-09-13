"""Clean-split evaluation on the full 51-shard train set.

Identical protocol to clean_eval.py -- epoch chosen on a held-out slice of
TRAIN, dev touched exactly once per seed -- with one change: the training
features come from the memmapped 51-shard cache (~7.2k samples) instead of the
2,278-sample .pt. The dev set and the ground-truth construction are byte-for-byte
the same as before, so the resulting IoU is directly comparable to the 0.1718
reported at 2,278.
"""
import argparse, sys, json, glob, io, time
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"datagen"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, cv2, pandas as pd
from PIL import Image
from src.device import pick_device
from model import TextCoeffHead, dice_bce_loss
from model_v2 import SpatialCoeffHead
from train_cached import batches
from cachemm import MMCache
import mask_extraction as ME

ap=argparse.ArgumentParser()
ap.add_argument("--cache", default=str(HERE/"cache_full_train"))
ap.add_argument("--seeds", type=int, default=12)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--configs", default="v1,v2")
ap.add_argument("--out", default=str(HERE/"clean_eval_full.json"))
ap.add_argument("--curves", action="store_true",
                help="also log the zeroed-instruction control each epoch")
A=ap.parse_args()

dev=pick_device("auto")
TR=MMCache(A.cache)
DV=torch.load(HERE/"cache_dev.pt",map_location="cpu")

FULL=[]
for f in sorted(glob.glob(str(ROOT/"data/magicbrush/data/dev-*.parquet"))):
    df=pd.read_parquet(f)
    for i in range(len(df)):
        src=Image.open(io.BytesIO(df.source_img.iloc[i]["bytes"])).convert("RGB")
        tgt=Image.open(io.BytesIO(df.target_img.iloc[i]["bytes"])).convert("RGB").resize(src.size)
        o=cv2.cvtColor(np.asarray(src),cv2.COLOR_RGB2BGR); e=cv2.cvtColor(np.asarray(tgt),cv2.COLOR_RGB2BGR)
        gt=ME.clean_mask(((ME.delta_e(o,e)>ME.DELTA_E_THRESHOLD)*255).astype(np.uint8))>0
        if 0.004<=float(gt.mean())<=0.45: FULL.append(gt)
print(f"train {TR.n}  dev {len(FULL)}\n", flush=True)

def cache_iou(head, D, idx, zero_text=False):
    head.eval(); v=[]
    with torch.no_grad():
        for sl in batches(len(idx),48,False):
            j=idx[sl]
            t=D["text"][j].to(dev).float()
            if zero_text: t=torch.zeros_like(t)
            lg=head(t, D["ctx"][j].to(dev).float(),
                    D["proto"][j].to(dev).float())
            p=(torch.sigmoid(lg)>0.5).float(); y=D["mask"][j].to(dev).float()
            inter=(p*y).sum((1,2,3)); union=((p+y)>0).float().sum((1,2,3)).clamp(min=1)
            v.append((inter/union).mean().item())
    head.train(); return float(np.mean(v))

def fullres(head):
    head.eval(); v=[]; byk={}
    with torch.no_grad():
        for s in range(0,DV["proto"].shape[0],32):
            j=torch.arange(s,min(s+32,DV["proto"].shape[0]))
            lg=head(DV["text"][j].to(dev).float(), DV["ctx"][j].to(dev).float(),
                    DV["proto"][j].to(dev).float())
            pr=torch.sigmoid(lg)[:,0].cpu().numpy()
            for q,ix in enumerate(j.tolist()):
                g=FULL[ix]; p=cv2.resize(pr[q],g.shape[::-1])>0.5
                u=(p|g).sum(); x=float((p&g).sum()/u) if u else 0.0
                v.append(x); byk.setdefault(DV["kind"][ix],[]).append(x)
    head.train(); return float(np.mean(v)), {k:float(np.mean(z)) for k,z in byk.items()}

class V1(torch.nn.Module):
    def __init__(s,**kw):
        super().__init__(); s.h=TextCoeffHead(**kw)
    def forward(s,t,c,p):
        co,b=s.h(t,c); return (torch.einsum("bc,bchw->bhw",co,p)+b[:,:,None]).unsqueeze(1)

ALL_SEEDS=(1368,1,2,3,4,5,6,7,8,9,10,11)
CURVES=[]

def run(make,name,seeds,epochs):
    out=[]
    for sd in seeds:
        t0=time.time()
        torch.manual_seed(sd); np.random.seed(sd)
        g=torch.Generator().manual_seed(sd)
        perm=torch.randperm(TR.n,generator=g)
        nval=int(0.15*len(perm)); vi,ti=perm[:nval],perm[nval:]
        head=make().to(dev); npar=sum(p.numel() for p in head.parameters())
        opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=0.01)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
        best=(-9,None); curve=[]
        for ep in range(epochs):
            tot=0.0; nb=0
            for sl in batches(len(ti),32,True,g):
                j=ti[sl]
                lg=head(TR["text"][j].to(dev).float(),TR["ctx"][j].to(dev).float(),
                        TR["proto"][j].to(dev).float())
                l,_,_=dice_bce_loss(lg,TR["mask"][j].to(dev).float(),w_dice=2.0)
                opt.zero_grad(); l.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.0); opt.step()
                tot+=float(l); nb+=1
            sch.step()
            s=cache_iou(head,TR,vi)                    # selection on TRAIN-val only
            # This was already computed for selection and then discarded. Keeping
            # it costs nothing and is the only way to plot a learning curve. The
            # zeroed-instruction pass is the one added cost (~10% of an epoch)
            # and is what shows language is doing the work, not the image prior.
            nt=cache_iou(head,TR,vi,zero_text=True) if A.curves else float("nan")
            curve.append(dict(epoch=ep+1, loss=tot/max(nb,1), val_iou=s,
                              val_iou_no_text=nt, delta=s-nt))
            if s>best[0]: best=(s,{k:t.clone() for k,t in head.state_dict().items()}); best_ep=ep+1
        head.load_state_dict(best[1])
        i,byk=fullres(head)                            # dev touched once
        out.append((i,byk)); CURVES.append(dict(config=name,seed=sd,best_epoch=best_ep,
                                                dev_iou=i,curve=curve))
        json.dump(CURVES, open(HERE/f"curves_{Path(A.out).stem}.json","w"), indent=2)
        print(f"    seed {sd}: dev full-res IoU {i:.4f}  ({(time.time()-t0)/60:.1f} min)",flush=True)
        # Dump after every seed, not every config: a 9-hour run that dies in
        # seed 10 should not lose the nine that finished.
        json.dump(dict(name=name, partial=True, n_train=TR.n, epochs=epochs,
                       seeds_done=[s for s in seeds[:len(out)]],
                       iou=[o[0] for o in out], by_kind=[o[1] for o in out]),
                  open(HERE/f"partial_{Path(A.out).stem}_{name.split()[0]}.json","w"), indent=2)
        torch.save(dict(head=best[1],seed=sd,dev_iou=i,n_train=TR.n,name=name),
                   HERE/f"head_full_{name.split()[0]}_s{sd}.pt")
    A_=np.array([o[0] for o in out])
    kk={k:float(np.mean([o[1].get(k,np.nan) for o in out])) for k in out[0][1]}
    sd_=float(A_.std(ddof=1)) if len(A_)>1 else 0.0
    print(f"  {name}: params {npar:,} | dev IoU {A_.mean():.4f} ± {sd_:.4f} | "
          + " ".join(f"{k} {v:.3f}" for k,v in sorted(kk.items()))+"\n",flush=True)
    return dict(name=name,params=int(npar),iou=float(A_.mean()),sd=sd_,
                by_kind=kk,seeds=A_.tolist(),n_train=TR.n,epochs=epochs)

MAKE={"v1": (lambda: V1(hidden=512), "v1 global coeff"),
      "v2": (lambda: SpatialCoeffHead(basis="none"), "v2 spatial"),
      "rand": (lambda: SpatialCoeffHead(basis="rand"), "v2 spatial+random"),
      "geom": (lambda: SpatialCoeffHead(basis="geom"), "v2 spatial+geom")}

if __name__=="__main__":
    seeds=ALL_SEEDS[:A.seeds]
    res=[]
    for c in A.configs.split(","):
        mk,nm=MAKE[c.strip()]
        print(f"{nm} ({len(seeds)} seeds, {A.epochs} epochs, n={TR.n}):",flush=True)
        res.append(run(mk,nm,seeds,A.epochs))
        json.dump(res, open(A.out,"w"), indent=2)
    print(f"{'head':22s} {'params':>11} {'dev IoU (clean)':>20}  per-kind")
    for r in res:
        print(f"{r['name']:22s} {r['params']:>11,} {r['iou']:>10.4f} ± {r['sd']:.4f}  "
              + " ".join(f"{k} {v:.3f}" for k,v in sorted(r['by_kind'].items())))
    print("\n  at n=2,278:  v1 0.1590  v2 0.1718 | CLIPSeg 0.1849 | human masks 0.1511")
    if len(res)>1 and len(seeds)>1:
        from scipy import stats; import itertools
        print("\npairwise (Welch t-test on seed means):")
        for a,b in itertools.combinations(range(len(res)),2):
            X,Y=res[a],res[b]
            t,pv=stats.ttest_ind_from_stats(X["iou"],X["sd"],len(X["seeds"]),
                                            Y["iou"],Y["sd"],len(Y["seeds"]),equal_var=False)
            print(f"  {X['name']:20s} vs {Y['name']:20s}  diff {X['iou']-Y['iou']:+.4f}  p={pv:.4f}")
