"""Clean-split evaluation — no test-set peeking.

eval_v2 selected the best epoch by dev IoU, which is the evaluation set. That
inflates every configuration equally so their ordering stands, but the absolute
numbers are not comparable to CLIPSeg or to exp09. Here the epoch is chosen on a
held-out slice of TRAIN, and dev is touched exactly once per configuration.
"""
import sys, json, glob, io
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
import mask_extraction as ME

dev=pick_device("auto")
TR=torch.load(HERE/"cache_train.pt",map_location="cpu")
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
print(f"train {TR['proto'].shape[0]}  dev {len(FULL)}\n", flush=True)

def cache_iou(head, D, idx):
    head.eval(); v=[]
    with torch.no_grad():
        for sl in batches(len(idx),48,False):
            j=idx[sl]
            lg=head(D["text"][j].to(dev).float(), D["ctx"][j].to(dev).float(),
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

def run(make,name,seeds=(1368,1,2,3,4,5,6,7,8,9,10,11),epochs=60):
    out=[]
    for sd in seeds:
        torch.manual_seed(sd); np.random.seed(sd)
        g=torch.Generator().manual_seed(sd)
        perm=torch.randperm(TR["proto"].shape[0],generator=g)
        nval=int(0.15*len(perm)); vi,ti=perm[:nval],perm[nval:]
        head=make().to(dev); npar=sum(p.numel() for p in head.parameters())
        opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=0.01)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
        best=(-9,None)
        for ep in range(epochs):
            for sl in batches(len(ti),32,True,g):
                j=ti[sl]
                lg=head(TR["text"][j].to(dev).float(),TR["ctx"][j].to(dev).float(),
                        TR["proto"][j].to(dev).float())
                l,_,_=dice_bce_loss(lg,TR["mask"][j].to(dev).float(),w_dice=2.0)
                opt.zero_grad(); l.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.0); opt.step()
            sch.step()
            s=cache_iou(head,TR,vi)                    # selection on TRAIN-val only
            if s>best[0]: best=(s,{k:t.clone() for k,t in head.state_dict().items()})
        head.load_state_dict(best[1])
        i,byk=fullres(head)                            # dev touched once
        out.append((i,byk)); print(f"    seed {sd}: dev full-res IoU {i:.4f}",flush=True)
    A=np.array([o[0] for o in out])
    kk={k:float(np.mean([o[1].get(k,np.nan) for o in out])) for k in out[0][1]}
    print(f"  {name}: params {npar:,} | dev IoU {A.mean():.4f} ± {A.std(ddof=1):.4f} | "
          + " ".join(f"{k} {v:.3f}" for k,v in sorted(kk.items()))+"\n",flush=True)
    return dict(name=name,params=int(npar),iou=float(A.mean()),sd=float(A.std(ddof=1)),
                by_kind=kk,seeds=A.tolist())

if __name__=="__main__":
    res=[]
    print("v1 global coefficients:");  res.append(run(lambda: V1(hidden=512), "v1 global coeff"))
    print("v2 spatial field:");        res.append(run(lambda: SpatialCoeffHead(basis="none"), "v2 spatial"))
    print("v2 spatial + RANDOM basis (control):")
    res.append(run(lambda: SpatialCoeffHead(basis="rand"), "v2 spatial+random"))
    print("v2 spatial + geometric:");  res.append(run(lambda: SpatialCoeffHead(basis="geom"), "v2 spatial+geom"))
    print(f"{'head':22s} {'params':>11} {'dev IoU (clean)':>20}  per-kind")
    for r in res:
        print(f"{r['name']:22s} {r['params']:>11,} {r['iou']:>10.4f} ± {r['sd']:.4f}  "
              + " ".join(f"{k} {v:.3f}" for k,v in sorted(r['by_kind'].items())))
    print("\n  reference  CLIPSeg 0.1849 | human masks 0.1511 | exp09 v1 0.1232")
    json.dump(res, open(HERE/"clean_eval.json","w"), indent=2)
    # significance against CLIPSeg and against v1
    from scipy import stats
    import itertools
    print("\npairwise (Welch t-test on seed means):")
    for a,b in itertools.combinations(range(len(res)),2):
        A,B=res[a],res[b]
        t,pv=stats.ttest_ind_from_stats(A["iou"],A["sd"],len(A["seeds"]),
                                        B["iou"],B["sd"],len(B["seeds"]),equal_var=False)
        print(f"  {A['name']:20s} vs {B['name']:20s}  diff {A['iou']-B['iou']:+.4f}  p={pv:.4f}")
