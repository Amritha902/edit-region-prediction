"""Choose the decision threshold honestly.

metrics_full showed dev IoU peaking at threshold 0.3 (0.2180) rather than the
0.5 we fixed a priori (0.2065). Adopting 0.3 because the DEV sweep says so would
be test-set tuning -- the same error that once inverted the v1/v2 ordering in
this project. So: select the threshold per seed on the held-out train-val slice
(the same slice that chose the epoch), then apply it to dev once.
"""
import glob, io, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"datagen"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, cv2, pandas as pd
from PIL import Image
from scipy import stats
from src.device import pick_device
from model import TextCoeffHead
from model_v2 import SpatialCoeffHead
from cachemm import MMCache
from train_cached import batches
import mask_extraction as ME

dev=pick_device("auto")
TR=MMCache(HERE/"cache_full_train"); DV=torch.load(HERE/"cache_dev.pt",map_location="cpu")
THRS=[0.1,0.2,0.3,0.4,0.5,0.6,0.7]
SEEDS=(1368,1,2,3,4,5,6,7,8,9,10,11)

FULL=[]
for f in sorted(glob.glob(str(ROOT/"data/magicbrush/data/dev-*.parquet"))):
    df=pd.read_parquet(f)
    for i in range(len(df)):
        src=Image.open(io.BytesIO(df.source_img.iloc[i]["bytes"])).convert("RGB")
        tgt=Image.open(io.BytesIO(df.target_img.iloc[i]["bytes"])).convert("RGB").resize(src.size)
        o=cv2.cvtColor(np.asarray(src),cv2.COLOR_RGB2BGR); e=cv2.cvtColor(np.asarray(tgt),cv2.COLOR_RGB2BGR)
        gt=ME.clean_mask(((ME.delta_e(o,e)>ME.DELTA_E_THRESHOLD)*255).astype(np.uint8))>0
        if 0.004<=float(gt.mean())<=0.45: FULL.append(gt)

class V1(torch.nn.Module):
    def __init__(s,**kw):
        super().__init__(); s.h=TextCoeffHead(**kw)
    def forward(s,t,c,p):
        co,b=s.h(t,c); return (torch.einsum("bc,bchw->bhw",co,p)+b[:,:,None]).unsqueeze(1)

def val_thr(head, vi):
    """Best threshold on the train-val slice, at cache resolution."""
    acc={t:[] for t in THRS}
    with torch.no_grad():
        for sl in batches(len(vi),48,False):
            j=vi[sl]
            lg=head(TR["text"][j].to(dev).float(),TR["ctx"][j].to(dev).float(),
                    TR["proto"][j].to(dev).float())
            pr=torch.sigmoid(lg); y=TR["mask"][j].to(dev).float()
            for t in THRS:
                p=(pr>t).float()
                inter=(p*y).sum((1,2,3)); union=((p+y)>0).float().sum((1,2,3)).clamp(min=1)
                acc[t].append((inter/union).mean().item())
    return max(THRS,key=lambda t: float(np.mean(acc[t]))), {t:float(np.mean(v)) for t,v in acc.items()}

def dev_at(head, t):
    v=[];byk={}
    with torch.no_grad():
        for s in range(0,DV["proto"].shape[0],32):
            j=torch.arange(s,min(s+32,DV["proto"].shape[0]))
            lg=head(DV["text"][j].to(dev).float(),DV["ctx"][j].to(dev).float(),
                    DV["proto"][j].to(dev).float())
            pr=torch.sigmoid(lg)[:,0].cpu().numpy()
            for q,ix in enumerate(j.tolist()):
                g=FULL[ix]; p=cv2.resize(pr[q],g.shape[::-1])>t
                u=(p|g).sum(); x=float((p&g).sum()/u) if u else 0.0
                v.append(x); byk.setdefault(DV["kind"][ix],[]).append(x)
    return float(np.mean(v)), {k:float(np.mean(z)) for k,z in byk.items()}

OUT={}
for tag,make in (("v2",lambda: SpatialCoeffHead(basis="none")),("v1",lambda: V1(hidden=512))):
    rows=[]
    for sd in SEEDS:
        ck=HERE/f"head_full_{tag}_s{sd}.pt"
        if not ck.exists(): continue
        head=make().to(dev); head.load_state_dict(torch.load(ck,map_location=dev)["head"]); head.eval()
        g=torch.Generator().manual_seed(sd)                 # same split as training
        perm=torch.randperm(TR.n,generator=g); vi=perm[:int(0.15*len(perm))]
        t,curve=val_thr(head,vi)
        iou,byk=dev_at(head,t)
        base,_=dev_at(head,0.5)
        rows.append(dict(seed=sd,thr=t,dev_iou=iou,dev_iou_at_05=base,by_kind=byk))
        print(f"  {tag} seed {sd}: train-val picks thr {t} -> dev {iou:.4f} "
              f"(0.5 gave {base:.4f})",flush=True)
    A=np.array([r["dev_iou"] for r in rows]); B=np.array([r["dev_iou_at_05"] for r in rows])
    from collections import Counter
    print(f"  {tag}: thr chosen {dict(Counter(r['thr'] for r in rows))} | "
          f"dev {A.mean():.4f} ± {A.std(ddof=1):.4f}  vs 0.5 {B.mean():.4f} ± {B.std(ddof=1):.4f}")
    t_,p_=stats.ttest_rel(A,B)
    print(f"       paired t-test vs fixed 0.5: {A.mean()-B.mean():+.4f}  p={p_:.2e}\n",flush=True)
    OUT[tag]=dict(rows=rows,mean=float(A.mean()),sd=float(A.std(ddof=1)),
                  mean_at_05=float(B.mean()),sd_at_05=float(B.std(ddof=1)),
                  paired_p=float(p_),
                  by_kind={k:float(np.mean([r["by_kind"][k] for r in rows if k in r["by_kind"]]))
                           for k in ("insert","modify","remove")})
json.dump(OUT,open(HERE/"threshold_select.json","w"),indent=2)
print("wrote train/threshold_select.json")
