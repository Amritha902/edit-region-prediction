"""v2 against v1, CLIPSeg and the controls — same protocol as exp09/11.

Evaluated at FULL RESOLUTION against the true (undilated) mask, which is the
honest metric from exp09: the 160x160 number is optimistic because the training
target is dilated by cv2.resize(INTER_AREA) > 0.
"""
import sys, json, glob, io
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"datagen"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, cv2, pandas as pd
from PIL import Image
from src.device import pick_device
from model import TextCoeffHead, dice_bce_loss, mask_metrics
from model_v2 import SpatialCoeffHead
from train_cached import batches
import mask_extraction as ME

dev=pick_device("auto")
TR=torch.load(HERE/"cache_train.pt",map_location="cpu")
DV=torch.load(HERE/"cache_dev.pt",map_location="cpu")
print(f"train {TR['proto'].shape[0]}  eval {DV['proto'].shape[0]}\n")

# full-resolution ground truth for the dev split, in cache order
print("building full-resolution dev targets ...", flush=True)
FULL=[]
for f in sorted(glob.glob(str(ROOT/"data/magicbrush/data/dev-*.parquet"))):
    df=pd.read_parquet(f)
    for i in range(len(df)):
        src=Image.open(io.BytesIO(df.source_img.iloc[i]["bytes"])).convert("RGB")
        tgt=Image.open(io.BytesIO(df.target_img.iloc[i]["bytes"])).convert("RGB").resize(src.size)
        o=cv2.cvtColor(np.asarray(src),cv2.COLOR_RGB2BGR); e=cv2.cvtColor(np.asarray(tgt),cv2.COLOR_RGB2BGR)
        gt=ME.clean_mask(((ME.delta_e(o,e)>ME.DELTA_E_THRESHOLD)*255).astype(np.uint8))>0
        fr=float(gt.mean())
        if not (0.004<=fr<=0.45): continue
        FULL.append(gt)
assert len(FULL)==DV["proto"].shape[0], f"{len(FULL)} vs {DV['proto'].shape[0]}"
print(f"  {len(FULL)} targets\n")

def full_res_eval(head, D, full):
    head.eval(); ious=[]; byk={}
    with torch.no_grad():
        for s in range(0, D["proto"].shape[0], 32):
            j=torch.arange(s, min(s+32, D["proto"].shape[0]))
            lg=head(D["text"][j].to(dev).float(), D["ctx"][j].to(dev).float(),
                    D["proto"][j].to(dev).float())
            pr=torch.sigmoid(lg)[:,0].cpu().numpy()
            for q,idx in enumerate(j.tolist()):
                g=full[idx]
                p=cv2.resize(pr[q], g.shape[::-1])>0.5
                u=(p|g).sum(); v=float((p&g).sum()/u) if u else 0.0
                ious.append(v); byk.setdefault(D["kind"][idx],[]).append(v)
    head.train()
    return float(np.mean(ious)), {k:float(np.mean(x)) for k,x in byk.items()}

def train_eval(make, name, seeds=(1368,1,2), epochs=40):
    out=[]
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        g=torch.Generator().manual_seed(s)
        tr=torch.randperm(TR["proto"].shape[0], generator=g)
        head=make().to(dev); npar=sum(p.numel() for p in head.parameters())
        opt=torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        best=(-9,None,None)
        for ep in range(epochs):
            for sl in batches(len(tr),32,True,g):
                j=tr[sl]
                lg=head(TR["text"][j].to(dev).float(), TR["ctx"][j].to(dev).float(),
                        TR["proto"][j].to(dev).float())
                l,_,_=dice_bce_loss(lg, TR["mask"][j].to(dev).float(), w_dice=2.0)
                opt.zero_grad(); l.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.0); opt.step()
            sch.step()
            if ep>=8 and (ep%4==0 or ep==epochs-1):
                i,byk=full_res_eval(head,DV,FULL)
                if i>best[0]: best=(i,byk,ep)
        out.append(best); print(f"    seed {s}: full-res IoU {best[0]:.4f} (ep{best[2]})", flush=True)
    A=np.array([o[0] for o in out])
    kk={k:float(np.mean([o[1].get(k,np.nan) for o in out])) for k in out[0][1]}
    print(f"  {name}: params {npar:,} | full-res IoU {A.mean():.4f} ± {A.std(ddof=1):.4f} | "
          + " ".join(f"{k} {v:.3f}" for k,v in sorted(kk.items())) + "\n", flush=True)
    return dict(name=name, params=int(npar), iou=float(A.mean()), sd=float(A.std(ddof=1)), by_kind=kk)

if __name__=="__main__":
    res=[]
    print("v1 — global coefficients (baseline):")
    class V1(torch.nn.Module):
        def __init__(s_,**kw):
            super().__init__(); s_.h=TextCoeffHead(**kw)
        def forward(s_,t,c,p):
            co,b=s_.h(t,c); return (torch.einsum("bc,bchw->bhw",co,p)+b[:,:,None]).unsqueeze(1)
    res.append(train_eval(lambda: V1(hidden=512), "v1 global coeff"))
    print("v2 — spatial field, no extra basis (isolates the field):")
    res.append(train_eval(lambda: SpatialCoeffHead(basis="none"), "v2 spatial, no basis"))
    print("v2 — spatial field + 12 RANDOM basis (control):")
    res.append(train_eval(lambda: SpatialCoeffHead(basis="rand"), "v2 spatial + random"))
    print("v2 — spatial field + 12 geometric basis:")
    res.append(train_eval(lambda: SpatialCoeffHead(basis="geom"), "v2 spatial + geometric"))
    print(f"{'head':26s} {'params':>11} {'full-res IoU':>18}  per-kind")
    for r in res:
        print(f"{r['name']:26s} {r['params']:>11,} {r['iou']:>9.4f} ± {r['sd']:.4f}  "
              + " ".join(f"{k} {v:.3f}" for k,v in sorted(r['by_kind'].items())))
    print(f"\n  reference — CLIPSeg 150M: 0.1849   MagicBrush human masks: 0.1511   v1 (exp09): 0.1232")
    json.dump(res, open(HERE/"v2_results.json","w"), indent=2)
