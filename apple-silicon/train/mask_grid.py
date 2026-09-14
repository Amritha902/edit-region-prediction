"""Can mask geometry alone predict Stage 2 net, without running diffusion?

If inpainting changes approximately everything inside the mask and nothing
outside, then
    change-recall  ~ fraction of the true region the mask covers   (mask recall)
    collateral     ~ fraction of everything else the mask covers   (mask FPR)
    net            ~ recall - FPR                                  (Youden's J)
Mask geometry is free. If J tracks the four measured points, the whole
threshold x dilation grid can be searched without diffusion and only the best
cells confirmed.
"""
import sys, glob, io, json, argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"datagen"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, cv2, pandas as pd
from PIL import Image
from src.device import pick_device
from model import EditRegionModel
from model_v2 import SpatialCoeffHead
import mask_extraction as ME

ap=argparse.ArgumentParser()
ap.add_argument("--n",type=int,default=60)
ap.add_argument("--seed",type=int,default=1368)
ap.add_argument("--thrs",default="0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6")
ap.add_argument("--dils",default="0,4,8,16,24,32,48,64")
a=ap.parse_args()
THRS=[float(x) for x in a.thrs.split(",")]; DILS=[int(x) for x in a.dils.split(",")]
dev=pick_device("auto")

rows=[]
for f in sorted(glob.glob(str(ROOT/"data/magicbrush/data/dev-*.parquet"))):
    df=pd.read_parquet(f)
    for i in range(len(df)):
        src=Image.open(io.BytesIO(df.source_img.iloc[i]["bytes"])).convert("RGB")
        tgt=Image.open(io.BytesIO(df.target_img.iloc[i]["bytes"])).convert("RGB").resize(src.size)
        o=cv2.cvtColor(np.asarray(src),cv2.COLOR_RGB2BGR); e=cv2.cvtColor(np.asarray(tgt),cv2.COLOR_RGB2BGR)
        gt=ME.clean_mask(((ME.delta_e(o,e)>ME.DELTA_E_THRESHOLD)*255).astype(np.uint8))>0
        if not (0.004<=float(gt.mean())<=0.45): continue
        rows.append(dict(src=src,gt=gt,ins=str(df.instruction.iloc[i])))
        if len(rows)>=a.n: break
    if len(rows)>=a.n: break

seg=EditRegionModel("best.pt", device=dev)
ck=torch.load(HERE/f"head_full_v2_s{a.seed}.pt", map_location=dev)
seg.head=SpatialCoeffHead(basis="none").to(dev); seg.head.load_state_dict(ck["head"]); seg.head.eval()

def fit(im,s=512):
    w,h=im.size; k=s/min(w,h)
    return im.resize((max(8,int(w*k)//8*8), max(8,int(h*k)//8*8)), Image.LANCZOS)

cache=[]
for r in rows:
    src=fit(r["src"]); W,H=src.size
    gt=cv2.resize(r["gt"].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST)>0
    im=src.resize((640,640),Image.BILINEAR)
    x=torch.from_numpy(np.asarray(im)).permute(2,0,1).float().div(255)[None].to(dev)
    with torch.no_grad():
        pr,ctx=seg.backbone(x); t=seg.encode_text([r["ins"]])
        lg=seg.head(t.float(), ctx.float(), pr.float())
    p=cv2.resize(torch.sigmoid(lg)[0,0].cpu().numpy(),(W,H))
    cache.append((p,gt))
print(f"{len(cache)} samples cached\n", flush=True)

K={d: cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*d+1,2*d+1)) for d in DILS if d>0}
grid={}
for t in THRS:
    for d in DILS:
        rec=[];fpr=[];ar=[]
        for p,gt in cache:
            m=p>t
            if d>0: m=cv2.dilate(m.astype(np.uint8),K[d])>0
            rec.append(float(m[gt].mean()) if gt.any() else 0.0)
            fpr.append(float(m[~gt].mean()) if (~gt).any() else 0.0)
            ar.append(float(m.mean()))
        R,F=float(np.mean(rec)),float(np.mean(fpr))
        grid[f"{t}_{d}"]=dict(thr=t,dil=d,mask_recall=R,mask_fpr=F,J=R-F,area=float(np.mean(ar)))

# calibrate against the four cells that were actually diffused
meas={(0.2,0):0.304,(0.2,16):0.355,(0.2,32):0.348,(0.2,64):0.320}
print("calibration — measured net vs the free proxy J:")
xs=[];ys=[]
for (t,d),net in meas.items():
    g=grid[f"{t}_{d}"]; xs.append(g["J"]); ys.append(net)
    print(f"  thr {t} dil {d:2d}:  measured net {net:+.3f}   J {g['J']:+.3f}   "
          f"mask recall {g['mask_recall']:.3f} fpr {g['mask_fpr']:.3f}")
r=float(np.corrcoef(xs,ys)[0,1])
print(f"\n  Pearson r between J and measured net: {r:.3f}")
order_ok = [d for _,d in sorted(zip(ys,[k[1] for k in meas]))] == \
           [d for _,d in sorted(zip(xs,[k[1] for k in meas]))]
print(f"  ranking of the four cells preserved: {order_ok}")

best=sorted(grid.values(), key=lambda g:-g["J"])[:12]
print(f"\ntop cells by J (free proxy):")
print(f"  {'thr':>5} {'dil':>4} {'mask rec':>9} {'fpr':>7} {'J':>8} {'area':>7}")
for g in best:
    print(f"  {g['thr']:>5} {g['dil']:>4} {g['mask_recall']:>9.3f} {g['mask_fpr']:>7.3f} "
          f"{g['J']:>+8.3f} {g['area']:>6.1%}")
json.dump(dict(grid=grid, calibration=dict(r=r, measured={f"{t}_{d}":v for (t,d),v in meas.items()}, order_preserved=bool(order_ok)),
               top=[g for g in best]), open(HERE/"mask_grid.json","w"), indent=2)
print("\nwrote train/mask_grid.json")
