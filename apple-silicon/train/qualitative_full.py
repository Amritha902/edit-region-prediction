"""Qualitative evidence from the full-scale v2 head: what it actually predicts.

Loads one trained seed from the 8,306-sample run and renders, per example:
source | DeltaE ground truth | our prediction | overlay, with the measured IoU.
Examples are picked to cover all three edit kinds, including the insertion case
the project is about -- shown as measured, not cherry-picked to flatter.
"""
import argparse, glob, io, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"datagen"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, cv2, pandas as pd
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from src.device import pick_device
from model import EditRegionModel
from model_v2 import SpatialCoeffHead
import mask_extraction as ME

ap=argparse.ArgumentParser()
ap.add_argument("--ckpt", default=str(HERE/"head_full_v2_s1368.pt"))
ap.add_argument("--per-kind", type=int, default=2)
ap.add_argument("--out", default=str(ROOT/"implementation/qualitative_full.png"))
A=ap.parse_args()

dev=pick_device("auto")
model=EditRegionModel("best.pt", device=dev)
ck=torch.load(A.ckpt, map_location=dev)
head=SpatialCoeffHead(basis="none").to(dev)
head.load_state_dict(ck["head"]); head.eval()
print(f"loaded {Path(A.ckpt).name}: seed {ck['seed']} dev IoU {ck['dev_iou']:.4f} "
      f"n_train {ck['n_train']}", flush=True)

df=pd.concat([pd.read_parquet(f) for f in
               sorted(glob.glob(str(ROOT/"data/magicbrush/data/dev-*.parquet")))],
              ignore_index=True)
def kind_of(s):
    s=s.strip().lower()
    return "insert" if s.startswith(("put ","add ")) else \
           "remove" if s.startswith(("remove","delete")) else "modify"

# MagicBrush is multi-turn: consecutive rows edit the SAME photo, so taking the
# first two of each kind showed three scenes twice over. Key on the source image
# so every row is a different photograph.
# ...and turn 2's source IS turn 1's target, so hashing the bytes does not catch
# it either. img_id is the scene; key on that.
picks=[]; count={}; seen=set()
for i in range(len(df)):
    k=kind_of(str(df.instruction.iloc[i]))
    if count.get(k,0)>=A.per_kind: continue
    h=int(df.img_id.iloc[i])
    if h in seen: continue
    seen.add(h); count[k]=count.get(k,0)+1; picks.append((i,k))
    if len(picks)>=3*A.per_kind: break
picks.sort(key=lambda t:("insert","remove","modify").index(t[1]))

rows=[]
for i,k in picks:
    src=Image.open(io.BytesIO(df.source_img.iloc[i]["bytes"])).convert("RGB")
    tgt=Image.open(io.BytesIO(df.target_img.iloc[i]["bytes"])).convert("RGB").resize(src.size)
    ins=str(df.instruction.iloc[i])
    o=cv2.cvtColor(np.asarray(src),cv2.COLOR_RGB2BGR); e=cv2.cvtColor(np.asarray(tgt),cv2.COLOR_RGB2BGR)
    gt=ME.clean_mask(((ME.delta_e(o,e)>ME.DELTA_E_THRESHOLD)*255).astype(np.uint8))>0
    img=src.resize((640,640),Image.BILINEAR)
    x=torch.from_numpy(np.asarray(img)).permute(2,0,1).float().div(255)[None].to(dev)
    with torch.no_grad():
        proto,ctx=model.backbone(x); temb=model.encode_text([ins])
        lg=head(temb.float(), ctx.float(), proto.float())
        pr=torch.sigmoid(lg)[0,0].cpu().numpy()
    pr=cv2.resize(pr,src.size)>0.5
    iou=float((pr&gt).sum()/max((pr|gt).sum(),1))
    rows.append((np.asarray(src),gt,pr,ins,k,iou))
    print(f"  {k:7s} IoU {iou:.3f}  {ins[:52]}", flush=True)

n=len(rows)
fig,ax=plt.subplots(n,4,figsize=(14.5,3.25*n))
if n==1: ax=ax[None,:]
for r,(s_,gt,pr,ins,k,iou) in enumerate(rows):
    ov=s_.copy().astype(float)
    ov[gt]=0.55*ov[gt]+0.45*np.array([255,80,80])      # truth  red
    ov[pr]=0.55*ov[pr]+0.45*np.array([80,200,120])     # pred   green
    ov[gt&pr]=0.45*ov[gt&pr]+0.55*np.array([250,220,60])  # both yellow
    for c,(im,ttl) in enumerate([(s_,"source"),(gt,"ΔE ground truth"),
                                 (pr,"our prediction"),(ov.astype(np.uint8),"overlay")]):
        ax[r,c].imshow(im, cmap=None if im.ndim==3 else "gray")
        ax[r,c].set_xticks([]); ax[r,c].set_yticks([])
        if r==0: ax[r,c].set_title(ttl, fontsize=11, fontweight="bold")
    ax[r,0].set_ylabel(f"{k}\nIoU {iou:.3f}", fontsize=11, fontweight="bold")
    ax[r,0].text(0.02,0.02,f'"{ins}"',transform=ax[r,0].transAxes,ha="left",
                 va="bottom",fontsize=9.5,style="italic",color="black",
                 bbox=dict(fc="white",alpha=0.82,ec="none",pad=2.5),
                 wrap=True)
fig.legend(handles=[Patch(color="#ff5050",label="ground truth only"),
                    Patch(color="#50c878",label="prediction only"),
                    Patch(color="#fadc3c",label="agreement")],
           loc="lower center", ncol=3, frameon=False, fontsize=10)
fig.suptitle(f"v2 spatial head, trained on {ck['n_train']:,} samples "
             f"(seed {ck['seed']}, dev IoU {ck['dev_iou']:.4f})",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0,0.035,1,0.975])
Path(A.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(A.out, dpi=150, bbox_inches="tight")
print(f"\nwrote {A.out}")
