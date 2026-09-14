"""Baselines we can actually run, scored on the same metric as ours.

AdaptEdit reports L1/L2/CLIP-I/CLIP-T/DINO on the edited image and never reports
a mask IoU, so there is no localization number of theirs to reproduce. This
builds the comparison it is missing: every method that produces a region from
an instruction, scored by IoU against the same ΔE targets, on the same
MagicBrush dev split.

  random          a centred box of the mean target area — floor
  full-frame      predict everything — what "no localization" scores
  MagicBrush GT   the human annotations the base paper TRAINS on
  CLIPSeg         150M text->mask, the standard referring segmenter
  FastSAM+CLIP    the inherited pipeline (exp01 scored 0/7 qualitatively)
  ours            1.4M text-conditioned prototype coefficients
"""
import glob, io, sys, json
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"datagen"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, cv2, pandas as pd
from PIL import Image
from src.device import pick_device
import mask_extraction as ME

dev=pick_device("auto")
N=503

def kind_of(s):
    s=s.strip().lower()
    return "insert" if s.startswith(("put ","add ")) else ("remove" if s.startswith(("remove","delete")) else "modify")

def iou(p,g):
    p=p.astype(bool); g=g.astype(bool)
    u=(p|g).sum()
    return float((p&g).sum()/u) if u else 0.0

# ---- load dev samples
rows=[]
for f in sorted(glob.glob(str(ROOT/"data/magicbrush/data/dev-*.parquet"))):
    df=pd.read_parquet(f)
    for i in range(len(df)):
        src=Image.open(io.BytesIO(df.source_img.iloc[i]["bytes"])).convert("RGB")
        tgt=Image.open(io.BytesIO(df.target_img.iloc[i]["bytes"])).convert("RGB").resize(src.size)
        hm=~(np.array(Image.open(io.BytesIO(df.mask_img.iloc[i]["bytes"])).convert("L")
             .resize(src.size,Image.NEAREST))>127)
        o=cv2.cvtColor(np.asarray(src),cv2.COLOR_RGB2BGR); e=cv2.cvtColor(np.asarray(tgt),cv2.COLOR_RGB2BGR)
        gt=ME.clean_mask(((ME.delta_e(o,e)>ME.DELTA_E_THRESHOLD)*255).astype(np.uint8))>0
        fr=float(gt.mean())
        if not (0.004<=fr<=0.45): continue
        rows.append(dict(src=src,gt=gt,hm=hm,ins=str(df.instruction.iloc[i]),kind=kind_of(str(df.instruction.iloc[i]))))
        if len(rows)>=N: break
    if len(rows)>=N: break
print(f"{len(rows)} dev samples")
mean_area=float(np.mean([r["gt"].mean() for r in rows]))
print(f"mean target area {mean_area:.2%}\n")

res={}
def record(name, fn):
    per={}; allv=[]
    for r in rows:
        v=fn(r); allv.append(v); per.setdefault(r["kind"],[]).append(v)
    res[name]=dict(all=float(np.mean(allv)), **{k:float(np.mean(v)) for k,v in per.items()})
    print(f"  {name:22s} {np.mean(allv):.4f}   " +
          "  ".join(f"{k} {np.mean(v):.4f}" for k,v in sorted(per.items())), flush=True)

print(f"{'method':24s} {'IoU':>6}   per-kind")
# floor: centred box of mean area
def centred(r):
    h,w=r["gt"].shape; side=int(np.sqrt(mean_area*h*w))
    m=np.zeros((h,w),bool)
    y0=max(0,h//2-side//2); x0=max(0,w//2-side//2)
    m[y0:y0+side, x0:x0+side]=True
    return iou(m,r["gt"])
record("random centred box", centred)
record("full frame", lambda r: iou(np.ones_like(r["gt"]), r["gt"]))
record("MagicBrush GT mask", lambda r: iou(r["hm"], r["gt"]))

# CLIPSeg
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
proc=CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
cs=CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(dev).eval()
def clipseg(r):
    inp=proc(text=[r["ins"]], images=[r["src"]], padding=True, return_tensors="pt").to(dev)
    with torch.no_grad(): lo=cs(**inp).logits
    p=torch.sigmoid(lo).squeeze().cpu().numpy()
    p=cv2.resize(p, r["gt"].shape[::-1])
    return iou(p>0.5, r["gt"])
record("CLIPSeg (150M)", clipseg)

# ours
from model import EditRegionModel, TextCoeffHead
ours=EditRegionModel("best.pt", device=dev)
ck=torch.load(HERE/"edit_region_head_clean.pt", map_location=dev); cfg=ck.get("cfg",{})
ours.head=TextCoeffHead(hidden=cfg.get("hidden",512), dropout=0.0).to(dev)
ours.head.load_state_dict(ck["head"]); ours.head.eval()
def ourm(r):
    im=r["src"].resize((640,640),Image.BILINEAR)
    x=torch.from_numpy(np.asarray(im)).permute(2,0,1).float().div(255)[None].to(dev)
    with torch.no_grad(): lg=ours(x,[r["ins"]])
    p=torch.sigmoid(lg)[0,0].cpu().numpy()
    p=cv2.resize(p, r["gt"].shape[::-1])
    return iou(p>0.5, r["gt"])
record("ours (1.4M trainable)", ourm)

json.dump(res, open(HERE/"baselines.json","w"), indent=2)
print(f"\nwrote {HERE/'baselines.json'}")
