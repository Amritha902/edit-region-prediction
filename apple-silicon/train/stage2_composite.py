"""Stage 2a — the region as a gate on one editor. No second model needed.

The cleanest controlled test of the project's central claim. ONE editor
(InstructPix2Pix) produces the edit; the only thing that varies is which mask
gates it back onto the source. So any difference is attributable to the mask,
not to a difference between two editors.

  A  whole frame          no gate — the status quo
  B  our predicted mask   Stage 1
  C  ground-truth mask    ceiling
  D  MagicBrush mask      what the base paper's supervision gives

Gating makes collateral change zero by construction, so the interesting
quantity is what it COSTS: how much of the intended edit is lost when the mask
is imperfect.

  recall_in    fraction of the true edit region that actually changed
  collateral   fraction of outside-region pixels changed  (0 by construction for B/C/D)
  net          recall_in - collateral   — the quantity that matters
  CLIP_in      does the gated result still match the instruction
"""
import sys, json, glob, io, time, argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"datagen"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, cv2, pandas as pd
from PIL import Image
from src.device import pick_device
from model import EditRegionModel, TextCoeffHead
import mask_extraction as ME

ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=60)
ap.add_argument("--steps",type=int,default=20); a=ap.parse_args()
dev=pick_device("auto")

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
        if not (0.004<=float(gt.mean())<=0.45): continue
        rows.append(dict(src=src,gt=gt,hm=hm,ins=str(df.instruction.iloc[i])))
        if len(rows)>=a.n: break
    if len(rows)>=a.n: break
print(f"{len(rows)} dev samples\n", flush=True)

seg=EditRegionModel("best.pt", device=dev)
ck=torch.load(HERE/"edit_region_head_clean.pt", map_location=dev)
seg.head=TextCoeffHead(hidden=512,dropout=0.0).to(dev); seg.head.load_state_dict(ck["head"]); seg.head.eval()

from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
ip=StableDiffusionInstructPix2PixPipeline.from_pretrained(str(ROOT/"models/instruct-pix2pix"),
   torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
   safety_checker=None, requires_safety_checker=False)
ip.scheduler=EulerAncestralDiscreteScheduler.from_config(ip.scheduler.config)
ip=ip.to(dev); ip.set_progress_bar_config(disable=True)
import clip as clipmod
cm,cp=clipmod.load("ViT-B/32", device=dev)

def fit(im,s=512):
    w,h=im.size; k=s/min(w,h)
    return im.resize((max(8,int(w*k)//8*8), max(8,int(h*k)//8*8)), Image.LANCZOS)

def gate(src, out, mask, feather=9):
    al=(mask.astype(np.uint8)*255)
    al=cv2.GaussianBlur(al,(feather|1,feather|1),0).astype(np.float32)[...,None]/255.
    return Image.fromarray((np.asarray(out,np.float32)*al +
                            np.asarray(src,np.float32)*(1-al)).astype(np.uint8))

def score(src,out,gt,ins):
    o=cv2.cvtColor(np.asarray(src),cv2.COLOR_RGB2BGR); e=cv2.cvtColor(np.asarray(out),cv2.COLOR_RGB2BGR)
    ch=ME.delta_e(o,e)>ME.DELTA_E_THRESHOLD
    rec=float(ch[gt].mean()) if gt.any() else 0.0
    col=float(ch[~gt].mean()) if (~gt).any() else 0.0
    ys,xs=np.where(gt)
    cs=float("nan")
    if len(ys)>8:
        crop=out.crop((xs.min(),ys.min(),xs.max()+1,ys.max()+1))
        with torch.no_grad():
            iv=cm.encode_image(cp(crop)[None].to(dev)); iv/=iv.norm(dim=-1,keepdim=True)
            tv=cm.encode_text(clipmod.tokenize([ins],truncate=True).to(dev)); tv/=tv.norm(dim=-1,keepdim=True)
            cs=float((iv@tv.T).item())
    return rec,col,cs

ARMS=["A whole frame","B our mask","C GT mask","D MagicBrush mask"]
res={k:{"rec":[],"col":[],"clip":[]} for k in ARMS}
t0=time.time()
for i,r in enumerate(rows):
    src=fit(r["src"]); W_,H_=src.size
    rs=lambda m: cv2.resize(m.astype(np.uint8),(W_,H_),interpolation=cv2.INTER_NEAREST)>0
    gt=rs(r["gt"]); hm=rs(r["hm"])
    im640=r["src"].resize((640,640),Image.BILINEAR)
    x=torch.from_numpy(np.asarray(im640)).permute(2,0,1).float().div(255)[None].to(dev)
    with torch.no_grad(): lg=seg(x,[r["ins"]])
    om=rs(cv2.resize(torch.sigmoid(lg)[0,0].cpu().numpy(), r["src"].size)>0.5)

    g=torch.Generator("cpu").manual_seed(1368+i)
    raw=ip(r["ins"], image=src, num_inference_steps=a.steps, guidance_scale=7.5,
           image_guidance_scale=2.5, generator=g).images[0].resize(src.size)
    for name,mk in zip(ARMS,[None,om,gt,hm]):
        out = raw if mk is None else gate(src,raw,mk)
        rec,col,cs = score(src,out,gt,r["ins"])
        res[name]["rec"].append(rec); res[name]["col"].append(col); res[name]["clip"].append(cs)
    if (i+1)%10==0:
        el=time.time()-t0
        print(f"  {i+1}/{len(rows)}  {el:.0f}s  eta {(len(rows)-i-1)*el/(i+1)/60:.1f} min", flush=True)

print(f"\n{'arm':22s} {'recall in ↑':>12} {'collateral ↓':>14} {'net ↑':>9} {'CLIP in ↑':>11}")
summ={}
for k in ARMS:
    v=res[k]; rec=np.mean(v["rec"]); col=np.mean(v["col"]); cl=np.nanmean(v["clip"])
    summ[k]=dict(recall=float(rec),collateral=float(col),net=float(rec-col),clip=float(cl),n=len(v["rec"]))
    print(f"{k:22s} {rec:>11.1%} {col:>13.1%} {rec-col:>+8.1%} {cl:>10.4f}")
A=summ["A whole frame"]; B=summ["B our mask"]
print(f"\n  our mask vs whole frame:  collateral {A['collateral']:.1%} -> {B['collateral']:.1%}"
      f"   net {A['net']:+.1%} -> {B['net']:+.1%}")
json.dump(summ, open(HERE/"stage2_composite.json","w"), indent=2)
print(f"wrote {HERE/'stage2_composite.json'}")
