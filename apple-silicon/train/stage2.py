"""Stage 2 — edit inside the predicted region, and measure what that buys.

Stage 1 predicts WHERE. This closes the loop: feed that region to a
mask-conditioned inpainting model and compare against editing the whole frame.

The project's central claim is that predicting the region reduces collateral
damage. That has never been tested here. This tests it directly, on the same
held-out dev split, with four arms:

  A  whole-frame edit            InstructPix2Pix, no mask — the status quo
  B  edit inside OUR mask        Stage 1 prediction -> SD inpainting
  C  edit inside the GT mask     the ceiling: perfect localization
  D  edit inside MagicBrush's    what the base paper's supervision would give

Metrics, all outside the ground-truth edit region (lower is better except CLIP):

  collateral   fraction of outside-region pixels changed (CIE76 dE > 12)
  PSNR_out     preservation of the untouched part of the image
  CLIP_in      does the edit match the instruction, inside the region
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
from model_v2 import SpatialCoeffHead
import mask_extraction as ME

ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=40)
ap.add_argument("--steps",type=int,default=20); a=ap.parse_args()
dev=pick_device("auto")
SD=ROOT/"models/sd-inpaint"

# ---- samples
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
print(f"{len(rows)} dev samples, {a.steps} steps\n", flush=True)

# ---- stage 1 (ours)
seg=EditRegionModel("best.pt", device=dev)
ck=torch.load(HERE/"edit_region_head_clean.pt", map_location=dev)
seg.head=TextCoeffHead(hidden=512,dropout=0.0).to(dev); seg.head.load_state_dict(ck["head"]); seg.head.eval()

def predict_mask(src, ins):
    im=src.resize((640,640),Image.BILINEAR)
    x=torch.from_numpy(np.asarray(im)).permute(2,0,1).float().div(255)[None].to(dev)
    with torch.no_grad(): lg=seg(x,[ins])
    p=torch.sigmoid(lg)[0,0].cpu().numpy()
    return cv2.resize(p, src.size)>0.5

# ---- editors
from diffusers import (StableDiffusionInpaintPipeline, StableDiffusionInstructPix2PixPipeline,
                       EulerAncestralDiscreteScheduler)
inp=StableDiffusionInpaintPipeline.from_pretrained(str(SD), torch_dtype=torch.float16,
    variant="fp16", use_safetensors=True, safety_checker=None,
    requires_safety_checker=False).to(dev)
inp.set_progress_bar_config(disable=True)
ip2p=StableDiffusionInstructPix2PixPipeline.from_pretrained(str(ROOT/"models/instruct-pix2pix"),
    torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    safety_checker=None, requires_safety_checker=False)
ip2p.scheduler=EulerAncestralDiscreteScheduler.from_config(ip2p.scheduler.config)
ip2p=ip2p.to(dev); ip2p.set_progress_bar_config(disable=True)

import clip as clipmod
cm,cp=clipmod.load("ViT-B/32", device=dev)

def fit(im,s=512):
    w,h=im.size; k=s/min(w,h)
    return im.resize((max(8,int(w*k)//8*8), max(8,int(h*k)//8*8)), Image.LANCZOS)

def metrics(src, out, gt, ins):
    o=cv2.cvtColor(np.asarray(src),cv2.COLOR_RGB2BGR); e=cv2.cvtColor(np.asarray(out),cv2.COLOR_RGB2BGR)
    ch=ME.delta_e(o,e)>ME.DELTA_E_THRESHOLD
    outside=~gt
    coll=float(ch[outside].mean()) if outside.any() else 0.0
    a_=np.asarray(src,np.float64); b_=np.asarray(out,np.float64)
    mse=float(((a_-b_)**2)[outside].mean()) if outside.any() else 0.0
    psnr=10*np.log10(255.0**2/max(mse,1e-9))
    ys,xs=np.where(gt)
    if len(ys)>8:
        crop=out.crop((xs.min(),ys.min(),xs.max()+1,ys.max()+1))
        with torch.no_grad():
            iv=cm.encode_image(cp(crop)[None].to(dev)); iv/=iv.norm(dim=-1,keepdim=True)
            tv=cm.encode_text(clipmod.tokenize([ins],truncate=True).to(dev)); tv/=tv.norm(dim=-1,keepdim=True)
            cs=float((iv@tv.T).item())
    else: cs=float("nan")
    return coll, psnr, cs

res={k:{"coll":[],"psnr":[],"clip":[]} for k in ("A whole-frame","B our mask","C GT mask","D MagicBrush mask")}
t0=time.time()
for i,r in enumerate(rows):
    src=fit(r["src"]); W_,H_=src.size
    gt=cv2.resize(r["gt"].astype(np.uint8),(W_,H_),interpolation=cv2.INTER_NEAREST)>0
    hm=cv2.resize(r["hm"].astype(np.uint8),(W_,H_),interpolation=cv2.INTER_NEAREST)>0
    om=cv2.resize(predict_mask(r["src"],r["ins"]).astype(np.uint8),(W_,H_),interpolation=cv2.INTER_NEAREST)>0
    g=torch.Generator("cpu").manual_seed(1368+i)

    out=ip2p(r["ins"], image=src, num_inference_steps=a.steps, guidance_scale=7.5,
             image_guidance_scale=2.5, generator=g).images[0]
    for k,v in zip(("coll","psnr","clip"), metrics(src,out,gt,r["ins"])): res["A whole-frame"][k].append(v)

    for name,mk in (("B our mask",om),("C GT mask",gt),("D MagicBrush mask",hm)):
        if mk.sum()<64: continue
        mi=Image.fromarray((mk*255).astype(np.uint8))
        g2=torch.Generator("cpu").manual_seed(1368+i)
        out=inp(prompt=r["ins"], image=src, mask_image=mi, num_inference_steps=a.steps,
                generator=g2).images[0].resize(src.size)
        for k,v in zip(("coll","psnr","clip"), metrics(src,out,gt,r["ins"])): res[name][k].append(v)
    if (i+1)%5==0:
        el=time.time()-t0
        print(f"  {i+1}/{len(rows)}  {el:.0f}s  eta {(len(rows)-i-1)*el/(i+1)/60:.1f} min", flush=True)

print(f"\n{'arm':22s} {'collateral ↓':>14} {'PSNR outside ↑':>16} {'CLIP inside ↑':>14}")
summ={}
for k,v in res.items():
    c=np.mean(v["coll"]); p=np.mean(v["psnr"]); cl=np.nanmean(v["clip"])
    summ[k]=dict(collateral=float(c),psnr=float(p),clip=float(cl),n=len(v["coll"]))
    print(f"{k:22s} {c:>13.1%} {p:>15.2f} {cl:>13.4f}")
A=summ["A whole-frame"]; B=summ["B our mask"]
print(f"\n  our mask vs whole-frame: collateral {A['collateral']:.1%} -> {B['collateral']:.1%} "
      f"({A['collateral']/max(B['collateral'],1e-9):.1f}× less), PSNR +{B['psnr']-A['psnr']:.2f} dB")
json.dump(summ, open(HERE/"stage2.json","w"), indent=2)
print(f"wrote {HERE/'stage2.json'}")
