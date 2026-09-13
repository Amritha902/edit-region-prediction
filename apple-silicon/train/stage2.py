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
ap.add_argument("--steps",type=int,default=20)
ap.add_argument("--seed",type=int,default=1368)
ap.add_argument("--thr",type=float,default=0.2)
ap.add_argument("--dilate",type=int,default=64)
ap.add_argument("--out",default="stage2_sd.json")
a=ap.parse_args()
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

# ---- stage 1 (ours): the v2 head trained on all 8,306 samples, not the old
# v1 checkpoint at 2,278. exp13 showed the raw mask is precision-biased -- 1.5%
# collateral but only 7.5% recall -- so gating on it discards most of the edit.
# Use the operating point that sweep established: threshold 0.2, dilate 64 px.
seg=EditRegionModel("best.pt", device=dev)
ck=torch.load(HERE/f"head_full_v2_s{a.seed}.pt", map_location=dev)
seg.head=SpatialCoeffHead(basis="none").to(dev)
seg.head.load_state_dict(ck["head"]); seg.head.eval()
print(f"stage 1: v2 spatial, seed {ck['seed']}, n_train {ck['n_train']}, "
      f"dev IoU {ck['dev_iou']:.4f}", flush=True)

def predict_prob(src, ins):
    im=src.resize((640,640),Image.BILINEAR)
    x=torch.from_numpy(np.asarray(im)).permute(2,0,1).float().div(255)[None].to(dev)
    with torch.no_grad():
        proto,ctx=seg.backbone(x); temb=seg.encode_text([ins])
        lg=seg.head(temb.float(), ctx.float(), proto.float())
    return cv2.resize(torch.sigmoid(lg)[0,0].cpu().numpy(), src.size)

def predict_mask(src, ins, thr=None, dil=None):
    thr = a.thr if thr is None else thr
    dil = a.dilate if dil is None else dil
    m = predict_prob(src, ins) > thr
    if dil > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dil+1, 2*dil+1))
        m = cv2.dilate(m.astype(np.uint8), k) > 0
    return m

# ---- editors
from diffusers import (StableDiffusionInpaintPipeline, StableDiffusionInstructPix2PixPipeline,
                       EulerAncestralDiscreteScheduler)
# The local copy was fetched without feature_extractor/safety_checker (we disable
# the checker anyway), so both must be passed as None or diffusers goes looking
# for a preprocessor_config.json that was never downloaded.
inp=StableDiffusionInpaintPipeline.from_pretrained(str(SD), torch_dtype=torch.float16,
    variant="fp16", use_safetensors=True, safety_checker=None,
    feature_extractor=None, requires_safety_checker=False).to(dev)
inp.set_progress_bar_config(disable=True)
ip2p=StableDiffusionInstructPix2PixPipeline.from_pretrained(str(ROOT/"models/instruct-pix2pix"),
    torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    safety_checker=None, feature_extractor=None, requires_safety_checker=False)
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
    rec=float(ch[gt].mean()) if gt.any() else 0.0
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
    return coll, psnr, cs, rec

res={k:{"coll":[],"psnr":[],"clip":[],"rec":[]} for k in
     ("A whole-frame","B our mask","C GT mask","D MagicBrush mask")}
t0=time.time()
for i,r in enumerate(rows):
    src=fit(r["src"]); W_,H_=src.size
    gt=cv2.resize(r["gt"].astype(np.uint8),(W_,H_),interpolation=cv2.INTER_NEAREST)>0
    hm=cv2.resize(r["hm"].astype(np.uint8),(W_,H_),interpolation=cv2.INTER_NEAREST)>0
    om=cv2.resize(predict_mask(r["src"],r["ins"]).astype(np.uint8),(W_,H_),interpolation=cv2.INTER_NEAREST)>0
    g=torch.Generator("cpu").manual_seed(1368+i)

    out=ip2p(r["ins"], image=src, num_inference_steps=a.steps, guidance_scale=7.5,
             image_guidance_scale=2.5, generator=g).images[0]
    for k,v in zip(("coll","psnr","clip","rec"), metrics(src,out,gt,r["ins"])): res["A whole-frame"][k].append(v)

    for name,mk in (("B our mask",om),("C GT mask",gt),("D MagicBrush mask",hm)):
        if mk.sum()<64: continue
        mi=Image.fromarray((mk*255).astype(np.uint8))
        g2=torch.Generator("cpu").manual_seed(1368+i)
        out=inp(prompt=r["ins"], image=src, mask_image=mi, num_inference_steps=a.steps,
                generator=g2).images[0].resize(src.size)
        for k,v in zip(("coll","psnr","clip","rec"), metrics(src,out,gt,r["ins"])): res[name][k].append(v)
    if (i+1)%5==0:
        el=time.time()-t0
        print(f"  {i+1}/{len(rows)}  {el:.0f}s  eta {(len(rows)-i-1)*el/(i+1)/60:.1f} min", flush=True)

print(f"\n{'arm':22s} {'recall ↑':>10} {'collateral ↓':>13} {'net ↑':>9} {'PSNR out ↑':>12} {'CLIP in ↑':>11}")
summ={}
for k,v in res.items():
    if not v["coll"]: continue
    c=np.mean(v["coll"]); p=np.mean(v["psnr"]); cl=np.nanmean(v["clip"]); rc=np.mean(v["rec"])
    summ[k]=dict(recall=float(rc),collateral=float(c),net=float(rc-c),
                 psnr=float(p),clip=float(cl),n=len(v["coll"]))
    print(f"{k:22s} {rc:>9.1%} {c:>12.1%} {rc-c:>+8.1%} {p:>11.2f} {cl:>10.4f}")
A=summ["A whole-frame"]; B=summ.get("B our mask")
if B:
    print(f"\n  ours vs whole-frame: net {A['net']:+.1%} -> {B['net']:+.1%}; "
          f"collateral {A['collateral']:.1%} -> {B['collateral']:.1%} "
          f"({B['collateral']/max(A['collateral'],1e-9):.1f}x, "
          f"{'MORE' if B['collateral']>A['collateral'] else 'less'}); "
          f"PSNR {B['psnr']-A['psnr']:+.2f} dB")
    C=summ.get("C GT mask")
    if C: print(f"  localization value: GT net {C['net']:+.1%} vs whole-frame {A['net']:+.1%} "
                f"= {C['net']/max(A['net'],1e-9):.1f}x")
json.dump(dict(config=vars(a), arms=summ), open(HERE/a.out,"w"), indent=2)
print(f"\nwrote train/{a.out}")
json.dump(summ, open(HERE/"stage2.json","w"), indent=2)
print(f"wrote {HERE/'stage2.json'}")
