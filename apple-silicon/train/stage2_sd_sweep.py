"""Re-tune the mask operating point for inpainting.

exp16 carried threshold 0.2 and 64 px dilation over from exp13, where the mask
only gates a finished edit so an over-large mask costs nothing. Under inpainting
the mask decides what gets REGENERATED, so the same dilation destroys content it
should have preserved: collateral rose to 39.1% against 24.2% for whole-frame.

This sweeps dilation for the predicted-mask arm only. Arms A, C and D are fixed
and are read from stage2_sd.json rather than recomputed, so every comparison is
against the identical baseline on the identical 60 samples.
"""
import sys, json, glob, io, time, argparse
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
ap.add_argument("--steps",type=int,default=20)
ap.add_argument("--seed",type=int,default=1368)
ap.add_argument("--thr",type=float,default=0.2)
ap.add_argument("--dilations",default="0,16,32")
a=ap.parse_args()
DILS=[int(x) for x in a.dilations.split(",")]
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
print(f"{len(rows)} samples | dilations {DILS} | thr {a.thr}\n", flush=True)

seg=EditRegionModel("best.pt", device=dev)
ck=torch.load(HERE/f"head_full_v2_s{a.seed}.pt", map_location=dev)
seg.head=SpatialCoeffHead(basis="none").to(dev)
seg.head.load_state_dict(ck["head"]); seg.head.eval()
print(f"stage 1: v2 spatial, n_train {ck['n_train']}, dev IoU {ck['dev_iou']:.4f}", flush=True)

def prob(src, ins):
    im=src.resize((640,640),Image.BILINEAR)
    x=torch.from_numpy(np.asarray(im)).permute(2,0,1).float().div(255)[None].to(dev)
    with torch.no_grad():
        pr,ctx=seg.backbone(x); t=seg.encode_text([ins])
        lg=seg.head(t.float(), ctx.float(), pr.float())
    return cv2.resize(torch.sigmoid(lg)[0,0].cpu().numpy(), src.size)

from diffusers import StableDiffusionInpaintPipeline
inp=StableDiffusionInpaintPipeline.from_pretrained(str(ROOT/"models/sd-inpaint"),
    torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    safety_checker=None, feature_extractor=None, requires_safety_checker=False).to(dev)
inp.set_progress_bar_config(disable=True)

def fit(im,s=512):
    w,h=im.size; k=s/min(w,h)
    return im.resize((max(8,int(w*k)//8*8), max(8,int(h*k)//8*8)), Image.LANCZOS)

def metrics(src,out,gt):
    o=cv2.cvtColor(np.asarray(src),cv2.COLOR_RGB2BGR); e=cv2.cvtColor(np.asarray(out),cv2.COLOR_RGB2BGR)
    ch=ME.delta_e(o,e)>ME.DELTA_E_THRESHOLD
    outside=~gt
    coll=float(ch[outside].mean()) if outside.any() else 0.0
    rec=float(ch[gt].mean()) if gt.any() else 0.0
    A=np.asarray(src,np.float64); B=np.asarray(out,np.float64)
    mse=float(((A-B)**2)[outside].mean()) if outside.any() else 0.0
    return rec, coll, 10*np.log10(255.0**2/max(mse,1e-9))

# cache the probability maps once — they do not depend on dilation
cache=[]
for r in rows:
    src=fit(r["src"]); W,H=src.size
    gt=cv2.resize(r["gt"].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST)>0
    p=cv2.resize(prob(r["src"],r["ins"]),(W,H))
    cache.append((src,gt,p,r["ins"]))
print("probability maps cached\n", flush=True)

out={}
t0=time.time()
for dil in DILS:
    rec=[];coll=[];ps=[];area=[]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dil+1,2*dil+1)) if dil>0 else None
    for i,(src,gt,p,ins) in enumerate(cache):
        m = p > a.thr
        if k is not None: m = cv2.dilate(m.astype(np.uint8),k)>0
        area.append(float(m.mean()))
        if m.sum()<64: continue
        g2=torch.Generator("cpu").manual_seed(1368+i)
        o=inp(prompt=ins, image=src, mask_image=Image.fromarray((m*255).astype(np.uint8)),
              num_inference_steps=a.steps, generator=g2).images[0].resize(src.size)
        r_,c_,ps_=metrics(src,o,gt); rec.append(r_); coll.append(c_); ps.append(ps_)
        if (i+1)%20==0:
            el=time.time()-t0
            print(f"  dil {dil}: {i+1}/{len(cache)}  {el:.0f}s", flush=True)
    R,C=float(np.mean(rec)),float(np.mean(coll))
    out[dil]=dict(recall=R,collateral=C,net=R-C,psnr=float(np.mean(ps)),
                  mask_area=float(np.mean(area)),n=len(rec))
    print(f"  dilate {dil:3d} px -> recall {R:.1%}  collateral {C:.1%}  net {R-C:+.1%}  "
          f"PSNR {np.mean(ps):.2f}  mask area {np.mean(area):.1%}", flush=True)

prev=json.load(open(HERE/"stage2_sd.json"))["arms"]
out[64]=dict(recall=prev["B our mask"]["recall"],collateral=prev["B our mask"]["collateral"],
             net=prev["B our mask"]["net"],psnr=prev["B our mask"]["psnr"],
             mask_area=None,n=prev["B our mask"]["n"])
json.dump(dict(thr=a.thr,n=a.n,steps=a.steps,arms_fixed=prev,sweep=out),
          open(HERE/"stage2_sd_sweep.json","w"), indent=2)
print(f"\n{'dilate':>8} {'recall':>9} {'collateral':>12} {'net':>9} {'PSNR out':>10}")
for dil in sorted(out):
    v=out[dil]
    print(f"{dil:>6} px {v['recall']:>8.1%} {v['collateral']:>11.1%} {v['net']:>+8.1%} {v['psnr']:>9.2f}")
best=max(out,key=lambda k:out[k]["net"])
print(f"\n  best net at {best} px: {out[best]['net']:+.1%}  (64 px gave {out[64]['net']:+.1%})")
print(f"  whole-frame {prev['A whole-frame']['net']:+.1%} | GT oracle {prev['C GT mask']['net']:+.1%} "
      f"| MagicBrush {prev['D MagicBrush mask']['net']:+.1%}")
print("wrote train/stage2_sd_sweep.json")
