"""Stage 2b — our mask is precision-biased. How much dilation maximises net?

Stage 2a: our mask gave collateral 1.5% but recall only 7.5%, so gating threw
away most of the edit (net +6.0% against whole-frame's +9.1%). The GT mask gave
+30.1%, so localization IS worth it — our mask is simply too tight.

This sweeps a dilation radius over the predicted mask. One edit per sample,
many gates, so the cost is one IP2P pass per image regardless of sweep size.
Also sweeps the sigmoid threshold, since a lower threshold is another way to
widen the mask.
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
        o=cv2.cvtColor(np.asarray(src),cv2.COLOR_RGB2BGR); e=cv2.cvtColor(np.asarray(tgt),cv2.COLOR_RGB2BGR)
        gt=ME.clean_mask(((ME.delta_e(o,e)>ME.DELTA_E_THRESHOLD)*255).astype(np.uint8))>0
        if not (0.004<=float(gt.mean())<=0.45): continue
        rows.append(dict(src=src,gt=gt,ins=str(df.instruction.iloc[i])))
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

def fit(im,s=512):
    w,h=im.size; k=s/min(w,h)
    return im.resize((max(8,int(w*k)//8*8), max(8,int(h*k)//8*8)), Image.LANCZOS)
def gate(src,out,mask,feather=9):
    al=cv2.GaussianBlur((mask.astype(np.uint8)*255),(feather|1,feather|1),0).astype(np.float32)[...,None]/255.
    return (np.asarray(out,np.float32)*al + np.asarray(src,np.float32)*(1-al)).astype(np.uint8)

THR=[0.2,0.1]; DIL=[64,96,128,160,200]
acc={(t,d):{"rec":[],"col":[]} for t in THR for d in DIL}
acc["whole"]={"rec":[],"col":[]}; acc["gt"]={"rec":[],"col":[]}
t0=time.time()
for i,r in enumerate(rows):
    src=fit(r["src"]); W_,H_=src.size
    gt=cv2.resize(r["gt"].astype(np.uint8),(W_,H_),interpolation=cv2.INTER_NEAREST)>0
    im640=r["src"].resize((640,640),Image.BILINEAR)
    x=torch.from_numpy(np.asarray(im640)).permute(2,0,1).float().div(255)[None].to(dev)
    with torch.no_grad(): prob=torch.sigmoid(seg(x,[r["ins"]]))[0,0].cpu().numpy()
    prob=cv2.resize(prob,(W_,H_))
    g=torch.Generator("cpu").manual_seed(1368+i)
    raw=np.asarray(ip(r["ins"], image=src, num_inference_steps=a.steps, guidance_scale=7.5,
        image_guidance_scale=2.5, generator=g).images[0].resize(src.size))
    o=cv2.cvtColor(np.asarray(src),cv2.COLOR_RGB2BGR)
    def sc(img):
        e=cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        ch=ME.delta_e(o,e)>ME.DELTA_E_THRESHOLD
        return (float(ch[gt].mean()) if gt.any() else 0., float(ch[~gt].mean()) if (~gt).any() else 0.)
    rr,cc=sc(raw); acc["whole"]["rec"].append(rr); acc["whole"]["col"].append(cc)
    rr,cc=sc(gate(src,raw,gt)); acc["gt"]["rec"].append(rr); acc["gt"]["col"].append(cc)
    for t in THR:
        base=prob>t
        for d in DIL:
            m = base if d==0 else cv2.dilate(base.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(d|1,d|1)))>0
            rr,cc=sc(gate(src,raw,m))
            acc[(t,d)]["rec"].append(rr); acc[(t,d)]["col"].append(cc)
    if (i+1)%10==0:
        el=time.time()-t0; print(f"  {i+1}/{len(rows)}  {el:.0f}s  eta {(len(rows)-i-1)*el/(i+1)/60:.1f} min",flush=True)

def mm(k): 
    r=np.mean(acc[k]["rec"]); c=np.mean(acc[k]["col"]); return r,c,r-c
print(f"\n{'config':22s} {'recall ↑':>10} {'collateral ↓':>13} {'net ↑':>9}")
r,c,n=mm("whole"); print(f"{'A whole frame':22s} {r:>9.1%} {c:>12.1%} {n:>+8.1%}")
r,c,n=mm("gt");    print(f"{'C GT mask (ceiling)':22s} {r:>9.1%} {c:>12.1%} {n:>+8.1%}")
print()
best=None; out={}
for t in THR:
    for d in DIL:
        r,c,n=mm((t,d)); out[f"thr{t}_dil{d}"]=dict(recall=r,collateral=c,net=n)
        star="  ←" if (best is None or n>best[0]) else ""
        if best is None or n>best[0]: best=(n,t,d)
        print(f"  thr {t:<4} dilate {d:<3}      {r:>9.1%} {c:>12.1%} {n:>+8.1%}{star}")
n,t,d=best
wn=mm("whole")[2]
print(f"\nBEST: threshold {t}, dilate {d}px  ->  net {n:+.1%}   (whole frame {wn:+.1%})")
print(f"  {'BEATS' if n>wn else 'does not beat'} whole-frame editing")
json.dump(dict(whole=dict(zip(('recall','collateral','net'),mm('whole'))),
               gt=dict(zip(('recall','collateral','net'),mm('gt'))),
               sweep=out, best=dict(net=n,threshold=t,dilate=d)),
          open(HERE/"stage2_dilate_ext.json","w"), indent=2)
print(f"wrote {HERE/'stage2_dilate.json'}")
