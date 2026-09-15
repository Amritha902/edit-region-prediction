"""Figure: the preprocessing chain, run on real dev samples.

Every panel is the actual array at that step, not an illustration.
"""
import sys, glob, io
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"datagen"))
sys.path.insert(0,"/Users/amritha/DATASCIENCE FINAL/report")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, cv2, pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import mask_extraction as ME
import figstyle as FS; FS.use()

rows=[]
seen=set()
for f in sorted(glob.glob(str(ROOT/"data/magicbrush/data/dev-*.parquet"))):
    df=pd.read_parquet(f)
    for i in range(len(df)):
        iid=str(df.img_id.iloc[i]) if "img_id" in df.columns else str(i)
        if iid in seen: continue
        src=Image.open(io.BytesIO(df.source_img.iloc[i]["bytes"])).convert("RGB")
        tgt=Image.open(io.BytesIO(df.target_img.iloc[i]["bytes"])).convert("RGB").resize(src.size)
        ins=str(df.instruction.iloc[i])
        o=cv2.cvtColor(np.asarray(src),cv2.COLOR_RGB2BGR)
        e=cv2.cvtColor(np.asarray(tgt),cv2.COLOR_RGB2BGR)
        de=ME.delta_e(o,e)
        raw=(de>ME.DELTA_E_THRESHOLD)
        clean=ME.clean_mask((raw*255).astype(np.uint8))>0
        frac=float(clean.mean())
        if not (0.01<=frac<=0.30): continue
        hm=~(np.array(Image.open(io.BytesIO(df.mask_img.iloc[i]["bytes"])).convert("L")
             .resize(src.size,Image.NEAREST))>127)
        seen.add(iid)
        rows.append(dict(src=src,tgt=tgt,de=de,raw=raw,clean=clean,hm=hm,ins=ins,frac=frac))
        if len(rows)>=3: break
    if len(rows)>=3: break

STEPS=["Source image","Target image","CIE76 ΔE in L*a*b*",
       "Threshold ΔE > 12","After 5×5 open/close,\ncomponents < 40 px dropped",
       "MagicBrush human mask\n(baseline, not the target)"]

import textwrap
fig,axes=plt.subplots(len(rows),6,figsize=(13.4,2.35*len(rows)+1.05))
for r,R in enumerate(rows):
    ims=[np.asarray(R["src"]),np.asarray(R["tgt"]),R["de"],R["raw"],R["clean"],R["hm"]]
    for c,(ax,im) in enumerate(zip(axes[r],ims)):
        if c<2: ax.imshow(im)
        elif c==2:
            ax.imshow(im,cmap="magma",vmin=0,vmax=40)
        else:
            ax.imshow(im,cmap="gray",vmin=0,vmax=1)
        FS.nogrid(ax)
        if r==0: ax.set_title(STEPS[c],fontsize=8.6,color=FS.INK,loc="left",pad=6)
    lab="\n".join(textwrap.wrap(f'"{R["ins"]}"',24))
    axes[r,0].text(-0.07,0.5,lab,transform=axes[r,0].transAxes,fontsize=8.2,
                   color=FS.INK,rotation=0,ha="right",va="center",style="italic")
    def badge(ax,txt,hi=False):
        ax.text(0.5,-0.055,txt,transform=ax.transAxes,fontsize=8.4,
                color=FS.CAT[1] if hi else FS.MUTED,ha="center",va="top",
                weight="bold" if hi else "normal")
    badge(axes[r,4],f'{R["frac"]*100:.1f}% of frame',hi=True)
    badge(axes[r,5],f'{R["hm"].mean()*100:.1f}% of frame')
    axes[r,5].text(0.5,-0.155,f'{R["hm"].mean()/max(R["frac"],1e-6):.1f}x larger',
                   transform=axes[r,5].transAxes,fontsize=7.8,color=FS.MUTED,
                   ha="center",va="top",style="italic")

FS.figtitle(fig,"Deriving the training target from the image pair",
            "Each panel is the actual array at that step. The last column is the dataset's own "
            "annotation at the same scale; it is used as a baseline, never as a target.")
plt.tight_layout(rect=[0.105,0.005,0.995,0.9], h_pad=2.6)
plt.savefig("/Users/amritha/DATASCIENCE FINAL/report/DA2_figures/pipeline.png",dpi=185)
print("wrote pipeline.png  |",len(rows),"samples")
