"""Figure: what the head actually combines.

The 32 FastSAM prototypes for one real image, plus the 12 fixed geometric
functions. These are the basis. The head only predicts coefficients over them,
which is the whole reason it can describe a region that contains no object.
"""
import sys, glob, io
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(ROOT))
sys.path.insert(0,"/Users/amritha/DATASCIENCE FINAL/report")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from src.device import pick_device
from model import EditRegionModel
from model_v2 import _geometric
import figstyle as FS; FS.use()

dev=pick_device("auto")
m=EditRegionModel("best.pt", device=dev)
df=pd.read_parquet(sorted(glob.glob(str(ROOT/"data/magicbrush/data/dev-*.parquet")))[0])
src=Image.open(io.BytesIO(df.source_img.iloc[1]["bytes"])).convert("RGB")
ins=str(df.instruction.iloc[1])
im=torch.from_numpy(np.asarray(src.resize((640,640))).astype(np.float32)/255.
                    ).permute(2,0,1)[None].to(dev)
with torch.no_grad(): proto,_=m.backbone(im)
P=proto[0].float().cpu().numpy(); G=_geometric().numpy()

fig=plt.figure(figsize=(13.2,7.4))

# input, top left
gsA=GridSpec(1,1,left=0.015,right=0.175,top=0.855,bottom=0.565,figure=fig)
ax=fig.add_subplot(gsA[0]); ax.imshow(src); FS.nogrid(ax)
ax.set_title("Input image",fontsize=9,color=FS.INK,loc="left",pad=5)
ax.text(0,-0.06,f'"{ins[:38]}"',transform=ax.transAxes,fontsize=7.8,
        color=FS.MUTED,va="top",style="italic")

# 32 prototypes
gsB=GridSpec(4,8,left=0.225,right=0.988,top=0.855,bottom=0.365,
             hspace=0.06,wspace=0.06,figure=fig)
for i in range(32):
    ax=fig.add_subplot(gsB[i//8,i%8])
    v=P[i]; lim=np.abs(v).max()+1e-6
    ax.imshow(v,cmap="RdBu_r",vmin=-lim,vmax=lim); FS.nogrid(ax)
    ax.text(0.05,0.93,str(i),transform=ax.transAxes,fontsize=6.4,color=FS.INK,va="top",
            bbox=dict(fc="white",ec="none",alpha=0.72,pad=0.7))
fig.text(0.225,0.875,"The 32 FastSAM prototypes for this image  ·  learned, frozen, image-dependent",
         fontsize=9,color=FS.INK,weight="bold",va="bottom")

# 12 geometric
gsC=GridSpec(2,6,left=0.225,right=0.80,top=0.285,bottom=0.035,
             hspace=0.14,wspace=0.06,figure=fig)
names=["1","x","y","x²","y²","xy","x³","y³","sin πx","sin πy","cos πx","cos πy"]
for i in range(12):
    ax=fig.add_subplot(gsC[i//6,i%6])
    v=G[i]; lim=np.abs(v).max()+1e-6
    ax.imshow(v,cmap="PuOr_r",vmin=-lim,vmax=lim); FS.nogrid(ax)
    ax.set_title(names[i],fontsize=7.6,color=FS.MUTED,loc="center",pad=2)
fig.text(0.225,0.305,"The 12 geometric functions  ·  fixed, closed form, zero parameters",
         fontsize=9,color=FS.INK,weight="bold",va="bottom")

fig.text(0.015,0.45,"mask  =  σ( Σᵢ cᵢ(x,y) · basisᵢ  +  b )",fontsize=11,color=FS.INK,
         weight="bold",va="top")
for k,t in enumerate([
    "The basis is not learned per instruction.",
    "Only the coefficients c are, and they come",
    "from the text through the head.",
    "",
    "Because the prototypes are basis functions",
    "and not objects, a text-driven combination",
    "can describe empty space. That is what",
    "makes insertion possible at all.",
    "",
    "Note that no prototype is the answer to this",
    "instruction. The region a river would occupy",
    "is not any one of them, but a combination is."]):
    fig.text(0.015,0.395-k*0.0255,t,fontsize=8.4,color=FS.MUTED,va="top")

FS.figtitle(fig,"The basis the head combines",
            "32 image-dependent prototypes at 160 × 160, plus 12 fixed geometric functions. "
            "Red and blue are opposite signs.")
plt.savefig("/Users/amritha/DATASCIENCE FINAL/report/DA2_figures/basis.png",dpi=185)
print("wrote basis.png")
