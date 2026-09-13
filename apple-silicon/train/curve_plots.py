"""Learning curves and the language ablation, v1 vs v2 at n=8,306.

The headline finding here is that the best epoch is 2-7 out of 60: validation
IoU peaks almost immediately and then decays. Selecting the checkpoint on a
held-out slice of train is what kept every reported number honest -- training
to the last epoch would have reported roughly half the IoU.
"""
import json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D=json.load(open(HERE/"curves_full.json"))
CFG={"v2 spatial":("#C2603F","v2 spatial field"),
     "v1 global coeff":("#5B7DB1","v1 global coefficients")}

def stack(cfg,key):
    rs=[r for r in D if r["config"]==cfg]
    return np.array([[e[key] for e in r["curve"]] for r in rs]), rs

fig=plt.figure(figsize=(15,8.6))
gs=fig.add_gridspec(2,2,hspace=0.34,wspace=0.22,
                    left=0.06,right=0.985,top=0.90,bottom=0.075)
ep=np.arange(1,61)

def band(ax,cfg,key,lw=2.0):
    A,rs=stack(cfg,key); c,lab=CFG[cfg]
    m,s=A.mean(0),A.std(0,ddof=1)
    ax.plot(ep,m,color=c,lw=lw,label=lab,zorder=3)
    ax.fill_between(ep,m-s,m+s,color=c,alpha=0.16,lw=0,zorder=2)
    return A,rs,m

# --- 1: validation IoU, full 60 epochs ---
ax=fig.add_subplot(gs[0,0])
for cfg in CFG:
    A,rs,m=band(ax,cfg,"val_iou")
    c,_=CFG[cfg]
    for r in rs:
        b=r["best_epoch"]
        ax.plot(b,r["curve"][b-1]["val_iou"],"v",color=c,ms=9,zorder=5,
                markeredgecolor="white",markeredgewidth=0.8)
ax.set_xlabel("epoch"); ax.set_ylabel("validation IoU (held-out train slice)")
ax.set_title("Validation IoU peaks in the first few epochs, then decays",
             fontsize=12,fontweight="bold",loc="left")
ax.legend(frameon=False,fontsize=10); ax.grid(alpha=0.25); ax.set_axisbelow(True)
ax.annotate("▼ = epoch actually selected",xy=(0.97,0.93),xycoords="axes fraction",
            ha="right",fontsize=9.5,color="#444")

# --- 2: zoom, first 15 epochs ---
ax=fig.add_subplot(gs[0,1])
for cfg in CFG:
    A,rs,m=band(ax,cfg,"val_iou")
    c,_=CFG[cfg]
    for r in rs:
        b=r["best_epoch"]
        ax.plot(b,r["curve"][b-1]["val_iou"],"v",color=c,ms=9,zorder=5,
                markeredgecolor="white",markeredgewidth=0.8)
ax.set_xlim(0.5,15.5)
lo=min(stack(c,"val_iou")[0][:,:15].min() for c in CFG)
hi=max(stack(c,"val_iou")[0][:,:15].max() for c in CFG)
ax.set_ylim(lo-0.004,hi+0.006)
ax.set_xlabel("epoch"); ax.set_ylabel("validation IoU")
ax.set_title("Zoom: epochs 1–15, where every checkpoint was chosen",
             fontsize=12,fontweight="bold",loc="left")
ax.grid(alpha=0.25); ax.set_axisbelow(True)

# --- 3: training loss ---
ax=fig.add_subplot(gs[1,0])
for cfg in CFG: band(ax,cfg,"loss")
ax.set_xlabel("epoch"); ax.set_ylabel("training loss (BCE + 2×Dice)")
ax.set_title("Training loss keeps falling while validation IoU falls too — overfitting",
             fontsize=12,fontweight="bold",loc="left")
ax.legend(frameon=False,fontsize=10); ax.grid(alpha=0.25); ax.set_axisbelow(True)

# --- 4: language ablation ---
ax=fig.add_subplot(gs[1,1])
for cfg in CFG:
    c,lab=CFG[cfg]
    A,_=stack(cfg,"val_iou"); N,_=stack(cfg,"val_iou_no_text")
    ax.plot(ep,A.mean(0),color=c,lw=2.0,label=f"{lab} — with instruction")
    ax.plot(ep,N.mean(0),color=c,lw=1.6,ls="--",alpha=0.8,
            label=f"{lab} — instruction zeroed")
ax.set_xlabel("epoch"); ax.set_ylabel("validation IoU")
ax.set_title("Language ablation: the instruction carries real signal throughout",
             fontsize=12,fontweight="bold",loc="left")
ax.legend(frameon=False,fontsize=8.8); ax.grid(alpha=0.25); ax.set_axisbelow(True)

bes={c:[r["best_epoch"] for r in D if r["config"]==c] for c in CFG}
fig.suptitle("Stage 1 learning curves, n=8,306 — 3 seeds per head, 60 epochs  "
             f"(best epoch: v2 {bes['v2 spatial']}, v1 {bes['v1 global coeff']})",
             fontsize=14,fontweight="bold",x=0.06,ha="left",y=0.963)
out=ROOT/"implementation/curves.png"
fig.savefig(out,dpi=155,bbox_inches="tight",facecolor="white")
print("wrote",out)
for cfg in CFG:
    A,_=stack(cfg,"val_iou"); N,_=stack(cfg,"val_iou_no_text")
    b=[r["best_epoch"] for r in D if r["config"]==cfg]
    print(f"{cfg:18s} best_ep {b}  peak {A.max(1).mean():.4f}  "
          f"ep60 {A[:,-1].mean():.4f}  delta@peak {(A-N).max(1).mean():+.4f}")
