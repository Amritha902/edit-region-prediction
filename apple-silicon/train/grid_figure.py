"""Heatmap of the Stage 2 operating-point grid.

Form: magnitude across two ordinal axes -> heatmap. Sequential single-hue ramps,
light to dark, one per panel; net and collateral run in opposite directions, so
they get different hues and each axis label states which way is better. Every
cell is labelled because a 4x5 grid is small enough to read exhaustively, which
is the one case where labelling every mark is correct rather than noise.
"""
import json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

g=json.load(open(HERE/"grid_final.json"))
THRS=sorted({v["thr"] for v in g.values()}); DILS=sorted({v["dil"] for v in g.values()})
NET=np.full((len(THRS),len(DILS)),np.nan); COL=np.full_like(NET,np.nan)
for v in g.values():
    i,j=THRS.index(v["thr"]),DILS.index(v["dil"])
    NET[i,j]=v["net"]*100; COL[i,j]=v["collateral"]*100
bi,bj=np.unravel_index(np.nanargmax(NET),NET.shape)

INK="#1b1a17"; MUTE="#6b6963"
fig,axes=plt.subplots(1,2,figsize=(13.2,4.5))
for ax,M,cmap,title,sub in (
    (axes[0],NET,"Blues","Net improvement","higher is better"),
    (axes[1],COL,"Oranges","Collateral change","lower is better")):
    im=ax.imshow(M,cmap=cmap,aspect="auto",
                 vmin=np.nanmin(M)-2, vmax=np.nanmax(M)+1)
    for i in range(len(THRS)):
        for j in range(len(DILS)):
            if np.isnan(M[i,j]): continue
            # label ink follows the cell's lightness, never the series colour
            frac=(M[i,j]-(np.nanmin(M)-2))/((np.nanmax(M)+1)-(np.nanmin(M)-2))
            ax.text(j,i,f"{M[i,j]:.1f}",ha="center",va="center",fontsize=11.5,
                    color="white" if frac>0.62 else INK,
                    fontweight="bold" if (i,j)==(bi,bj) else "normal")
    ax.set_xticks(range(len(DILS))); ax.set_xticklabels([f"{d}" for d in DILS])
    ax.set_yticks(range(len(THRS))); ax.set_yticklabels([f"{t}" for t in THRS])
    ax.set_xlabel("dilation (px)", fontsize=11, color=INK)
    ax.set_ylabel("decision threshold", fontsize=11, color=INK)
    ax.set_title(f"{title}   —   {sub}", fontsize=12.5, color=INK, pad=10, loc="left")
    ax.add_patch(Rectangle((bj-0.5,bi-0.5),1,1,fill=False,edgecolor=INK,lw=2.4,zorder=5))
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0, colors=MUTE)

# Annotations go outside the cells. Inside, they collided with the values in
# the 0.2/48 and 0.5/64 cells.
axes[0].text(0.0,-0.235,"boxed cell is the optimum: threshold 0.3, dilation 32 px",
    transform=axes[0].transAxes, fontsize=10.5, color=INK, style="italic")
axes[1].text(0.0,-0.235,"whole-frame editing, for reference, is 24.2",
    transform=axes[1].transAxes, fontsize=10.5, color=MUTE, style="italic")
fig.suptitle("Stage 2 operating point, swept jointly — 20 cells, 60 held-out samples each",
             fontsize=13.5, fontweight="bold", x=0.012, ha="left", y=1.02, color=INK)
fig.tight_layout()
out=HERE.parent/"grid_heatmap.png"
fig.savefig(out,dpi=155,bbox_inches="tight",facecolor="white")
print("wrote",out)
print(f"  optimum cell: thr {THRS[bi]} dil {DILS[bj]} -> net {NET[bi,bj]:.1f}%, collateral {COL[bi,bj]:.1f}%")
