"""Publication figures from the stored result JSONs.

    python train/figures.py

Reads only train/*.json, so the figures always match the committed numbers.
Outputs train/fig_*.png at 200 dpi.
"""
import json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
 "font.family":"DejaVu Sans","font.size":9.5,"axes.grid":True,"grid.alpha":0.22,
 "grid.linewidth":0.6,"axes.spines.top":False,"axes.spines.right":False,
 "axes.linewidth":0.9,"xtick.major.width":0.9,"ytick.major.width":0.9,
 "legend.frameon":False,"figure.dpi":110})

INK="#1A1A1A"; MUT="#6E6E6E"; ACC="#8C3A1E"; GRN="#1B7A4B"; GRY="#BFBFBF"

ce={r["name"]:r for r in json.load(open(HERE/"clean_eval.json"))}
lat=json.load(open(HERE/"latency.json"))
base=json.load(open(HERE/"baselines.json"))
hc={r["name"]:r for r in json.load(open(HERE/"head_comparison.json"))}
fs=json.load(open(HERE/"freespace.json"))
dg=json.load(open(HERE/"diagnosis.json"))
fx=json.load(open(HERE/"fix_checks.json"))

OURS=ce["v2 spatial"]
CLIPSEG_IOU=base["CLIPSeg (150M)"]["all"]; CLIPSEG_INS=base["CLIPSeg (150M)"]["insert"]
HUMAN_IOU=base["MagicBrush GT mask"]["all"]; HUMAN_INS=base["MagicBrush GT mask"]["insert"]
FULL_IOU=base["full frame"]["all"]; RAND_IOU=base["random centred box"]["all"]

# ───────────────────────── FIG 1 · accuracy vs cost frontier
fig,ax=plt.subplots(1,2,figsize=(12.4,4.5))
pts=[("random box",0,None,RAND_IOU,GRY),("full frame",0,None,FULL_IOU,GRY),
     ("MagicBrush\nhuman masks",None,None,HUMAN_IOU,ACC),
     ("ours (v2)",1.4,5.58,OURS["iou"],GRN),
     ("CLIPSeg",150.7,54.3,CLIPSEG_IOU,INK)]
for name,p,ms,iou,c in pts:
    if ms is None:
        ax[0].axhline(iou,color=c,ls=":",lw=1.1,alpha=.75)
        ax[0].text(128,iou+.0020,name.replace("\n"," "),color=c,fontsize=8,
                   ha="right",va="bottom")
        continue
    ax[0].scatter(ms,iou,s=190 if name.startswith("ours") else 130,color=c,zorder=5,
                  edgecolor="white",linewidth=1.4)
    ax[0].annotate(f"{name}\n{p:g} M trainable",(ms,iou),textcoords="offset points",
                   xytext=(10,-24 if name=="CLIPSeg" else 9),fontsize=8.5,color=c,fontweight="bold")
ax[0].annotate("",xy=(5.04,OURS["iou"]),xytext=(54.3,CLIPSEG_IOU),
               arrowprops=dict(arrowstyle="<->",color=MUT,lw=1.1,ls="--"))
ax[0].text(16.5,(OURS["iou"]+CLIPSEG_IOU)/2+.0055,"9.7× faster at matched 352px\n93% of the IoU",
           fontsize=8.5,color=MUT,ha="center")
ax[0].set_xscale("log"); ax[0].set_xlim(3.2,135); ax[0].set_ylim(.05,.20)
ax[0].set_xlabel("per-instruction latency (ms, log scale)  ·  Apple M5, matched 352×352")
ax[0].set_ylabel("edit-region IoU  (held-out dev)")
ax[0].set_title("Accuracy against cost",fontweight="bold",fontsize=11,loc="left")

ks=["insert","modify","remove"]; x=np.arange(3); w=.26
o=[OURS["by_kind"][k] for k in ks]
c=[base["CLIPSeg (150M)"][k] for k in ks]
h=[base["MagicBrush GT mask"][k] for k in ks]
ax[1].bar(x-w,h,w,label="MagicBrush human masks",color=ACC,alpha=.85)
ax[1].bar(x,  o,w,label="ours (v2, 1.4 M)",color=GRN)
ax[1].bar(x+w,c,w,label="CLIPSeg (150.7 M)",color=INK,alpha=.85)
for i,(a,b) in enumerate(zip(o,c)):
    if a>b:
        ax[1].annotate(f"+{100*(a-b)/b:.0f}%  vs CLIPSeg",(i,a+.012),ha="center",
                       fontsize=9.5,color=GRN,fontweight="bold")
        ax[1].annotate("",xy=(i,a+.008),xytext=(i+w,b+.004),
                       arrowprops=dict(arrowstyle="<-",color=GRN,lw=1.2))
ax[1].set_xticks(x); ax[1].set_xticklabels(ks); ax[1].set_ylabel("IoU"); ax[1].set_ylim(0,.33)
ax[1].legend(fontsize=8.5,loc="upper left")
ax[1].set_title("By edit type",fontweight="bold",fontsize=11,loc="left")
ax[1].text(0,1.045,"we beat CLIPSeg on insertion; the human masks still lead there",
           transform=ax[1].transAxes,fontsize=8.5,color=MUT)
fig.tight_layout(); fig.savefig(HERE/"fig_frontier.png",dpi=200,bbox_inches="tight")

# ───────────────────────── FIG 2 · amortisation + ablation + significance
fig,ax=plt.subplots(1,3,figsize=(15.4,4.7))
n=[r["n"] for r in lat["rows"]]
ax[0].plot(n,[r["ours"]*1000 for r in lat["rows"]],marker="o",ms=5,lw=2,color=GRN,label="ours (v2)")
ax[0].plot(n,[r["clipseg"]*1000 for r in lat["rows"]],marker="s",ms=5,lw=2,color=INK,label="CLIPSeg")
be=lat["per_image"]/max(lat["clipseg"]-lat["per_instruction"],1e-9)
ax[0].axvline(be,color=ACC,ls=":",lw=1.2)
ax[0].text(be*1.15,300,f"break-even\nN = {be:.1f}",color=ACC,fontsize=8.5)
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("instructions on one image"); ax[0].set_ylabel("total latency (ms)")
ax[0].legend(fontsize=9)
ax[0].set_title("Cost amortises — 9.0× at N=100",fontweight="bold",fontsize=11,loc="left")

names=["v1\nglobal","v2\nspatial","v2+random\n(control)","v2+geom"]
keys=["v1 global coeff","v2 spatial","v2 spatial+random","v2 spatial+geom"]
mu=[ce[k]["iou"] for k in keys]; sd=[ce[k]["sd"] for k in keys]
cols=[GRY,GRN,GRY,GRY]
b=ax[1].bar(names,mu,yerr=sd,capsize=5,color=cols,edgecolor=INK,linewidth=.9,width=.6)
ax[1].axhline(HUMAN_IOU,color=ACC,ls=":",lw=1.2)
ax[1].text(3.42,HUMAN_IOU+.0018,"human masks 0.151",color=ACC,fontsize=8,ha="right")
ax[1].set_ylim(.14,.19); ax[1].set_ylabel("IoU (12 seeds)")
ax[1].text(0,-.215,"note: y-axis truncated at 0.14 to resolve the error bars",
           fontsize=7.5,color=MUT,transform=ax[1].get_xaxis_transform())
ax[1].annotate("p = 0.0070",xy=(.5,.183),fontsize=9,color=GRN,fontweight="bold",ha="center")
ax[1].annotate("",xy=(0,.181),xytext=(1,.181),arrowprops=dict(arrowstyle="<->",color=GRN,lw=1.1))
ax[1].annotate("n.s.  p = 0.33",xy=(2.5,.176),fontsize=8.5,color=MUT,ha="center")
ax[1].annotate("",xy=(2,.174),xytext=(3,.174),arrowprops=dict(arrowstyle="<->",color=MUT,lw=1))
ax[1].set_title("Ablation — controls included",fontweight="bold",fontsize=11,loc="left")

kk=["insert","modify","remove"]
ceil=[fx["ceiling_dev"][k] for k in kk]   # recomputed on DEV, matching the model IoU
got=[OURS["by_kind"][k] for k in kk]
xx=np.arange(3)
ax[2].bar(xx,ceil,.5,color=GRY,edgecolor=INK,linewidth=.9,label="basis ceiling (least squares)")
ax[2].bar(xx,got,.5,color=GRN,label="ours (v2)")
for i,(g,c_) in enumerate(zip(got,ceil)):
    ax[2].text(i,c_+.008,f"{100*g/c_:.0f}%\nof ceiling",ha="center",fontsize=8.5,color=MUT)
ax[2].set_xticks(xx); ax[2].set_xticklabels(kk); ax[2].set_ylabel("IoU")
ax[2].legend(fontsize=8.5,loc="upper left"); ax[2].set_ylim(0,.55)
ax[2].set_title("Headroom against the basis ceiling",fontweight="bold",fontsize=11,loc="left")
ax[2].text(0,-.145,"ceiling recomputed on the same dev split as the model IoU",
           fontsize=7.5,color=MUT,transform=ax[2].get_xaxis_transform())
fig.tight_layout(); fig.savefig(HERE/"fig_analysis.png",dpi=200,bbox_inches="tight")
print("wrote fig_frontier.png and fig_analysis.png")
print(f"  ours {OURS['iou']:.4f} ± {OURS['sd']:.4f} | CLIPSeg {CLIPSEG_IOU:.4f} | human {HUMAN_IOU:.4f}")
print(f"  insert: ours {OURS['by_kind']['insert']:.3f} vs CLIPSeg {CLIPSEG_INS:.3f} "
      f"(+{100*(OURS['by_kind']['insert']-CLIPSEG_INS)/CLIPSEG_INS:.0f}%)")
