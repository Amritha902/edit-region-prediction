"""Side-by-side technical plots: 2,278 vs 8,306 samples, v1 vs v2.

Only clean-protocol numbers appear here -- epoch chosen on a held-out slice of
train, dev touched once per seed, same 503 dev turns throughout. scaling.json is
deliberately NOT plotted alongside: it used cache-val IoU at 160x160 over 40
epochs, so its values are not on this axis.
"""
import json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _s; _s.path.insert(0,"/Users/amritha/DATASCIENCE FINAL/report")
import figstyle as FS; FS.use()
from scipy import stats

# exp08 fitted these by least squares. Use the DEV recomputation from
# exp12/fix_checks.json, not exp08's train-set numbers: our IoU is measured on
# dev, and the remove ceiling differs there (0.4325 vs 0.4575).
CEIL={"remove":0.4324874186422676,"modify":0.3866544426453545,
      "insert":0.21188534100485198}
CLIPSEG, HUMAN = 0.1849, 0.1511
C={"v1":"#C65A16","v2":"#1F6FB2","ceil":"#C9C5BC","ref":"#6B6862"}

def load():
    g={}
    old=json.load(open(HERE/"clean_eval.json"))
    for r in old:
        if r["name"].startswith("v1"): g[("v1",2278)]=(r["seeds"], r["by_kind"])
        if r["name"]=="v2 spatial":    g[("v2",2278)]=(r["seeds"], r["by_kind"])
    # The completed run is authoritative. partial_*.json is only a crash
    # checkpoint, and a later run with the same config name overwrites it --
    # the 3-seed curve run clobbered the 12-seed v2 partial exactly that way.
    fin=HERE/"clean_eval_full.json"
    if fin.exists():
        for r in json.load(open(fin)):
            tag="v2" if r["name"].startswith("v2") else "v1"
            g[(tag,r["n_train"])]=(r["seeds"], r["by_kind"])
    for tag,f in (("v2","partial_clean_eval_full_v2.json"),
                  ("v1","partial_clean_eval_full_v1.json")):
        p=HERE/f
        if not p.exists(): continue
        d=json.load(open(p))
        if len(d["iou"])<2 or (tag,d["n_train"]) in g: continue
        bk={k:float(np.mean([b[k] for b in d["by_kind"] if k in b])) for k in CEIL}
        g[(tag,d["n_train"])]=(d["iou"], bk)
    return g

G=load()
keys=[k for k in [("v1",2278),("v2",2278),("v1",8306),("v2",8306)] if k in G]
print("have:", [(k[0],k[1],len(G[k][0])) for k in keys])

fig=plt.figure(figsize=(15.5,9.2))
gs=fig.add_gridspec(2,3,hspace=0.42,wspace=0.30,
                    left=0.055,right=0.985,top=0.90,bottom=0.075)

# --- A: per-seed distribution, side by side -------------------------------
ax=fig.add_subplot(gs[0,:2])
rng=np.random.default_rng(0)
for i,k in enumerate(keys):
    v=np.array(G[k][0]); tag,n=k
    x=i+rng.uniform(-0.13,0.13,len(v))
    ax.scatter(x,v,s=34,color=C[tag],alpha=0.72,zorder=3,
               edgecolor="white",linewidth=0.6)
    m,sd=v.mean(),v.std(ddof=1)
    ax.errorbar(i,m,yerr=sd,fmt="_",color="black",markersize=30,
                capsize=7,lw=1.9,zorder=4)
    ax.text(i,m+sd+0.0055,f"{m:.4f}",ha="center",fontsize=10.5,fontweight="bold")
ax.axhline(CLIPSEG,color=C["ref"],ls="--",lw=1.3)
ax.text(-0.46,CLIPSEG+0.0017,"CLIPSeg 0.1849 (150.7M params)",
        fontsize=9,color=C["ref"],ha="left")
ax.axhline(HUMAN,color=C["ref"],ls=":",lw=1.3)
ax.text(-0.46,HUMAN+0.0017,"MagicBrush human masks 0.1511",
        fontsize=9,color=C["ref"],ha="left")
ax.set_xticks(range(len(keys)))
ax.set_xticklabels([f"{t}\nn={n:,}" for t,n in keys],fontsize=10.5)
ax.set_ylabel("dev IoU (full resolution, 503 turns)")
ax.set_title("Dev IoU per seed, by head and training-set size",
             fontsize=12,fontweight="bold",loc="left")
ax.set_xlim(-0.5,len(keys)-0.5)
ax.grid(axis="y",alpha=0.25); ax.set_axisbelow(True)

# --- B: significance ------------------------------------------------------
ax=fig.add_subplot(gs[0,2]); ax.axis("off")
rows=[]
if ("v2",8306) in G and ("v2",2278) in G:
    a,b=np.array(G[("v2",8306)][0]),np.array(G[("v2",2278)][0])
    t,p=stats.ttest_ind(a,b,equal_var=False)
    rows.append(("v2: 8,306 vs 2,278",f"{a.mean()-b.mean():+.4f}",f"{p:.1e}"))
if ("v2",8306) in G and ("v1",8306) in G:
    a,b=np.array(G[("v2",8306)][0]),np.array(G[("v1",8306)][0])
    t,p=stats.ttest_ind(a,b,equal_var=False)
    rows.append(("v2 vs v1 @ 8,306",f"{a.mean()-b.mean():+.4f}",f"{p:.1e}"))
if ("v2",8306) in G:
    a=np.array(G[("v2",8306)][0])
    for nm,ref in (("vs CLIPSeg",CLIPSEG),("vs human masks",HUMAN)):
        t,p=stats.ttest_1samp(a,ref)
        rows.append((f"v2 @8,306 {nm}",f"{a.mean()-ref:+.4f}",f"{p:.1e}"))
ax.text(0,1.0,"Significance",fontsize=12,fontweight="bold",va="top")
y=0.86
ax.text(0.0,y,"comparison",fontsize=9.5,color="#555"); ax.text(0.62,y,"Δ IoU",fontsize=9.5,color="#555")
ax.text(0.855,y,"p",fontsize=9.5,color="#555"); y-=0.085
for nm,d,p in rows:
    ax.text(0.0,y,nm,fontsize=10)
    ax.text(0.62,y,d,fontsize=10,fontweight="bold",
            color="#1a7f45" if d.startswith("+") else "#b3261e")
    ax.text(0.855,y,p,fontsize=10); y-=0.105
ax.text(0,y-0.02,"Welch t-test on seed means;\none-sample against fixed baselines.",
        fontsize=8.6,color="#666",va="top")

# --- C: per-kind vs ceiling ----------------------------------------------
ax=fig.add_subplot(gs[1,:2])
kinds=["insert","modify","remove"]; w=0.26
xs=np.arange(len(kinds))
show=[k for k in [("v2",2278),("v2",8306)] if k in G]
for i,k in enumerate(show):
    tag,n=k; bk=G[k][1]
    v=[bk[kk] for kk in kinds]
    ax.bar(xs+(i-(len(show)-1)/2)*w,v,w*0.92,
           color=C[tag],alpha=0.55 if n==2278 else 1.0,
           label=f"{tag}, n={n:,}",zorder=3)
    for x,val in zip(xs+(i-(len(show)-1)/2)*w,v):
        ax.text(x,val+0.006,f"{val:.3f}",ha="center",fontsize=9)
for x,kk in zip(xs,kinds):
    ax.plot([x-0.37,x+0.37],[CEIL[kk]]*2,color=C["ceil"],lw=2.4,zorder=4)
    ax.text(x,CEIL[kk]+0.008,f"ceiling {CEIL[kk]:.3f}",fontsize=9,
            color="#5b6470",ha="center",va="bottom")
ax.set_xticks(xs); ax.set_xticklabels(kinds,fontsize=11)
ax.set_ylabel("dev IoU"); ax.set_ylim(0,0.52)
ax.set_title("Dev IoU by edit kind, against the least-squares ceiling",
             fontsize=12,fontweight="bold",loc="left")
ax.legend(frameon=False,fontsize=9.5,loc="upper left"); ax.grid(axis="y",alpha=0.25)
ax.set_axisbelow(True)

# --- D: fraction of ceiling ----------------------------------------------
ax=fig.add_subplot(gs[1,2])
for i,k in enumerate(show):
    tag,n=k; bk=G[k][1]
    f=[bk[kk]/CEIL[kk]*100 for kk in kinds]
    ax.barh(np.arange(len(kinds))+(i-(len(show)-1)/2)*0.34,f,0.31,
            color=C[tag],alpha=0.55 if n==2278 else 1.0,
            label=f"n={n:,}",zorder=3)
    for y_,val in zip(np.arange(len(kinds))+(i-(len(show)-1)/2)*0.34,f):
        ax.text(val+1.2,y_,f"{val:.0f}%",va="center",fontsize=9)
ax.set_yticks(range(len(kinds))); ax.set_yticklabels(kinds,fontsize=10.5)
ax.set_xlabel("% of the ceiling reached"); ax.set_xlim(0,80)
ax.set_title("Fraction of the ceiling reached",fontsize=12,fontweight="bold",loc="left")
ax.legend(frameon=False,fontsize=9); ax.grid(axis="x",alpha=0.25); ax.set_axisbelow(True)

fig.suptitle("Stage 1: dev IoU at 2,278 and 8,306 training samples",
             fontsize=15,fontweight="bold",x=0.055,ha="left",y=0.965)
out=ROOT/"implementation/results_full.png"; out.parent.mkdir(exist_ok=True)
fig.savefig(out,dpi=155,bbox_inches="tight",facecolor="white")
print("wrote",out)
