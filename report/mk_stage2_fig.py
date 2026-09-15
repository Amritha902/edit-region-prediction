"""Stage 2 and cost figures. Neutral titles, direct labels, no legend boxes."""
import sys, json; sys.path.insert(0,".")
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import figstyle as FS; FS.use()
T=Path("/Users/amritha/DATASCIENCE FINAL/instruct-seg-edit-mac/train")
s2=json.load(open(T/"stage2.json")); lat=json.load(open(T/"latency_v2.json"))
mf=json.load(open(T/"metrics_full.json")); gr=json.load(open(T/"grid_final.json"))

fig=plt.figure(figsize=(13.2,8.2))
gs=fig.add_gridspec(2,3,hspace=0.66,wspace=0.32,left=0.058,right=0.982,top=0.815,bottom=0.075)

# ── A: four arms, recall vs collateral ───────────────────────────────────
ax=fig.add_subplot(gs[0,0:2])
arms=["A whole-frame","D MagicBrush mask","B our mask","C GT mask"]
lbl =["Whole frame\n(no mask)","MagicBrush\nhuman mask","Our predicted\nmask","Ground-truth\nmask"]
x=np.arange(len(arms)); w=0.36
rec=[s2[a]["recall"]*100 for a in arms]; col=[s2[a]["collateral"]*100 for a in arms]
ax.bar(x-w/2,rec,w,color=FS.CAT[0],label="recall",zorder=3)
ax.bar(x+w/2,col,w,color=FS.CAT[1],label="collateral",zorder=3)
for i,(r,c) in enumerate(zip(rec,col)):
    ax.text(i-w/2,r+1.3,f"{r:.0f}",ha="center",fontsize=8.6,color=FS.CAT[0],weight="bold")
    ax.text(i+w/2,c+1.3,f"{c:.0f}",ha="center",fontsize=8.6,color=FS.CAT[1],weight="bold")
ax.set_xticks(x); ax.set_xticklabels(lbl,fontsize=8.4)
ax.set_ylabel("% of pixels"); ax.set_ylim(0,92)
ax.text(-w/2,rec[0]+7.5,"recall",fontsize=8.6,color=FS.CAT[0],weight="bold",ha="center")
ax.text(w/2,col[0]+7.5,"collateral",fontsize=8.6,color=FS.CAT[1],weight="bold",ha="center")
FS.title(ax,"Edit recall and collateral damage, by gating strategy",
         "60 held-out samples, Stable Diffusion inpainting. Recall is how much of the intended edit landed; "
         "collateral is change outside the true region.", wrap=104)

# ── B: net ───────────────────────────────────────────────────────────────
ax=fig.add_subplot(gs[0,2])
net=[s2[a]["net"]*100 for a in arms]
cols=[FS.MUTED,FS.MUTED,FS.CAT[0],FS.CAT[2]]
b=ax.barh(np.arange(len(arms)),net,color=cols,height=0.62,zorder=3)
for i,v in enumerate(net):
    ax.text(v+1.1,i,f"{v:.1f}%",va="center",fontsize=8.8,
            color=cols[i],weight="bold")
ax.set_yticks(np.arange(len(arms)))
ax.set_yticklabels([l.replace("\n"," ") for l in lbl],fontsize=8.3)
ax.invert_yaxis(); ax.set_xlim(0,62); ax.set_xlabel("net gain, recall minus collateral")
ax.grid(False); ax.xaxis.grid(True)
FS.title(ax,"Net gain","Higher is better.")

# ── C: precision-recall frontier ─────────────────────────────────────────
ax=fig.add_subplot(gs[1,0])
sw=mf["v2"]["sweep"]; ths=sorted(sw,key=float)
pr=[sw[t]["precision"] for t in ths]; rc=[sw[t]["recall"] for t in ths]
ax.plot(rc,pr,"-o",color=FS.CAT[0],ms=4.5,zorder=3)
for t in ["0.1","0.3","0.5","0.9"]:
    ax.annotate(f"thr {t}",(sw[t]["recall"],sw[t]["precision"]),
                textcoords="offset points",xytext=(7,6),fontsize=8,color=FS.MUTED)
bi=max(ths,key=lambda t:sw[t]["iou"])
ax.plot([sw[bi]["recall"]],[sw[bi]["precision"]],"o",ms=10,mfc="none",
        mec=FS.CAT[1],mew=2,zorder=4)
ax.annotate(f"best IoU {sw[bi]['iou']:.4f}",(sw[bi]["recall"],sw[bi]["precision"]),
            textcoords="offset points",xytext=(10,-15),fontsize=8.2,
            color=FS.CAT[1],weight="bold")
ax.set_xlabel("recall"); ax.set_ylabel("precision")
FS.title(ax,"Precision and recall by threshold","v2, mean of 12 seeds.", wrap=52)

# ── D: latency ───────────────────────────────────────────────────────────
ax=fig.add_subplot(gs[1,1])
names=["CLIPSeg","Ours v1","Ours v2"]
ms=[lat["clipseg"]["per_instruction_ms"],lat["v1"]["352"]["per_instruction_ms"],
    lat["v2"]["352"]["per_instruction_ms"]]
pa=[lat["clipseg"]["params"],lat["v1"]["params"],lat["v2"]["params"]]
cols=[FS.MUTED,FS.FAINT,FS.CAT[0]]
ax.bar(names,ms,color=cols,width=0.55,zorder=3)
for i,(v,p) in enumerate(zip(ms,pa)):
    ax.text(i,v+1.4,f"{v:.1f} ms",ha="center",fontsize=8.8,color=FS.INK,weight="bold")
    ax.text(i,v+5.0,f"{p/1e6:.1f} M params",ha="center",fontsize=7.8,color=FS.MUTED)
ax.set_ylabel("ms per instruction"); ax.set_ylim(0,68)
FS.title(ax,"Inference cost per instruction",
         "Resolution matched at 352 px. The backbone runs once per image; the head runs per instruction.", wrap=52)

# ── E: grid, net vs collateral scatter ───────────────────────────────────
ax=fig.add_subplot(gs[1,2])
ths_g=[0.2,0.3,0.4,0.5]
for k,t in enumerate(ths_g):
    cells=[gr[f"{t}_{d}"] for d in [0,16,32,48,64] if f"{t}_{d}" in gr]
    ax.plot([c["collateral"]*100 for c in cells],[c["net"]*100 for c in cells],
            "-o",ms=4,color=FS.SEQ_COOL[k+2],zorder=3)
    ax.text(cells[-1]["collateral"]*100-1.2,cells[-1]["net"]*100-1.6,f"thr {t}",
            fontsize=7.8,color=FS.SEQ_COOL[k+3],va="top",ha="right")
best=gr["0.3_32"]
ax.plot([best["collateral"]*100],[best["net"]*100],"o",ms=11,mfc="none",mec=FS.CAT[1],mew=2,zorder=4)
ax.annotate("0.3 / 32 px",(best["collateral"]*100,best["net"]*100),
            textcoords="offset points",xytext=(-40,-2),fontsize=8.2,color=FS.CAT[1],weight="bold")
wf=s2["A whole-frame"]
ax.plot([wf["collateral"]*100],[wf["net"]*100],"s",ms=6,color=FS.CAT[3],zorder=4)
ax.annotate("whole frame",(wf["collateral"]*100,wf["net"]*100),
            textcoords="offset points",xytext=(6,-3),fontsize=8,color=FS.CAT[3])
ax.set_xlabel("collateral damage, %"); ax.set_ylabel("net gain, %"); ax.set_xlim(6,45)
FS.title(ax,"All 20 operating points","Each line is one threshold swept over dilation.", wrap=52)

FS.figtitle(fig,"Stage 2: does predicting the region improve the edit?",
            "Every value measured on held-out development samples with real Stable Diffusion inpainting.")
plt.savefig("DA2_figures/stage2.png",dpi=185)
print("wrote stage2.png")
