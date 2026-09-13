"""The full metric suite a segmentation result is normally judged on.

The report so far quotes IoU alone. IoU hides the precision/recall trade-off
entirely -- two models with the same IoU can behave completely differently, one
timid and one over-eager -- and 0.5 is an arbitrary operating point. This
evaluates every saved seed checkpoint on the same 503 dev turns and reports
precision, recall, F1/Dice and IoU, a threshold sweep, area calibration, and
per-kind breakdowns, with 95% CIs and Cohen's d alongside the p-values.
"""
import glob, io, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"datagen"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, cv2, pandas as pd
from PIL import Image
from scipy import stats
from src.device import pick_device
from model import TextCoeffHead
from model_v2 import SpatialCoeffHead
import mask_extraction as ME

dev=pick_device("auto")
DV=torch.load(HERE/"cache_dev.pt",map_location="cpu")
THRS=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

FULL=[]
for f in sorted(glob.glob(str(ROOT/"data/magicbrush/data/dev-*.parquet"))):
    df=pd.read_parquet(f)
    for i in range(len(df)):
        src=Image.open(io.BytesIO(df.source_img.iloc[i]["bytes"])).convert("RGB")
        tgt=Image.open(io.BytesIO(df.target_img.iloc[i]["bytes"])).convert("RGB").resize(src.size)
        o=cv2.cvtColor(np.asarray(src),cv2.COLOR_RGB2BGR); e=cv2.cvtColor(np.asarray(tgt),cv2.COLOR_RGB2BGR)
        gt=ME.clean_mask(((ME.delta_e(o,e)>ME.DELTA_E_THRESHOLD)*255).astype(np.uint8))>0
        if 0.004<=float(gt.mean())<=0.45: FULL.append(gt)
print(f"dev turns {len(FULL)}", flush=True)

class V1(torch.nn.Module):
    def __init__(s,**kw):
        super().__init__(); s.h=TextCoeffHead(**kw)
    def forward(s,t,c,p):
        co,b=s.h(t,c); return (torch.einsum("bc,bchw->bhw",co,p)+b[:,:,None]).unsqueeze(1)

def probs(head):
    """Full-resolution sigmoid maps for the whole dev split."""
    out=[]
    head.eval()
    with torch.no_grad():
        for s in range(0,DV["proto"].shape[0],32):
            j=torch.arange(s,min(s+32,DV["proto"].shape[0]))
            lg=head(DV["text"][j].to(dev).float(), DV["ctx"][j].to(dev).float(),
                    DV["proto"][j].to(dev).float())
            pr=torch.sigmoid(lg)[:,0].cpu().numpy()
            for q,ix in enumerate(j.tolist()):
                out.append(cv2.resize(pr[q], FULL[ix].shape[::-1]))
    return out

def score(P, thr):
    """Micro-averaged over turns: mean of per-image rates, matching the IoU convention."""
    io_=[];pr_=[];re_=[];f1_=[];pa=[];ta=[];byk={}
    for ix,(p_,g) in enumerate(zip(P,FULL)):
        p=p_>thr
        tp=float((p&g).sum()); fp=float((p&~g).sum()); fn=float((~p&g).sum())
        iou=tp/max(tp+fp+fn,1); prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1)
        f1=2*prec*rec/max(prec+rec,1e-9)
        io_.append(iou); pr_.append(prec); re_.append(rec); f1_.append(f1)
        pa.append(float(p.mean())); ta.append(float(g.mean()))
        byk.setdefault(DV["kind"][ix],[]).append((iou,prec,rec,f1))
    bk={k:dict(zip(("iou","precision","recall","f1"),
                   np.array(v).mean(0).tolist())) for k,v in byk.items()}
    return dict(iou=float(np.mean(io_)), precision=float(np.mean(pr_)),
                recall=float(np.mean(re_)), f1=float(np.mean(f1_)),
                pred_area=float(np.mean(pa)), true_area=float(np.mean(ta)),
                by_kind=bk)

RES={}
for tag,make in (("v2",lambda: SpatialCoeffHead(basis="none")),
                 ("v1",lambda: V1(hidden=512))):
    cks=sorted(glob.glob(str(HERE/f"head_full_{tag}_s*.pt")))
    per_seed=[]; sweep={t:[] for t in THRS}
    for c in cks:
        st=torch.load(c,map_location=dev)["head"]
        head=make().to(dev); head.load_state_dict(st)
        P=probs(head)
        per_seed.append(score(P,0.5))
        for t in THRS: sweep[t].append(score(P,t))
        print(f"  {tag} {Path(c).stem[-5:]}: IoU {per_seed[-1]['iou']:.4f} "
              f"P {per_seed[-1]['precision']:.4f} R {per_seed[-1]['recall']:.4f} "
              f"F1 {per_seed[-1]['f1']:.4f}", flush=True)
    RES[tag]=dict(n_seeds=len(cks), per_seed=per_seed,
                  sweep={str(t):{k:float(np.mean([s[k] for s in v]))
                                 for k in ("iou","precision","recall","f1","pred_area")}
                         for t,v in sweep.items()},
                  true_area=float(np.mean([s["true_area"] for s in per_seed])))

def agg(tag,key):
    return np.array([s[key] for s in RES[tag]["per_seed"]])

print("\n" + "="*76)
print(f"{'':14s} {'IoU':>18s} {'Precision':>18s} {'Recall':>18s} {'F1/Dice':>12s}")
for tag in ("v2","v1"):
    row=[]
    for k in ("iou","precision","recall","f1"):
        a=agg(tag,k); ci=stats.t.interval(0.95,len(a)-1,a.mean(),stats.sem(a))
        row.append(f"{a.mean():.4f}±{a.std(ddof=1):.4f}" if k!="f1" else f"{a.mean():.4f}")
        RES[tag].setdefault("summary",{})[k]=dict(mean=float(a.mean()),
            sd=float(a.std(ddof=1)), ci_lo=float(ci[0]), ci_hi=float(ci[1]))
    print(f"{tag:14s} {row[0]:>18s} {row[1]:>18s} {row[2]:>18s} {row[3]:>12s}")

print("\n95% CI on the mean (t, 11 df):")
for tag in ("v2","v1"):
    s=RES[tag]["summary"]
    print(f"  {tag}: IoU [{s['iou']['ci_lo']:.4f}, {s['iou']['ci_hi']:.4f}]  "
          f"P [{s['precision']['ci_lo']:.4f}, {s['precision']['ci_hi']:.4f}]  "
          f"R [{s['recall']['ci_lo']:.4f}, {s['recall']['ci_hi']:.4f}]")

a,b=agg("v2","iou"),agg("v1","iou")
d=(a.mean()-b.mean())/np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
t,p=stats.ttest_ind(a,b,equal_var=False)
print(f"\nv2 vs v1: diff {a.mean()-b.mean():+.4f}  p={p:.2e}  Cohen's d={d:.2f}")
RES["v2_vs_v1"]=dict(diff=float(a.mean()-b.mean()),p=float(p),cohens_d=float(d))

print("\nthreshold sweep (v2, mean over 12 seeds):")
print(f"  {'thr':>5s} {'IoU':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'pred area':>10s}")
for t in THRS:
    s=RES["v2"]["sweep"][str(t)]
    print(f"  {t:>5.1f} {s['iou']:>8.4f} {s['precision']:>8.4f} {s['recall']:>8.4f} "
          f"{s['f1']:>8.4f} {s['pred_area']:>9.2%}")
print(f"  true area {RES['v2']['true_area']:.2%}")
best=max(THRS,key=lambda t: RES["v2"]["sweep"][str(t)]["iou"])
print(f"  best IoU at threshold {best} ({RES['v2']['sweep'][str(best)]['iou']:.4f}) "
      f"vs 0.5 ({RES['v2']['sweep']['0.5']['iou']:.4f})")
RES["best_thr_v2"]=best

print("\nper kind at 0.5 (v2, mean over seeds):")
for k in ("insert","modify","remove"):
    v=np.array([[s["by_kind"][k][m] for m in ("iou","precision","recall","f1")]
                for s in RES["v2"]["per_seed"] if k in s["by_kind"]]).mean(0)
    print(f"  {k:7s} IoU {v[0]:.4f}  P {v[1]:.4f}  R {v[2]:.4f}  F1 {v[3]:.4f}")

json.dump(RES, open(HERE/"metrics_full.json","w"), indent=2)
print("\nwrote train/metrics_full.json")
