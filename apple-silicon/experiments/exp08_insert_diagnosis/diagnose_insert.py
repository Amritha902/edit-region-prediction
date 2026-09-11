"""Why does insertion not improve with more data?

Three hypotheses, tested in order. Each is falsifiable.

 H1  Too few samples.      insert is 157/2631 (6%). Test: train insert-only, and
                           compare against modify subsampled to the same count.
 H2  Bad targets.          deltaE may mis-recover insertion regions. Test: measure
                           target statistics per kind (size, fragmentation, and
                           whether the region even lands where an object is not).
 H3  Basis cannot express it. The 32 prototypes are trained for objects. Test: fit
                           coefficients DIRECTLY to each ground-truth mask by least
                           squares - the best any coefficient predictor could do.
                           If this ceiling is low for insert, the basis is the wall.
"""
import sys, json
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(ROOT))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from src.device import pick_device
from model import TextCoeffHead, dice_bce_loss, mask_metrics
from train_cached import batches, run_eval

dev=pick_device("auto")
D=torch.load(HERE/"cache_train.pt",map_location="cpu")
kinds=np.array(D["kind"]); n=len(kinds)
print(f"{n} samples: " + ", ".join(f"{k}={int((kinds==k).sum())}" for k in sorted(set(kinds))))

# ---------------------------------------------------------------- H2 targets
print("\n=== H2  target statistics by kind ===")
print(f"{'kind':8s} {'n':>5} {'area':>8} {'blobs':>7} {'largest':>8} {'edge%':>7}")
import cv2
stats={}
for k in sorted(set(kinds)):
    idx=np.where(kinds==k)[0]
    ar=[];bl=[];lg=[];eg=[]
    for i in idx[:400]:
        m=D["mask"][i,0].numpy().astype(np.uint8)
        a=m.mean(); ar.append(a)
        nlab,lab,st,_=cv2.connectedComponentsWithStats(m,connectivity=8)
        bl.append(nlab-1)
        lg.append(st[1:,cv2.CC_STAT_AREA].max()/max(m.sum(),1) if nlab>1 else 0)
        er=cv2.erode(m,np.ones((3,3),np.uint8))
        eg.append((m-er).sum()/max(m.sum(),1))
    stats[k]=dict(area=float(np.mean(ar)),blobs=float(np.mean(bl)),
                  largest=float(np.mean(lg)),edge=float(np.mean(eg)))
    print(f"{k:8s} {len(idx):>5} {np.mean(ar):>7.2%} {np.mean(bl):>7.2f} "
          f"{np.mean(lg):>7.2%} {np.mean(eg):>6.1%}")

# ---------------------------------------------------------------- H3 ceiling
print("\n=== H3  basis ceiling: least-squares fit of coefficients to the TRUE mask ===")
print("    (the best IoU ANY coefficient predictor could achieve with these 32 prototypes)")
print(f"{'kind':8s} {'n':>5} {'ceiling IoU':>12} {'ceiling Dice':>13}")
ceil={}
for k in sorted(set(kinds)):
    idx=np.where(kinds==k)[0][:250]
    ious=[];dices=[]
    for i in idx:
        P=D["proto"][i].float().reshape(32,-1)          # 32 x (160*160)
        y=D["mask"][i,0].float().reshape(-1)
        A=torch.cat([P,torch.ones(1,P.shape[1])],0).T    # add bias column
        # logistic target: fit in logit space via ridge least squares
        t=(y*2-1)*4.0
        sol=torch.linalg.lstsq(A.double(), t.double().unsqueeze(1)).solution.squeeze(1)
        pred=(A.double()@sol).reshape(1,1,160,160)
        m=mask_metrics(pred.float(), D["mask"][i][None].float())
        ious.append(m["iou"])
        p=(torch.sigmoid(pred)>0.5).float(); g=D["mask"][i][None].float()
        dices.append((2*(p*g).sum()/(p.sum()+g.sum()).clamp(min=1)).item())
    ceil[k]=dict(iou=float(np.mean(ious)),dice=float(np.mean(dices)))
    print(f"{k:8s} {len(idx):>5} {np.mean(ious):>12.4f} {np.mean(dices):>13.4f}")

# ---------------------------------------------------------------- H1 data
print("\n=== H1  is it just sample count? train per-kind at matched n ===")
def train_subset(idx, seed=1368, epochs=40):
    torch.manual_seed(seed); np.random.seed(seed)
    g=torch.Generator().manual_seed(seed)
    perm=torch.randperm(len(idx),generator=g)
    idx=torch.tensor(idx)[perm]
    nv=max(15,int(len(idx)*0.2)); va,tr=idx[:nv],idx[nv:]
    head=TextCoeffHead(hidden=512,dropout=0.0).to(dev)
    opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    best=-9;bi=0
    for ep in range(epochs):
        for sl in batches(len(tr),64,True,g):
            j=tr[sl]
            p=D["proto"][j].to(dev).float(); c=D["ctx"][j].to(dev).float()
            t=D["text"][j].to(dev).float(); y=D["mask"][j].to(dev).float()
            co,b=head(t,c)
            lg=(torch.einsum("bc,bchw->bhw",co,p)+b[:,:,None]).unsqueeze(1)
            l,_,_=dice_bce_loss(lg,y,w_dice=2.0); opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.0); opt.step()
        sch.step()
        v,_=run_eval(head,D,va,dev); nt,_=run_eval(head,D,va,dev,zero_text=True)
        if v["iou"]-nt["iou"]>best: best=v["iou"]-nt["iou"]; bi=v["iou"]
    return bi,best,len(tr),len(va)

n_ins=int((kinds=="insert").sum())
print(f"{'setting':28s} {'n_train':>8} {'IoU':>8} {'delta':>8}")
for name,idx in [("insert only", np.where(kinds=="insert")[0]),
                 ("modify @ same n", np.where(kinds=="modify")[0][:n_ins]),
                 ("remove @ same n", np.where(kinds=="remove")[0][:n_ins])]:
    if len(idx)<40: print(f"{name:28s}  too few ({len(idx)})"); continue
    i,d,ntr,nva=train_subset(idx)
    print(f"{name:28s} {ntr:>8} {i:>8.4f} {d:>+8.4f}")

json.dump(dict(target_stats=stats, basis_ceiling=ceil), open(HERE/"diagnosis.json","w"), indent=2)
print(f"\nwrote {HERE/'diagnosis.json'}")
