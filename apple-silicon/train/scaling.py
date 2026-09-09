"""Does more data help? IoU and language-contribution vs training-set size."""
import json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(ROOT))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from src.device import pick_device
from model import TextCoeffHead, dice_bce_loss
from train_cached import batches, run_eval

def train_at(D, tr_i, va_i, dev, seed, epochs=40):
    torch.manual_seed(seed); np.random.seed(seed)
    head=TextCoeffHead(hidden=512,dropout=0.0).to(dev)
    opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    g=torch.Generator().manual_seed(seed); bd=-9; bi=0; bn=0; bk=None
    for ep in range(epochs):
        for sl in batches(len(tr_i),64,True,g):
            j=tr_i[sl]
            p=D["proto"][j].to(dev).float(); c=D["ctx"][j].to(dev).float()
            t=D["text"][j].to(dev).float();  y=D["mask"][j].to(dev).float()
            co,b=head(t,c)
            lg=(torch.einsum("bc,bchw->bhw",co,p)+b[:,:,None]).unsqueeze(1)
            l,_,_=dice_bce_loss(lg,y,w_dice=2.0)
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.0); opt.step()
        sch.step()
        v,byk=run_eval(head,D,va_i,dev); nt,_=run_eval(head,D,va_i,dev,zero_text=True)
        if v["iou"]-nt["iou"]>bd: bd=v["iou"]-nt["iou"]; bi=v["iou"]; bn=nt["iou"]; bk=byk
    return bi,bn,bd,bk

def main():
    dev=pick_device("auto")
    D=torch.load(HERE/"cache_all.pt",map_location="cpu")
    n=D["proto"].shape[0]
    g=torch.Generator().manual_seed(1368); perm=torch.randperm(n,generator=g)
    nv=150; va_i=perm[:nv]; pool=perm[nv:]
    sizes=[100,200,300,450,600,len(pool)]
    print(f"{n} total | fixed val {nv} | pool {len(pool)}\n")
    print(f"{'N_train':>8} {'IoU':>17} {'no-text':>17} {'delta':>17}")
    out=[]
    for N in sizes:
        if N>len(pool): continue
        I=[];T=[];Dl=[];K=[]
        for s in (1368,1,2,3,4):
            tr=pool[:N]
            i,nt,d,byk=train_at(D,tr,va_i,dev,s)
            I.append(i);T.append(nt);Dl.append(d);K.append(byk)
        I,T,Dl=map(np.array,(I,T,Dl))
        kk={k:float(np.mean([x.get(k,np.nan) for x in K])) for k in K[0]}
        out.append(dict(n=int(N),iou=I.mean(),iou_sd=I.std(ddof=1),
                        notext=T.mean(),notext_sd=T.std(ddof=1),
                        delta=Dl.mean(),delta_sd=Dl.std(ddof=1),by_kind=kk))
        print(f"{N:>8} {I.mean():>8.4f} ± {I.std(ddof=1):.4f} {T.mean():>8.4f} ± {T.std(ddof=1):.4f} "
              f"{Dl.mean():>+8.4f} ± {Dl.std(ddof=1):.4f}", flush=True)
    (HERE/"scaling.json").write_text(json.dumps(out,indent=2))

if __name__=="__main__": main()
