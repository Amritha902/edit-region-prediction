"""Their mask-construction mechanism vs ours, controlled.

AdaptEdit's full system cannot run here: 20.43 B frozen backbone, 5.0 B
trainable, multi-GPU. But its MaskPredictor (§3.6) is a *mechanism*, and a
mechanism can be reimplemented at laptop scale and compared fairly.

Both heads receive IDENTICAL inputs — the frozen FastSAM prototypes as the
spatial feature map, the pooled image context, and the CLIP text embedding —
are trained on the same data with the same loss, schedule and seeds, and are
evaluated on the same held-out dev split. The ONLY difference is how the mask
is constructed.

  QueryGridHead  (theirs, §3.6 eq. 8)
      M = σ( Upsample( ConvHead( CA(q=γ⊙Q+β, k,v=φ(features)) ) ) )
      A learned 16×16 query grid, FiLM-modulated by the instruction,
      cross-attends to the patchified features, then a conv head decodes and
      upsamples. The mask is DECODED from learned queries.

  CoeffHead      (ours)
      M = σ( Σ cᵢ · protoᵢ + b )
      The instruction predicts 32 coefficients over the existing prototypes.
      The mask is a COMBINATION of an existing basis.

This is the comparison the project actually rests on, so it is run with the
control discipline of exp10: same capacity budget where possible, and reported
with per-kind numbers.
"""
import sys, json, argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(ROOT))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from src.device import pick_device
from model import TextCoeffHead, dice_bce_loss, mask_metrics
from train_cached import batches

P=160

class QueryGridHead(nn.Module):
    """AdaptEdit §3.6 eq.8, reimplemented at this scale."""
    def __init__(self, text_dim=512, img_dim=640, proto_c=32, d=256,
                 grid=16, heads=8, dropout=0.0):
        super().__init__()
        self.grid=grid; self.d=d
        self.queries=nn.Parameter(torch.randn(grid*grid, d)*0.02)
        self.film=nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, 2*d))
        self.kv=nn.Conv2d(proto_c, d, 1)                      # patchify features
        self.ctx=nn.Sequential(nn.LayerNorm(img_dim), nn.Linear(img_dim, d))
        self.attn=nn.MultiheadAttention(d, heads, batch_first=True, dropout=dropout)
        self.norm=nn.LayerNorm(d)
        self.conv=nn.Sequential(nn.Conv2d(d,64,3,padding=1), nn.GELU(),
                                nn.Conv2d(64,16,3,padding=1), nn.GELU(),
                                nn.Conv2d(16,1,1))
    def forward(self, text, ctx, proto):
        B=proto.shape[0]
        gb=self.film(text); g,b=gb.chunk(2,-1)
        q=self.queries[None].expand(B,-1,-1)*(1+g[:,None])+b[:,None]   # FiLM
        f=F.adaptive_avg_pool2d(proto, 20)                             # patchify
        kv=self.kv(f).flatten(2).transpose(1,2)                        # B,400,d
        kv=kv+self.ctx(ctx)[:,None]
        o,_=self.attn(q, kv, kv); o=self.norm(o+q)
        o=o.transpose(1,2).reshape(B,self.d,self.grid,self.grid)
        m=self.conv(o)
        return F.interpolate(m, size=(P,P), mode="bilinear", align_corners=False)

class CoeffWrap(nn.Module):
    """Ours, wrapped to the same call signature."""
    def __init__(self, **kw):
        super().__init__(); self.h=TextCoeffHead(**kw)
    def forward(self, text, ctx, proto):
        c,b=self.h(text,ctx)
        return (torch.einsum("bc,bchw->bhw",c,proto)+b[:,:,None]).unsqueeze(1)

def evaluate(head, D, idx, dev, zero_text=False, bs=48):
    head.eval(); io_=[]; byk={}
    with torch.no_grad():
        for sl in batches(len(idx), bs, False):
            j=idx[sl]
            pr=D["proto"][j].to(dev).float(); cx=D["ctx"][j].to(dev).float()
            t=D["text"][j].to(dev).float()
            if zero_text: t=torch.zeros_like(t)
            y=D["mask"][j].to(dev).float()
            lg=head(t,cx,pr)
            io_.append(mask_metrics(lg,y)["iou"])
            ks=[D["kind"][k] for k in j.tolist()]
            for kk in set(ks):
                s=[q for q,x in enumerate(ks) if x==kk]
                byk.setdefault(kk,[]).append(mask_metrics(lg[s],y[s])["iou"])
    head.train()
    return float(np.mean(io_)), {k:float(np.mean(v)) for k,v in byk.items()}

def run(make, name, TR, DV, dev, seeds, epochs=40):
    va=torch.arange(DV["proto"].shape[0]); out=[]
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        g=torch.Generator().manual_seed(s)
        tr=torch.randperm(TR["proto"].shape[0], generator=g)
        head=make().to(dev)
        npar=sum(p.numel() for p in head.parameters())
        opt=torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        best=(-9,0,0,None)
        for ep in range(epochs):
            for sl in batches(len(tr),48,True,g):
                j=tr[sl]
                pr=TR["proto"][j].to(dev).float(); cx=TR["ctx"][j].to(dev).float()
                t=TR["text"][j].to(dev).float(); y=TR["mask"][j].to(dev).float()
                lg=head(t,cx,pr)
                l,_,_=dice_bce_loss(lg,y,w_dice=2.0)
                opt.zero_grad(); l.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.0); opt.step()
            sch.step()
            i,byk=evaluate(head,DV,va,dev); nt,_=evaluate(head,DV,va,dev,zero_text=True)
            if i-nt>best[0]: best=(i-nt,i,nt,byk)
        d,i,nt,byk=best; out.append((i,nt,d,byk))
        print(f"    seed {s}: IoU {i:.4f}  no-text {nt:.4f}  delta {d:+.4f}  | "
              + " ".join(f"{k}:{v:.3f}" for k,v in sorted(byk.items())), flush=True)
    A=np.array([[o[0],o[1],o[2]] for o in out])
    kk={k:float(np.mean([o[3].get(k,np.nan) for o in out])) for k in out[0][3]}
    print(f"  {name}: params {npar:,} | IoU {A[:,0].mean():.4f} ± {A[:,0].std(ddof=1):.4f} "
          f"| delta {A[:,2].mean():+.4f}\n")
    return dict(name=name, params=int(npar), iou=A[:,0].mean(), iou_sd=A[:,0].std(ddof=1),
                notext=A[:,1].mean(), delta=A[:,2].mean(), delta_sd=A[:,2].std(ddof=1), by_kind=kk)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",type=int,default=5); a=ap.parse_args()
    dev=pick_device("auto")
    TR=torch.load(HERE/"cache_train.pt",map_location="cpu")
    DV=torch.load(HERE/"cache_dev.pt",map_location="cpu")
    print(f"train {TR['proto'].shape[0]} (train shards)  eval {DV['proto'].shape[0]} (dev shards)\n")
    seeds=[1368,1,2,3,4][:a.seeds]
    res=[]
    print("AdaptEdit MaskPredictor mechanism (query-grid decoder, d=256):")
    res.append(run(lambda: QueryGridHead(d=256), "query-grid d=256", TR,DV,dev,seeds))
    print("AdaptEdit mechanism, smaller (d=128):")
    res.append(run(lambda: QueryGridHead(d=128), "query-grid d=128", TR,DV,dev,seeds))
    print("Ours (prototype coefficients):")
    res.append(run(lambda: CoeffWrap(hidden=512,dropout=0.0), "coefficients h=512", TR,DV,dev,seeds))
    print("Ours, larger:")
    res.append(run(lambda: CoeffWrap(hidden=1024,dropout=0.0), "coefficients h=1024", TR,DV,dev,seeds))
    print(f"{'head':24s} {'params':>10} {'IoU':>18} {'delta':>10}  per-kind")
    for r in res:
        print(f"{r['name']:24s} {r['params']:>10,} {r['iou']:>8.4f} ± {r['iou_sd']:.4f} "
              f"{r['delta']:>+9.4f}  " + " ".join(f"{k} {v:.3f}" for k,v in sorted(r['by_kind'].items())))
    json.dump(res, open(HERE/"head_comparison.json","w"), indent=2, default=float)
    print(f"\nwrote {HERE/'head_comparison.json'}")
