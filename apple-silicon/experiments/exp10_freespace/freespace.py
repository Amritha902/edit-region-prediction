"""Does augmenting the basis with free-space prototypes raise the insertion ceiling?

exp08 measured that fitting coefficients directly to the ground-truth mask caps
insertion at IoU 0.211 against 0.458 for removal: FastSAM's 32 prototypes were
trained to segment objects and cannot span empty space. This tests the fix.

We append k extra basis functions that are NOT object-derived and re-measure the
same least-squares ceiling. Three families, so we learn *which kind* of extra
basis helps rather than only whether more basis helps:

  free    complement of the union of detected instance masks, plus its
          distance transform and coarse bands - "where no object is"
  geom    a smooth spatial polynomial basis (x, y, x², y², xy, ...) - pure
          position, no image content
  rand    k random smooth fields - the control. If rand helps as much as free,
          the gain is just extra capacity, not free-space information.

Reporting the ceiling, not a trained model, isolates representational capacity
from optimisation.
"""
import sys, json
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"datagen"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, cv2
from src.device import pick_device
from model import mask_metrics

P=160
D=torch.load(HERE/"cache_train.pt",map_location="cpu")
kinds=np.array(D["kind"])
print(f"{len(kinds)} samples: " + ", ".join(f"{k}={int((kinds==k).sum())}" for k in sorted(set(kinds))))

# --- geometric basis (shared across images) -------------------------------
yy,xx=np.mgrid[0:P,0:P]; yy=(yy/(P-1))*2-1; xx=(xx/(P-1))*2-1
GEOM=[np.ones_like(xx),xx,yy,xx*xx,yy*yy,xx*yy,xx**3,yy**3,
      np.sin(np.pi*xx),np.sin(np.pi*yy),np.cos(np.pi*xx),np.cos(np.pi*yy)]
GEOM=np.stack(GEOM).astype(np.float32)

rng=np.random.default_rng(1368)
RAND=np.stack([cv2.GaussianBlur(rng.standard_normal((P,P)).astype(np.float32),(0,0),12)
               for _ in range(12)])
RAND/= (RAND.std(axis=(1,2),keepdims=True)+1e-6)

def freespace_basis(proto):
    """Derive 'where no object is' from the prototypes themselves.

    We have no detector output in the cache, so occupancy is approximated by the
    prototype energy: regions the object-trained basis responds to strongly are
    where objects are. Its complement is candidate free space.
    """
    occ=np.abs(proto).sum(0)
    occ=(occ-occ.min())/(np.ptp(occ)+1e-6)
    free=1.0-occ
    b=cv2.GaussianBlur(free,(0,0),3)
    dist=cv2.distanceTransform((free>0.5).astype(np.uint8),cv2.DIST_L2,3)
    dist=dist/(dist.max()+1e-6)
    # coarse vertical/horizontal bands of the free region — "above / beside"
    bands=[]
    for f in (0.33,0.66):
        m=np.zeros((P,P),np.float32); m[:int(P*f),:]=1; bands.append(free*m)
        m2=np.zeros((P,P),np.float32); m2[:,:int(P*f)]=1; bands.append(free*m2)
    return np.stack([free,b,dist,occ]+bands).astype(np.float32)

def ceiling(idx, extra_fn, name, cap=220):
    out={}
    for k in sorted(set(kinds)):
        sel=[i for i in idx if kinds[i]==k][:cap]
        vals=[]
        for i in sel:
            proto=D["proto"][i].float().numpy()             # 32 x P x P
            B=[proto.reshape(32,-1)]
            ex=extra_fn(proto)
            if ex is not None: B.append(ex.reshape(ex.shape[0],-1))
            B.append(np.ones((1,P*P),np.float32))
            A=torch.from_numpy(np.concatenate(B,0)).T.double()
            y=D["mask"][i,0].float().reshape(-1)
            t=((y*2-1)*4.0).double().unsqueeze(1)
            sol=torch.linalg.lstsq(A,t).solution.squeeze(1)
            pred=(A@sol).reshape(1,1,P,P).float()
            vals.append(mask_metrics(pred, D["mask"][i][None].float())["iou"])
        out[k]=float(np.mean(vals))
    return out

allidx=list(range(len(kinds)))
print(f"\n{'basis':28s} {'extra':>6} {'insert':>9} {'modify':>9} {'remove':>9}")
rows={}
for name,fn,ne in [
    ("32 prototypes (baseline)", lambda p: None, 0),
    ("+ 12 geometric",           lambda p: GEOM, 12),
    ("+ 12 random smooth (ctrl)",lambda p: RAND, 12),
    ("+ 8 free-space",           freespace_basis, 8),
    ("+ 8 free-space + 12 geom", lambda p: np.concatenate([freespace_basis(p),GEOM]), 20),
]:
    r=ceiling(allidx,fn,name); rows[name]=dict(extra=ne,**r)
    print(f"{name:28s} {ne:>6} {r['insert']:>9.4f} {r['modify']:>9.4f} {r['remove']:>9.4f}", flush=True)

b=rows["32 prototypes (baseline)"]
print("\nchange in the INSERT ceiling vs baseline:")
for n,r in rows.items():
    if n.startswith("32"): continue
    print(f"  {n:28s} {r['insert']-b['insert']:+.4f}  ({r['insert']/b['insert']:.2f}×)")
json.dump(rows, open(HERE/"freespace.json","w"), indent=2)
print(f"\nwrote {HERE/'freespace.json'}")
