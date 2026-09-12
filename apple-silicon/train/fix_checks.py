"""Two consistency fixes the figures needed.

(1) The basis ceiling in exp08 was computed on cache_train.pt when that file was
    the MERGED 2781 (dev+train). The figure compares it against a dev IoU, so the
    ceiling must be recomputed on the dev set itself.

(2) Latency was measured with CLIPSeg at its native 352x352 and ours at 640x640
    — ours processes 3.3x more pixels. That understates our advantage, but it is
    not a matched comparison. Measure ours at 352 as well.
"""
import sys, json, time
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(ROOT))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from PIL import Image
from src.device import pick_device
from model import EditRegionModel, TextCoeffHead, mask_metrics

dev=pick_device("auto")

# ---------- (1) ceiling on dev ----------
D=torch.load(HERE/"cache_dev.pt",map_location="cpu")
kinds=np.array(D["kind"]); P=160
print("basis ceiling recomputed ON THE DEV SET (least squares to the true mask)")
print(f"{'kind':8s} {'n':>5} {'ceiling IoU':>12}")
ceil={}
for k in sorted(set(kinds)):
    idx=np.where(kinds==k)[0]
    v=[]
    for i in idx:
        A=torch.cat([D["proto"][i].float().reshape(32,-1),
                     torch.ones(1,P*P)],0).T.double()
        y=D["mask"][i,0].float().reshape(-1)
        sol=torch.linalg.lstsq(A,((y*2-1)*4.0).double().unsqueeze(1)).solution.squeeze(1)
        pred=(A@sol).reshape(1,1,P,P).float()
        v.append(mask_metrics(pred, D["mask"][i][None].float())["iou"])
    ceil[k]=float(np.mean(v))
    print(f"{k:8s} {len(idx):>5} {np.mean(v):>12.4f}")

# ---------- (2) matched-resolution latency ----------
ours=EditRegionModel("best.pt", device=dev)
ck=torch.load(HERE/"edit_region_head_clean.pt", map_location=dev)
ours.head=TextCoeffHead(hidden=512,dropout=0.0).to(dev)
ours.head.load_state_dict(ck["head"]); ours.head.eval()
img=Image.open(ROOT/"experiments/exp01_gap/images/000000000785.jpg").convert("RGB")
INSTR=["put a hat on the person","remove the skis","make the jacket blue"]*10
def sync():
    if dev=="mps": torch.mps.synchronize()
print("\nmatched-resolution latency")
res={}
for size in (352,640):
    x=torch.from_numpy(np.asarray(img.resize((size,size)))).permute(2,0,1).float().div(255)[None].to(dev)
    for _ in range(3):
        with torch.no_grad(): ours.backbone(x)
    sync(); t=time.perf_counter()
    for _ in range(10):
        with torch.no_grad(): proto,ctx=ours.backbone(x)
    sync(); pim=(time.perf_counter()-t)/10
    with torch.no_grad(): proto,ctx=ours.backbone(x)
    sync(); t=time.perf_counter()
    for s in INSTR:
        with torch.no_grad():
            e=ours.encode_text([s]); c,b=ours.head(e,ctx)
            _=(torch.einsum("bc,bchw->bhw",c,proto)+b[:,:,None])
    sync(); pin=(time.perf_counter()-t)/len(INSTR)
    res[size]=dict(per_image=pim, per_instruction=pin)
    print(f"  ours @{size}x{size}: per-image {pim*1000:6.1f} ms   per-instruction {pin*1000:5.2f} ms")
print(f"  CLIPSeg @352x352 (its native): per-instruction 54.3 ms")
print(f"\n  matched at 352:  {54.3/(res[352]['per_instruction']*1000):.1f}× faster per instruction")
json.dump(dict(ceiling_dev=ceil, latency_by_size={str(k):v for k,v in res.items()}),
          open(HERE/"fix_checks.json","w"), indent=2)
print(f"\nwrote {HERE/'fix_checks.json'}")
