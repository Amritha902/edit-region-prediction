"""Per-instruction latency for BOTH heads, resolution-matched.

latency.py and fix_checks.py both timed TextCoeffHead (v1, 1.4M). The reported
model is v2 (4.7M), whose coefficient-field decoder is strictly more work, so
its latency cannot be assumed equal. Time both at CLIPSeg's native 352 and at
our 640, on the same images, same device.
"""
import json, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(ROOT))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from PIL import Image
from src.device import pick_device
from model import EditRegionModel, TextCoeffHead
from model_v2 import SpatialCoeffHead

dev=pick_device("auto")
m=EditRegionModel("best.pt", device=dev)
img=Image.open("test_input.png").convert("RGB")
TXT=["put a hat on the dog"]*8
REP, WARM = 40, 8

class V1(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.h=TextCoeffHead(hidden=512,dropout=0.0)
    def forward(s,t,c,p):
        co,b=s.h(t,c); return (torch.einsum("bc,bchw->bhw",co,p)+b[:,:,None]).unsqueeze(1)

def sync():
    if dev=="mps": torch.mps.synchronize()

res={}
for name,make,npar in (("v1",lambda:V1(),1_399_073),
                       ("v2",lambda:SpatialCoeffHead(basis="none"),4_693_633)):
    head=make().to(dev).eval()
    real=sum(p.numel() for p in head.parameters())
    res[name]={"params":real}
    for size in (352,640):
        x=torch.from_numpy(np.asarray(img.resize((size,size)))).permute(2,0,1)\
           .float().div(255)[None].to(dev)
        with torch.no_grad():
            for _ in range(WARM): proto,ctx=m.backbone(x)
            sync(); t0=time.perf_counter()
            for _ in range(REP): proto,ctx=m.backbone(x)
            sync(); per_img=(time.perf_counter()-t0)/REP
            temb=m.encode_text(TXT[:1])
            for _ in range(WARM): head(temb.float(),ctx.float(),proto.float())
            sync(); t0=time.perf_counter()
            for _ in range(REP):
                temb=m.encode_text(TXT[:1])
                head(temb.float(),ctx.float(),proto.float())
            sync(); per_ins=(time.perf_counter()-t0)/REP
        res[name][size]={"per_image_ms":per_img*1e3,"per_instruction_ms":per_ins*1e3}
        print(f"{name} @{size}: per-image {per_img*1e3:7.2f} ms | "
              f"per-instruction {per_ins*1e3:6.3f} ms | params {real:,}", flush=True)

CLIPSEG_MS, CLIPSEG_P = 54.3, 150_747_746
print(f"\nCLIPSeg @352 (native): {CLIPSEG_MS} ms, {CLIPSEG_P:,} trainable params")
for n in res:
    r=res[n][352]["per_instruction_ms"]
    print(f"  {n} matched @352: {CLIPSEG_MS/r:5.1f}x faster per instruction | "
          f"{CLIPSEG_P/res[n]['params']:5.1f}x fewer trainable params")
    br=res[n][352]["per_image_ms"]/(CLIPSEG_MS-r)
    print(f"       break-even at N = {br:.2f} instructions on one image")
res["clipseg"]={"per_instruction_ms":CLIPSEG_MS,"params":CLIPSEG_P,"size":352}
json.dump(res, open(HERE/"latency_v2.json","w"), indent=2)
print("\nwrote train/latency_v2.json")
