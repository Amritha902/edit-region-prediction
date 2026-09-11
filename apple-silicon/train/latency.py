"""Latency and cost, measured — not asserted.

Accuracy is contested; cost is not. This measures the real numbers on this
machine, for a single instruction and amortised across many instructions on one
image, because the second is the normal interaction pattern and the place the
architecture actually differs.

Ours splits into a per-IMAGE stage (frozen backbone, gives 32 prototypes) and a
per-INSTRUCTION stage (CLIP text + a 1.4M MLP + a weighted sum). CLIPSeg has no
such split: every instruction re-runs the full 150M encoder-decoder.
"""
import sys, json, time
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(ROOT))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from PIL import Image
from src.device import pick_device
from model import EditRegionModel, TextCoeffHead

dev=pick_device("auto")
img=Image.open(ROOT/"experiments/exp01_gap/images/000000000785.jpg").convert("RGB")
INSTR=["put a hat on the person","remove the skis","make the jacket blue",
       "add a bird above the person","remove the person"]*8

def sync():
    if dev=="mps": torch.mps.synchronize()
    elif dev.startswith("cuda"): torch.cuda.synchronize()

# ---------------- ours ----------------
ours=EditRegionModel("best.pt", device=dev)
ck=torch.load(HERE/"edit_region_head_clean.pt", map_location=dev)
ours.head=TextCoeffHead(hidden=512, dropout=0.0).to(dev)
ours.head.load_state_dict(ck["head"]); ours.head.eval()
x=torch.from_numpy(np.asarray(img.resize((640,640)))).permute(2,0,1).float().div(255)[None].to(dev)

for _ in range(3):
    with torch.no_grad(): ours.backbone(x)
sync()
t=time.perf_counter()
for _ in range(10):
    with torch.no_grad(): proto,ctx=ours.backbone(x)
sync(); per_image=(time.perf_counter()-t)/10

with torch.no_grad(): proto,ctx=ours.backbone(x)
for _ in range(3):
    with torch.no_grad():
        e=ours.encode_text([INSTR[0]]); c,b=ours.head(e,ctx)
sync()
t=time.perf_counter()
for s in INSTR:
    with torch.no_grad():
        e=ours.encode_text([s]); c,b=ours.head(e,ctx)
        _=(torch.einsum("bc,bchw->bhw",c,proto)+b[:,:,None])
sync(); per_instr=(time.perf_counter()-t)/len(INSTR)
print(f"OURS   per-image {per_image*1000:7.1f} ms   per-instruction {per_instr*1000:7.2f} ms")

# ---------------- CLIPSeg ----------------
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
proc=CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
cs=CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(dev).eval()
for _ in range(3):
    inp=proc(text=[INSTR[0]], images=[img], padding=True, return_tensors="pt").to(dev)
    with torch.no_grad(): cs(**inp)
sync()
t=time.perf_counter()
for s in INSTR:
    inp=proc(text=[s], images=[img], padding=True, return_tensors="pt").to(dev)
    with torch.no_grad(): cs(**inp)
sync(); cseg=(time.perf_counter()-t)/len(INSTR)
print(f"CLIPSEG  per-instruction {cseg*1000:7.1f} ms  (no per-image reuse possible)\n")

print(f"{'N instructions on one image':32s} {'ours':>10} {'CLIPSeg':>10} {'speedup':>9}")
rows=[]
for N in (1,2,5,10,20,50,100):
    o=per_image + N*per_instr; c=N*cseg
    rows.append(dict(n=N, ours=o, clipseg=c, speedup=c/o))
    print(f"{N:>32} {o*1000:>9.1f}ms {c*1000:>9.1f}ms {c/o:>8.1f}×")

par_ours=sum(p.numel() for p in ours.head.parameters())
par_cs=sum(p.numel() for p in cs.parameters())
print(f"\ntrainable  ours {par_ours:,}   CLIPSeg {par_cs:,}   ratio {par_cs/par_ours:.0f}×")
print(f"break-even at N = {per_image/max(cseg-per_instr,1e-9):.2f} instructions")
json.dump(dict(per_image=per_image, per_instruction=per_instr, clipseg=cseg,
               params_ours=int(par_ours), params_clipseg=int(par_cs), rows=rows),
          open(HERE/"latency.json","w"), indent=2)
print(f"wrote {HERE/'latency.json'}")
