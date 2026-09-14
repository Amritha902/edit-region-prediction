"""Live demo: image + instruction -> predicted edit region -> edited image.

Self-contained. No server, no notebook. Everything runs on this machine.

    python demo.py --image test_input.png --instruction "put a hat on the dog"
    python demo.py --image photo.jpg --instruction "remove the car" --edit

Prints the timing split the architecture is built around: the backbone runs
once per image, the head runs once per instruction.
"""
import argparse, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/"train")); sys.path.insert(0,str(HERE)); sys.path.insert(0,str(HERE/"datagen"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, cv2
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.device import pick_device
from model import EditRegionModel
from model_v2 import SpatialCoeffHead

ap=argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--instruction", required=True)
ap.add_argument("--seed", type=int, default=1368)
ap.add_argument("--thr", type=float, default=0.3)
ap.add_argument("--dilate", type=int, default=0,
                help="0 shows the raw prediction; 32 is the editing operating point")
ap.add_argument("--edit", action="store_true", help="also run SD inpainting inside the region")
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--out", default="demo_out.png")
a=ap.parse_args()

dev=pick_device("auto")
print(f"\n  device            {dev}")

# Timings are meaningless if something else is on the GPU. The benchmark figures
# (33.6 ms per image, 5.30 ms per instruction) were measured on an idle machine;
# under contention this demo has shown 10x those, which would misrepresent the
# model in the wrong direction during a live demonstration.
import subprocess
_busy = subprocess.run(["pgrep","-f","stage2_sd_sweep|clean_eval_full|precompute_full"],
                       capture_output=True, text=True).stdout.split()
if _busy:
    print(f"  !! WARNING        {len(_busy)} training/eval process(es) are using the GPU.")
    print(f"  !!                Timings below are contention-inflated, NOT the benchmark.")
    print(f"  !!                Stop them before demonstrating, or quote the report's numbers.")
src=Image.open(a.image).convert("RGB")
w,h=src.size; k=512/min(w,h)
src=src.resize((max(8,int(w*k)//8*8), max(8,int(h*k)//8*8)), Image.LANCZOS)
print(f"  image             {Path(a.image).name}  {src.size[0]}x{src.size[1]}")
print(f"  instruction       \"{a.instruction}\"")

t0=time.perf_counter()
m=EditRegionModel("best.pt", device=dev)
ck=torch.load(HERE/f"train/head_full_v2_s{a.seed}.pt", map_location=dev)
head=SpatialCoeffHead(basis="none").to(dev); head.load_state_dict(ck["head"]); head.eval()
print(f"  model loaded      {time.perf_counter()-t0:.1f}s   "
      f"(head {sum(p.numel() for p in head.parameters()):,} trainable params, "
      f"dev IoU {ck['dev_iou']:.4f} on {ck['n_train']:,} training samples)")

im=src.resize((640,640),Image.BILINEAR)
x=torch.from_numpy(np.asarray(im)).permute(2,0,1).float().div(255)[None].to(dev)

def sync():
    if dev=="mps": torch.mps.synchronize()

# Warm up first. The very first MPS call compiles Metal graphs and costs about
# a second; timing that instead of steady state would understate the model by
# roughly 20x and make it look slower than the baseline it beats.
with torch.no_grad():
    for _ in range(3):
        proto,ctx=m.backbone(x)
        head(m.encode_text([a.instruction]).float(), ctx.float(), proto.float())
sync()

REP=5
t=time.perf_counter()
with torch.no_grad():
    for _ in range(REP): proto,ctx=m.backbone(x)
sync(); t_img=(time.perf_counter()-t)*1000/REP

t=time.perf_counter()
with torch.no_grad():
    for _ in range(REP):
        temb=m.encode_text([a.instruction])
        lg=head(temb.float(), ctx.float(), proto.float())
sync(); t_ins=(time.perf_counter()-t)*1000/REP

prob=cv2.resize(torch.sigmoid(lg)[0,0].cpu().numpy(), src.size)
mask=prob>a.thr
if a.dilate>0:
    kk=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*a.dilate+1,2*a.dilate+1))
    mask=cv2.dilate(mask.astype(np.uint8),kk)>0

print(f"\n  timings, steady state after warm-up, mean of {REP}:")
print(f"    per-image   (frozen backbone)      {t_img:7.1f} ms   paid ONCE per photograph")
print(f"    per-instruction (CLIP + 4.7M head) {t_ins:7.1f} ms   paid per instruction")
print(f"    CLIPSeg                               54.3 ms   per instruction, every time")
print(f"    -> {54.3/max(t_ins,1e-9):.1f}x cheaper per instruction, 32x fewer trainable params")
print(f"\n  cost of N instructions on this one image:")
for N in (1,5,20,100):
    ours=(t_img+N*t_ins)/1000; theirs=N*54.3/1000
    print(f"    N={N:>3}   ours {ours:6.2f}s   CLIPSeg {theirs:6.2f}s   "
          f"{'%.1fx faster'%(theirs/ours) if theirs>ours else '%.1fx slower'%(ours/theirs)}")
print(f"\n  predicted region covers {mask.mean():.1%} of the frame "
      f"(threshold {a.thr}, dilation {a.dilate} px)")

panels=[(np.asarray(src),"source"),(mask,"predicted edit region")]
ov=np.asarray(src).copy().astype(float)
ov[mask]=0.55*ov[mask]+0.45*np.array([80,200,120])
panels.append((ov.astype(np.uint8),"region on the image"))

if a.edit:
    from diffusers import StableDiffusionInpaintPipeline
    t=time.perf_counter()
    inp=StableDiffusionInpaintPipeline.from_pretrained(str(HERE/"models/sd-inpaint"),
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False).to(dev)
    inp.set_progress_bar_config(disable=True)
    print(f"  inpainting model loaded {time.perf_counter()-t:.1f}s")
    t=time.perf_counter()
    g=torch.Generator("cpu").manual_seed(a.seed)
    out=inp(prompt=a.instruction, image=src,
            mask_image=Image.fromarray((mask*255).astype(np.uint8)),
            num_inference_steps=a.steps, generator=g).images[0].resize(src.size)
    print(f"  edit generated          {time.perf_counter()-t:.1f}s  ({a.steps} steps)")
    panels.append((np.asarray(out),"edited inside the region"))

fig,ax=plt.subplots(1,len(panels),figsize=(4.2*len(panels),4.6))
for i,(im_,ttl) in enumerate(panels):
    ax[i].imshow(im_, cmap=None if im_.ndim==3 else "gray")
    ax[i].set_title(ttl, fontsize=12); ax[i].set_xticks([]); ax[i].set_yticks([])
fig.suptitle(f'"{a.instruction}"', fontsize=13, style="italic")
fig.tight_layout()
fig.savefig(a.out, dpi=140, bbox_inches="tight")
print(f"\n  wrote {a.out}\n")
