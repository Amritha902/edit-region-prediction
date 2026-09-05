"""Stages 2-3 - apply the edit, then recover the edit mask from the pixel diff.

InstructPix2Pix regenerates the whole frame, so every pixel differs a little and
the threshold's real job is rejecting that global drift, not detecting the edit.
DELTA_E_THRESHOLD was tuned on synthetic pairs (see mask_extraction docstring);
--pilot sweeps it against real ones before we commit hours of compute.

  python datagen/generate_pairs.py --pilot 24
  python datagen/generate_pairs.py --all
"""
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))

import numpy as np, torch, cv2
from PIL import Image
from src.device import pick_device
import mask_extraction as ME

MODEL   = ROOT / "models/instruct-pix2pix"
IMGDIR  = ROOT / "data/coco/images"
OUT     = HERE / "pairs"
SIZE    = 512
STEPS   = 20
GUID    = 7.5     # how hard to follow the instruction
IMG_GUID= 1.5     # how hard to stay faithful to the input image


def load_pipe(dev):
    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        str(MODEL), torch_dtype=torch.float16, variant="fp16",
        safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def is_degenerate(img):
    """fp16 VAE decode can overflow to NaN on MPS, which lands as a flat frame.

    Measured identical output between fp16 and fp32 on the samples tried, so we
    run fp16 (5.2 GB vs 8.3 GB, same speed) and catch the failure per sample
    rather than paying for fp32 everywhere. A flat frame is also what a totally
    failed edit looks like, so this filter earns its place either way.
    """
    a = np.asarray(img, dtype=np.float32)
    return bool(np.isnan(a).any() or a.std() < 3.0)


def fit(img, size=SIZE):
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(8, int(w*s)//8*8), max(8, int(h*s)//8*8)), Image.LANCZOS)
    return img


def box_stats(mask, box):
    """How much of the recovered mask lands inside the intended instance box."""
    H, W = mask.shape
    x1, y1, x2, y2 = [int(v) for v in (box[0]*W, box[1]*H, box[2]*W, box[3]*H)]
    x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W, x2), min(H, y2)
    tot = mask.sum()
    if tot == 0: return 0.0
    return float(mask[y1:y2, x1:x2].sum()) / float(tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in (HERE/"samples.jsonl").open()]
    if a.pilot:
        # even spread across the three kinds
        byk = {}
        for r in rows: byk.setdefault(r["kind"], []).append(r)
        per = a.pilot // len(byk)
        rows = [r for k in byk for r in byk[k][:per]]
    else:
        rows = rows[a.start:]

    dev = pick_device("auto")
    OUT.mkdir(exist_ok=True); (OUT/"vis").mkdir(exist_ok=True)

    # Resume: a full run is ~10 h, so it has to survive being interrupted.
    if not a.pilot:
        before = len(rows)
        rows = [r for r in rows if not (OUT/f"{r['id']}_dE.npy").exists()]
        if before != len(rows):
            print(f"resuming: {before-len(rows)} already done, {len(rows)} left")

    print(f"device {dev} | {len(rows)} samples | steps {STEPS}")
    if not rows: print("nothing to do"); return
    pipe = load_pipe(dev)

    SWEEP = [8, 12, 16, 20, 25, 30]
    recs, t0, n_degen = [], time.time(), 0
    for i, r in enumerate(rows):
        src = Image.open(IMGDIR/r["image"]).convert("RGB")
        src = fit(src)
        g = torch.Generator(device="cpu").manual_seed(1368 + i)
        edited = pipe(r["text"], image=src, num_inference_steps=STEPS,
                      guidance_scale=GUID, image_guidance_scale=IMG_GUID,
                      generator=g).images[0]
        if is_degenerate(edited):
            n_degen += 1
            print(f"    degenerate output on {r['id']}, skipping", flush=True)
            continue

        o = cv2.cvtColor(np.array(src),    cv2.COLOR_RGB2BGR)
        e = cv2.cvtColor(np.array(edited), cv2.COLOR_RGB2BGR)
        dE = ME.delta_e(o, e)

        rec = dict(id=r["id"], kind=r["kind"], sub=r["sub"], text=r["text"],
                   image=r["image"], target_box=r["target_box"])
        for t in SWEEP:
            m = ME.clean_mask(((dE > t) * 255).astype(np.uint8)) > 0
            n, _ = cv2.connectedComponents(m.astype(np.uint8))
            rec[f"cov@{t}"]  = float(m.mean())
            rec[f"inbox@{t}"] = box_stats(m, r["target_box"])
            rec[f"blobs@{t}"] = int(n - 1)
        recs.append(rec)

        if a.pilot:
            m = ME.clean_mask(((dE > ME.DELTA_E_THRESHOLD)*255).astype(np.uint8)) > 0
            vis = np.array(edited).copy(); vis[m] = (vis[m]*0.45 + np.array([255,0,180])*0.55).astype(np.uint8)
            Image.fromarray(np.hstack([np.array(src), np.array(edited), vis])).save(
                OUT/"vis"/f"{r['kind']}_{i:03d}.png")
        else:
            edited.save(OUT/f"{r['id']}_edit.jpg", quality=92)
            np.save(OUT/f"{r['id']}_dE.npy", dE.astype(np.float16))

        if (i+1) % 5 == 0 or i == len(rows)-1:
            el = time.time()-t0
            eta = (len(rows)-i-1) * el/(i+1) / 3600
            print(f"  {i+1}/{len(rows)}  {el:6.0f}s  {el/(i+1):.1f}s/edit  "
                  f"eta {eta:4.1f}h  degen {n_degen}", flush=True)
        if not a.pilot and (i+1) % 50 == 0:
            (HERE/f"pairs_{a.start}.json").write_text(json.dumps(recs, indent=2))

    name = "pilot.json" if a.pilot else f"pairs_{a.start}.json"
    (HERE/name).write_text(json.dumps(recs, indent=2))
    print(f"\nwrote {HERE/name}   ({(time.time()-t0)/max(1,len(rows)):.2f}s per edit, "
          f"{n_degen} degenerate)")

if __name__ == "__main__":
    main()
