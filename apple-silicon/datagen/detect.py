"""Stage 0 - ground each image in real objects, then emit instructions.

Uses a COCO-pretrained YOLOv8x-seg (80 classes) rather than the project's
best.pt, which was fine-tuned to a single class 'object' and so cannot name
what it finds. We need class names and instance boxes to build spatially
disambiguated and affordance-grounded instructions.

Output: datagen/samples.jsonl, one line per (image, instruction) pair.
"""
import json, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import instructions as I
from src.device import pick_device

TARGET_PAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
CONF = 0.35          # detection must be trustworthy - this is our grounding
IMGDIR = ROOT / "data/coco/images"

def main():
    from ultralytics import YOLO
    dev = pick_device("auto")
    det = YOLO("yolov8x-seg.pt")
    names = det.names
    rng = random.Random(1368)

    imgs = sorted(IMGDIR.glob("*.jpg"))
    print(f"{len(imgs)} images on disk, target {TARGET_PAIRS} pairs")

    out = HERE / "samples.jsonl"
    n_pairs = 0; n_imgs = 0; kinds = {}
    t0 = time.time()
    with out.open("w") as fh:
        for p in imgs:
            if n_pairs >= TARGET_PAIRS: break
            r = det.predict(str(p), conf=CONF, imgsz=640, verbose=False, device=dev)[0]
            if r.boxes is None or len(r.boxes) == 0: continue
            H, W = r.orig_shape
            dets = []
            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                x1, y1, x2, y2 = b / [W, H, W, H]
                dets.append(dict(cls=names[int(c)], box=[float(x1), float(y1), float(x2), float(y2)],
                                 area_frac=float((x2-x1)*(y2-y1))))
            cands = I.build(dets, rng)
            if not cands: continue
            # at most 2 per image, so 2000 pairs span ~1000 images not 250
            for s in I.balanced_pick(cands, 2, rng):
                if n_pairs >= TARGET_PAIRS: break
                s["image"] = p.name; s["id"] = f"{p.stem}_{n_pairs:05d}"
                fh.write(json.dumps(s) + "\n")
                kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
                n_pairs += 1
            n_imgs += 1
            if n_imgs % 100 == 0:
                print(f"  {n_imgs} imgs -> {n_pairs} pairs  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nDONE {n_pairs} pairs from {n_imgs} images in {time.time()-t0:.0f}s")
    print("  by kind:", kinds)
    print("  ->", out)

if __name__ == "__main__":
    main()
