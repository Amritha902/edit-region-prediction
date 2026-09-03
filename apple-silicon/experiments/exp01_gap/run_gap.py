"""Experiment 01 - does the inherited model predict the EDIT REGION?

The inherited Stage-1 model is FastSAM + post-hoc CLIP selection: it segments
every object class-agnostically, then CLIP picks whichever EXISTING mask best
matches the instruction text. Structurally it can only ever return a region
that is already an object.

This probes that with three edit types. MODIFY and REMOVE have a referent
object, so selection can work. INSERT does not -- the edit region is empty
space -- so there is no mask for CLIP to pick, and the model must fail.

Run:  python experiments/exp01_gap/run_gap.py
"""
import sys, os, json, warnings, time
from pathlib import Path

warnings.filterwarnings("ok" and "ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
from PIL import Image
from src.model import SegmentationModel

# (image, instruction, edit type, where the edit region actually is)
CASES = [
    ("000000000785.jpg", "make the skier's jacket blue",      "modify", "the red jacket"),
    ("000000000785.jpg", "remove the ski poles",              "remove", "the two poles"),
    ("000000000785.jpg", "put a backpack on the skier",       "insert", "EMPTY SPACE on her back"),
    ("000000000785.jpg", "add a pine tree on the left",       "insert", "EMPTY SNOW, left third"),
    ("000000039769.jpg", "remove the cat on the left",        "remove", "left cat only, not both"),
    ("000000039769.jpg", "put a collar on the right cat",     "insert", "EMPTY SPACE at its neck"),
    ("000000002149.jpg", "make the front apple red",          "modify", "one apple, not all"),
]

def main():
    seg = SegmentationModel()
    out = HERE / "out"; out.mkdir(exist_ok=True)
    rows = []
    for i, (fn, instr, kind, truth) in enumerate(CASES, 1):
        img = Image.open(HERE / "images" / fn).convert("RGB")
        t0 = time.time()
        try:
            (res, mask) = seg.segment_image(img, [instr], return_result=True)
            m = np.array(mask) > 127
            cov = float(m.mean())
            # how many disjoint blobs did it return?
            import cv2
            n, _ = cv2.connectedComponents(m.astype(np.uint8))
            blobs = n - 1
            res.save(out / f"{i:02d}_{kind}.png")
            err = None
        except Exception as e:
            cov, blobs, err = float("nan"), -1, f"{type(e).__name__}: {e}"
        dt = time.time() - t0
        rows.append(dict(case=i, image=fn, instruction=instr, kind=kind,
                         truth=truth, coverage=cov, blobs=blobs, sec=dt, error=err))
        status = err if err else f"covers {cov:6.2%} of frame, {blobs} blob(s)"
        print(f"  [{i}] {kind:6s} | {instr:34s} -> {status}  ({dt:.2f}s)")

    (HERE / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {HERE/'results.json'} and {out}/")

if __name__ == "__main__":
    print("=== Experiment 01: can the inherited model find the edit region? ===\n")
    main()
