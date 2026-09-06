"""Experiment 03 - does an edit-ready field amortise across instructions?

Claim under test: the instruction-independent part of scene analysis can be
computed once and reused, so N instructions cost 1 analysis + N cheap
resolutions rather than N analyses.

Also tests the thing selection cannot do: resolving an insertion to empty
space via affordance slots.
"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"datagen"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
from ultralytics import YOLO
from src.device import pick_device
import editfield as EF
import topology as TP

IMAGES = {
    "skier": "000000000785.jpg",
    "cats":  "000000039769.jpg",
}
INSTRUCTIONS = [
    "put a backpack on the person",   "put a hat on the person",
    "add a scarf around the person",  "remove the person",
    "make the person blue",           "put a bird above the person",
    "add a bench beside the person",  "remove the skis",
    "make the skis red",              "put a flag above the skis",
    "remove the left cat",            "remove the right cat",
    "put a collar on the left cat",   "put a bow tie on the right cat",
    "make the left cat orange",       "make the right cat grey",
    "add a toy beside the left cat",  "put a blanket on the right cat",
    "remove the remote",              "make the remote black",
]

def main():
    dev = pick_device("auto")
    det = YOLO("yolov8x-seg.pt"); names = det.names
    imgdir = ROOT/"experiments/exp01_gap/images"
    rows, fields = [], {}

    print("=== building fields (once per image) ===")
    for tag, fn in IMAGES.items():
        t0 = time.time()
        r = det.predict(str(imgdir/fn), conf=0.25, imgsz=1024, retina_masks=True,
                        verbose=False, device=dev)[0]
        det_sec = time.time() - t0
        f = EF.build(r, names)
        fields[tag] = f
        s = EF.summary(f)
        print(f"  {tag:6s} detect {det_sec:5.2f}s + field {s['build_sec']:5.2f}s "
              f"| {s['regions']} regions {s['classes']} | free {s['free_frac']:.0%} "
              f"| {s['slots']} slots | {s['supports']} supports")
        rows.append(dict(stage="build", image=tag, detect_sec=det_sec, **s))

    print("\n=== resolving 20 instructions against the cached fields ===")
    hit = 0; tot_resolve = 0.0
    for instr in INSTRUCTIONS:
        best = None
        for tag, f in fields.items():
            m, meta = EF.resolve(f, instr)
            tot_resolve += meta["sec"]
            if m is not None and (best is None or m.sum() > 0):
                best = (tag, m, meta); break
        if best is None:
            print(f"  MISS  {instr}")
            rows.append(dict(stage="resolve", instr=instr, ok=False)); continue
        tag, m, meta = best; hit += 1
        d = TP.describe(m)
        print(f"  {meta['kind']:6s} {instr:34s} -> {tag:5s} {meta['region']:8s} "
              f"slot={str(meta['slot']):6s} cov={m.mean():6.2%} shape={d['kind']:9s} "
              f"leak={meta['leak_risk']:.2f} ({meta['sec']*1000:.2f} ms)")
        rows.append(dict(stage="resolve", instr=instr, ok=True, image=tag,
                         kind=meta["kind"], region=meta["region"], slot=meta["slot"],
                         coverage=float(m.mean()), shape=d["kind"],
                         leak_risk=meta["leak_risk"], sec=meta["sec"]))

    build_total = sum(r["detect_sec"]+r["build_sec"] for r in rows if r["stage"]=="build")
    n = len(INSTRUCTIONS)
    print(f"\n  resolved {hit}/{n}")
    print(f"  field build (2 images, once)   {build_total:7.2f} s")
    print(f"  20 resolutions total           {tot_resolve*1000:7.2f} ms")
    print(f"  amortised per instruction      {(build_total+tot_resolve)/n*1000:7.1f} ms")
    print(f"  recompute-every-time would be  {(build_total/2*n):7.2f} s  "
          f"({(build_total/2*n)/((build_total+tot_resolve)):.0f}x more)")
    (HERE/"results.json").write_text(json.dumps(rows, indent=2, default=str))

if __name__ == "__main__":
    main()
