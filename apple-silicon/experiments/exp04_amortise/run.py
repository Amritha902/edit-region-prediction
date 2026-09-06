"""Experiment 04 - the amortisation curve, 1 to 500 instructions.

A single 10x number is weak evidence. The claim is that scene understanding is
instruction-independent and therefore amortises, which predicts a specific
shape: recompute cost grows linearly in N, field cost is constant plus a
negligible per-query term. This measures that over N = 1..500 on real images.

Instructions are generated from each field's own detected regions crossed with
templates, so every one refers to something actually present.
"""
import sys, json, time, random
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"datagen"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
from ultralytics import YOLO
from src.device import pick_device
import editfield as EF

IMAGES = ["000000000785.jpg", "000000039769.jpg", "000000002149.jpg"]
NS = [1, 5, 10, 20, 50, 100, 200, 500]
COLORS = ["red","blue","green","yellow","purple","orange","black","white"]
ITEMS  = ["hat","bird","scarf","backpack","collar","lamp","flag","cushion","toy","bag"]
PREPS  = ["on","above","beside","below"]

def make_instructions(fields, n, rng):
    out = []
    pool = [(tag, r["cls"]) for tag, f in fields.items() for r in f["regions"]]
    if not pool: return out
    while len(out) < n:
        tag, cls = pool[rng.randrange(len(pool))]
        k = rng.random()
        if k < 0.34:
            out.append((tag, f"put a {rng.choice(ITEMS)} {rng.choice(PREPS)} the {cls}"))
        elif k < 0.67:
            out.append((tag, f"remove the {cls}"))
        else:
            out.append((tag, f"make the {cls} {rng.choice(COLORS)}"))
    return out

def main():
    dev = pick_device("auto")
    det = YOLO("yolov8x-seg.pt"); names = det.names
    imgdir = ROOT/"experiments/exp01_gap/images"
    rng = random.Random(1368)

    # one analysis per image, timed
    fields, analyse_sec = {}, {}
    for fn in IMAGES:
        t0 = time.time()
        r = det.predict(str(imgdir/fn), conf=0.25, imgsz=1024, retina_masks=True,
                        verbose=False, device=dev)[0]
        f = EF.build(r, names)
        analyse_sec[fn] = time.time() - t0
        fields[fn] = f
        print(f"  {fn}  analyse {analyse_sec[fn]:.3f}s  "
              f"{len(f['regions'])} regions  {sum(len(x['slots']) for x in f['regions'])} slots")
    mean_analyse = float(np.mean(list(analyse_sec.values())))
    build_total  = float(sum(analyse_sec.values()))
    print(f"\n  mean analysis {mean_analyse*1000:.0f} ms/image, "
          f"field built once for {len(IMAGES)} images = {build_total:.2f}s\n")

    print(f"{'N':>5} {'field total':>12} {'recompute':>11} {'speedup':>8} "
          f"{'per-query':>11} {'resolved':>9}")
    rows = []
    for n in NS:
        instrs = make_instructions(fields, n, random.Random(1368 + n))
        t0 = time.time(); ok = 0
        for tag, txt in instrs:
            m, meta = EF.resolve(fields[tag], txt)
            if m is not None and m.sum() > 0: ok += 1
        q = time.time() - t0
        field_total = build_total + q
        recompute   = mean_analyse * n
        rows.append(dict(n=n, field_total=field_total, query_total=q,
                         recompute=recompute, speedup=recompute/field_total,
                         resolved=ok, rate=ok/n))
        print(f"{n:>5} {field_total:>11.3f}s {recompute:>10.2f}s "
              f"{recompute/field_total:>7.1f}x {q/n*1e6:>9.0f} us {ok}/{n:<5}")

    # break-even: N where recompute overtakes the one-off field cost
    be = build_total / max(mean_analyse, 1e-9)
    print(f"\n  break-even at N = {be:.1f} instructions "
          f"(= number of images, as expected: the field costs one analysis each)")
    print(f"  per-query cost is {rows[-1]['query_total']/rows[-1]['n']*1e6:.0f} us, "
          f"{mean_analyse/(rows[-1]['query_total']/rows[-1]['n']):.0f}x cheaper than an analysis")
    (HERE/"results.json").write_text(json.dumps(rows, indent=2))

if __name__ == "__main__":
    main()
