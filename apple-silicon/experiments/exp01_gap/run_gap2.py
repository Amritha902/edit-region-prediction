"""Experiment 02 - the honest baseline, and where it breaks.

exp01 showed the inherited pipeline ran FastSAM at a confidence that left only
1-2 candidate masks, so CLIP text-selection was vacuous. Fixed here: conf=0.05
gives a real candidate pool (~25 masks). We then do the CLIP selection
ourselves so we can see the full ranking, not just the winner.

The question this answers: with the baseline working properly, can
select-an-existing-mask reach the edit region for each edit type?
"""
import sys, os, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)

import numpy as np, torch, clip
from PIL import Image
from ultralytics import FastSAM
from src.device import pick_device

CONF, IMGSZ = 0.05, 1024
CASES = [
    ("000000000785.jpg", "the skier's red jacket",   "modify", "jacket region exists as an object"),
    ("000000000785.jpg", "the ski poles",            "remove", "poles exist as an object"),
    ("000000000785.jpg", "a backpack on the skier",  "insert", "EMPTY - nothing to select"),
    ("000000000785.jpg", "a pine tree on the left",  "insert", "EMPTY - nothing to select"),
    ("000000039769.jpg", "the cat on the left",      "remove", "instance disambiguation"),
    ("000000039769.jpg", "a collar on the right cat","insert", "EMPTY - nothing to select"),
    ("000000002149.jpg", "the front apple",          "modify", "instance disambiguation"),
]

def main():
    dev = pick_device("auto")
    sam = FastSAM("models/instruct-seg-edit/best.pt")
    cm, cp = clip.load("ViT-B/32", device=dev)
    out = HERE/"out2"; out.mkdir(exist_ok=True); rows=[]

    for i,(fn,text,kind,note) in enumerate(CASES,1):
        p = HERE/"images"/fn
        r = sam.predict(str(p), conf=CONF, iou=0.9, imgsz=IMGSZ, retina_masks=True,
                        max_det=300, verbose=False, device=dev)[0]
        if r.masks is None: print(f"[{i}] no masks"); continue
        masks = r.masks.data.cpu().numpy() > 0.5
        orig  = r.orig_img[:, :, ::-1]
        crops, keep = [], []
        for j,b in enumerate(r.boxes.xyxy.tolist()):
            if masks[j].sum() <= 100: continue
            x1,y1,x2,y2 = (int(v) for v in b)
            if x2-x1 < 4 or y2-y1 < 4: continue
            crops.append(cp(Image.fromarray(orig[y1:y2, x1:x2])).to(dev)); keep.append(j)
        if not crops: print(f"[{i}] no usable crops"); continue
        with torch.no_grad():
            imf = cm.encode_image(torch.stack(crops)); imf /= imf.norm(dim=-1,keepdim=True)
            txf = cm.encode_text(clip.tokenize([text]).to(dev)); txf /= txf.norm(dim=-1,keepdim=True)
            sim = (imf @ txf.T).squeeze(1).float().cpu().numpy()
        order = np.argsort(-sim)
        best  = keep[order[0]]
        m     = masks[best]
        # margin = how decisively did the winner beat runner-up?
        margin = float(sim[order[0]] - sim[order[1]]) if len(order)>1 else float("nan")
        rows.append(dict(case=i,image=fn,text=text,kind=kind,note=note,
                         n_candidates=len(keep),coverage=float(m.mean()),
                         top_sim=float(sim[order[0]]),margin=margin))
        vis = orig.copy(); vis[m] = (vis[m]*0.4 + np.array([255,105,180])*0.6).astype(np.uint8)
        Image.fromarray(vis).save(out/f"{i:02d}_{kind}_{text[:18].replace(' ','_')}.png")
        print(f"  [{i}] {kind:6s} | {text:26s} | {len(keep):3d} cands | "
              f"picked covers {m.mean():6.2%} | sim {sim[order[0]]:.3f} margin {margin:+.4f}")
    (HERE/"results2.json").write_text(json.dumps(rows,indent=2))
    print(f"\nwrote results2.json + {out}/")

if __name__ == "__main__":
    print("=== Experiment 02: honest baseline (conf=0.05, real candidate pool) ===\n")
    main()
