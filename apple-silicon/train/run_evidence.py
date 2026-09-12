"""Render the real run logs as an implementation-evidence sheet.

Every line is copied verbatim from a log this machine produced. Layout is done
in absolute inches rather than axes fractions -- the fractional version silently
clipped the last panel and overlapped the title.
"""
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def tail(p, pat, n):
    try: L=[l.rstrip() for l in Path(p).read_text().splitlines() if re.search(pat,l)]
    except FileNotFoundError: return ["(missing)"]
    return L[-n:] if L else ["(not started)"]

def short(s):
    return s.replace(str(ROOT)+"/", "").replace(str(ROOT), ".")

panels = [
 ("1 — feature cache, all 51 MagicBrush shards",
  "$ python train/precompute_full.py --split train --batch 8",
  tail(ROOT/"precompute_full.log", r"device mps|8807/8807|kept \d+ of|kinds|wrote ", 6)),
 ("2 — train/dev leakage audit (55 shards hashed)",
  "$ python train/leakage_check.py",
  tail(ROOT/"leakage.log", r"dev distinct|train distinct|overlapping|CLEAN|LEAKAGE", 5)),
 ("3 — full-scale training, 12 seeds × 60 epochs",
  "$ python train/clean_eval_full.py --seeds 12 --configs v2,v1",
  tail(ROOT/"full_train.log", r"^train \d|spatial|global|seed \d+:|params", 10)),
]

LH, PAD, GAP, TOP = 0.215, 0.11, 0.34, 0.62      # inches
W = 12.0
blocks=[]; y=TOP
for title,cmd,lines in panels:
    y += 0.26                                     # title line
    h = PAD*2 + LH*(len(lines)+1)
    blocks.append((title,cmd,lines,y,h)); y += h + GAP
H = y + 0.12

fig = plt.figure(figsize=(W,H))
fig.text(0.5, 1-0.30/H, "Implementation — actual run output on Apple M5 (MPS)",
         ha="center", va="center", fontsize=14, fontweight="bold")

for title,cmd,lines,ytop,h in blocks:
    fig.text(0.026, 1-(ytop-0.09)/H, title, fontsize=11.5,
             fontweight="bold", va="center", color="#111111")
    fig.patches.append(Rectangle((0.024, 1-(ytop+h)/H), 0.952, h/H,
        transform=fig.transFigure, fc="#12161c", ec="#39414d", lw=1.1, zorder=0))
    yy = ytop + PAD + LH*0.72
    fig.text(0.038, 1-yy/H, cmd, family="monospace", fontsize=9.7,
             color="#7fd1a6", va="center", zorder=2)
    for ln in lines:
        yy += LH
        col = "#e6e9ef"
        if re.search(r"CLEAN|overlapping images\s+0\b", ln): col="#7fd1a6"
        if re.search(r"LEAKAGE|Error|Traceback|Killed", ln):  col="#ff7b72"
        if re.search(r"IoU 0\.\d+", ln):                      col="#ffd479"
        fig.text(0.038, 1-yy/H, short(ln)[:112], family="monospace",
                 fontsize=9.1, color=col, va="center", zorder=2)

out = ROOT/"implementation/run_evidence.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=160, facecolor="white", bbox_inches="tight", pad_inches=0.12)
print("wrote", out)
