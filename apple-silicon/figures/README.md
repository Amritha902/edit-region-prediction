# Figures

Regenerate with `python train/figures.py`. The script reads only the committed
result JSONs, so the figures cannot drift from the numbers.

| file | panels |
|---|---|
| `fig_frontier.png` | accuracy against per-instruction latency; IoU by edit type |
| `fig_analysis.png` | cost amortisation; ablation with controls and p-values; headroom against the basis ceiling |

## Two corrections applied after self-inspection

**The headroom panel compared incompatible numbers.** The basis ceiling came
from exp08, computed on the old merged cache (2,781 = train + dev), and was
plotted against a *dev* IoU. Recomputed on dev: insert 0.2119, modify 0.3867,
remove 0.4325 — the last differs materially from the 0.4575 previously shown.

**The latency comparison was not resolution-matched.** CLIPSeg runs at its
native 352×352; ours was measured at 640×640, i.e. 3.3× more pixels. Matched at
352 the figure is **9.7×** faster per instruction (5.58 ms vs 54.3 ms), and that
is the number on the plot. At 640 ours is 5.20 ms, so the advantage holds either
way, but only the matched comparison is reported as the headline.

## Two disclosures on the figures themselves

- The per-edit-type panel no longer claims "insertion is where we win" without
  qualification: we beat CLIPSeg there (0.092 vs 0.054) but the human masks
  still lead (0.120).
- The ablation panel's y-axis is truncated at 0.14 to resolve the error bars,
  and says so.

## Known limitation not yet quantified

The `remove` ceiling on dev is computed from only 32 samples. That bar carries
wide uncertainty which is not shown. Every other figure uses ≥100 samples.
