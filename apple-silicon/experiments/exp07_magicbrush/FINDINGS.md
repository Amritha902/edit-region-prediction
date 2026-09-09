# Experiment 07 — how coarse is the supervision the base paper trains on?

AdaptEdit (arXiv:2604.23763) supervises its MaskPredictor on MagicBrush
human-annotated edit regions. We compared all 528 dev turns (72 images) against
the region that actually changed between source and target, recovered by CIE76
ΔE in L\*a\*b\*.

| edit type | n | human mask | actually changed | ratio | recall | **precision** | **IoU** |
|---|---:|---:|---:|---:|---:|---:|---:|
| insert | 103 | 60.0% | 7.0% | **11.9×** | 97.8% | 11.9% | **0.12** |
| modify | 390 | 62.9% | 11.2% | 8.5× | 96.5% | 17.1% | 0.17 |
| remove | 35 | 63.6% | 8.9% | 7.1× | 96.2% | 13.4% | 0.13 |
| **all** | **528** | **62.4%** | **10.3%** | **9.1×** | **96.7%** | **15.8%** | **0.16** |

## Reading

**Recall 96.7%** — the annotations are correct. The edit almost always falls
inside the annotated region.

**Precision 15.8%** — only about a sixth of the annotated region actually
changes. Median IoU against the true changed region is **0.16**.

These are region-of-interest scribbles — a human roughly brushing "work
somewhere in here" for an inpainting tool — not edit regions. A predictor
supervised on them learns to emit regions roughly an order of magnitude too
large. Insertion is the worst case at 11.9× and IoU 0.12, which is the most
likely reason the base paper's limitations section singles insertion out.

## Two corrections of record

1. An earlier claim that insertion regions "cannot be annotated" was **wrong**.
   MagicBrush annotates them. The defensible claim is the measured one: they are
   annotated ~9× too coarsely.
2. The first run of this measurement had the **mask polarity inverted**.
   MagicBrush uses bright = preserve, so the edit region is the complement.
   Uninverted, recall came out at 0.3% and IoU at 0.001 — obviously wrong, which
   is what prompted the check. Every number above uses the corrected polarity.

## Why this matters for the project

It converts the gap from an argument into a measurement, and it points directly
at the fix. Our ΔE-derived masks are exactly the tight region these annotations
lack — the same measurement used here to expose the problem, turned into the
supervision signal.
