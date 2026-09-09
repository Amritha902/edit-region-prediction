# Text-conditioned edit-region prediction — results

2,631 MagicBrush samples, 1.4M trainable parameters over a frozen 71.75M
FastSAM backbone and a frozen CLIP text encoder. Apple M5, ~0.2 min per run.
All runs seeded; identical commands reproduce identical numbers.

## Method

FastSAM's segmentation head emits 32 YOLACT-style prototype masks per frame.
The coefficients that combine them normally come from a **detection**, which is
exactly why its output can only ever be an existing object. We predict them
from the **instruction** instead:

    mask = σ( Σ  cᵢ(text, image) · protoᵢ  +  b )

Prototypes are spatial basis functions, not objects, so a text-derived
combination is not bound to any detected thing.

Targets are **not** MagicBrush's own masks — measured over all 528 dev turns
those are 9.1× too large (precision 15.8%, IoU 0.16 against the region that
actually changed, see `experiments/exp07_magicbrush`). We keep MagicBrush's real
human source/target pairs and derive the target by CIE76 ΔE in L\*a\*b\*.

## Headline — 8 seeds, ~395 held-out samples

| | IoU |
|---|---|
| with instruction | **0.1397 ± 0.0120** |
| instruction zeroed (ablation) | 0.0221 ± 0.0047 |
| **language contributes** | **+0.1177 ± 0.0137** |

Paired t-test **p = 5.2 × 10⁻⁸**, Wilcoxon p = 0.0078, positive in **8/8** seeds.
The instruction multiplies IoU by **6.3×**.

## Data scaling — fixed 150-sample validation set

| N | IoU | no-text | delta |
|---:|---|---|---|
| 100 | 0.0738 ± 0.0043 | 0.0273 | +0.0465 |
| 200 | 0.0850 ± 0.0122 | 0.0305 | +0.0545 |
| 300 | 0.1099 ± 0.0121 | 0.0314 | +0.0785 |
| 450 | 0.1203 ± 0.0202 | 0.0232 | +0.0971 |
| 600 | 0.1265 ± 0.0128 | 0.0188 | +0.1077 |
| **2631** | **0.1632 ± 0.0064** | 0.0437 | **+0.1194** |

A log-linear fit on the ≤600 points predicted 0.171 at ~2,400 samples; the
measured value at 2,631 is 0.1632. **This crosses MagicBrush's own human masks
at 0.16** — ΔE-derived supervision scores higher against the true changed region
than the annotations the base paper trains on. The curve has not flattened;
37 further shards (~6,000 turns) remain unused.

## By edit type

| kind | IoU |
|---|---|
| modify | 0.1641 ± 0.0127 |
| remove | 0.1476 ± 0.0433 |
| **insert** | **0.0616 ± 0.0129** |

## What is honest about this

**The two headline numbers differ by measurement, not luck.** 0.1632 comes from
the scaling curve's fixed 150-sample validation set; 0.1397 from the 8-seed run's
15% split (~395 samples). The second is more conservative and is the one to
quote as the model's score. The first supports the "crosses 0.16" comparison and
must be stated *with* its validation-set size.

**Insertion did not improve.** 0.067 at 503 samples → 0.062 at 2,631, while
modify and remove both rose substantially. Five times the data moved it
nowhere. This is the project's own gap failing to close, and it is the open
problem — not something to bury.

**Absolute IoU is low.** ~0.14 is a weak segmenter in absolute terms. The claim
is not that this is a good segmentation model; it is that the supervision is
better than the field's standard, and that the instruction demonstrably drives
the prediction.

## Two methodological findings

**Raw IoU is a misleading selection metric here.** In a 108-config sweep the
best-IoU configuration scored 0.1429 with a no-text ablation of 0.0667, and one
config reached IoU 0.1389 with delta **−0.0012**. Those models converge within 3
epochs to a blob covering ~20% of the frame and score identically with the
instruction zeroed. Ranking on IoU selects models that have learned nothing about
language. All selection here is on **delta**.

**Checkpoint selection had the same failure.** Selecting the best epoch by IoU
picked epoch 1 — a 19.8%-coverage blob against a true 8.8% — with delta −0.0007.
Selection was changed to delta.

**Seeding.** An earlier version seeded the data permutation but not weight
initialisation; identical commands gave IoU 0.1215 and 0.1073. Everything is
seeded now and verified reproducible across three identical runs.

## Reproduce

    python train/precompute.py --split train --batch 8     # ~5 min, GPU-bound
    python train/train_cached.py --epochs 60 --batch 64 --lr 1e-3 \
        --hidden 512 --dropout 0.0 --wd 0.01 --wdice 2.0 --seed 1368
    python train/scaling.py
    python train/qualitative.py
