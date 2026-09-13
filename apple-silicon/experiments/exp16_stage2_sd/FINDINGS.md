# Experiment 16 — Stage 2 with a real inpainting model

exp13 gated one editor's whole-frame output through a mask, so the mask could
only subtract and an over-large mask cost nothing. This repeats the test with
Stable Diffusion inpainting, where the mask defines what the model regenerates.
Same 60 held-out samples, same predicted masks, 20 steps.

| gating mask | recall ↑ | collateral ↓ | net ↑ | PSNR out ↑ | CLIP in ↑ |
|---|---:|---:|---:|---:|---:|
| none — whole frame | 33.3% | 24.2% | +9.1% | 22.56 | 0.2445 |
| **ours, tuned (thr 0.2, dilate 64 px)** | **71.1%** | 39.1% | **+32.0%** | 14.11 | **0.2623** |
| MagicBrush human mask | 78.4% | 51.7% | +26.8% | 13.30 | 0.2550 |
| **ground-truth region** | 59.4% | **6.4%** | **+53.0%** | **27.36** | 0.2464 |

## What changed against exp13

**Localization is worth more here: 5.8× against 3.3×.** An inpainting model given
a correct region can synthesise content; a compositing pipeline can only pass an
edit through a stencil. +53.0% for the oracle against +9.1% whole-frame.

**Our mask now beats the human annotation, +32.0% against +26.8%.** exp13 had us
losing that comparison (14.2 against 18.5). The ordering reverses once the mask
is used to generate rather than to gate.

## The qualification, which matters more

We win on net by raising recall to 71.1%, **not** by reducing collateral.
Collateral goes the wrong way: 39.1% against 24.2% for whole-frame editing, a
factor of **1.62 MORE**, and PSNR outside the region drops 8.45 dB. The oracle
shows what precision buys instead — 6.4% collateral at +53.0% net.

The cause is the operating point. We carried 64 px of dilation over from exp13,
where enlarging the mask can only recover more of the edit. Under inpainting an
over-large mask regenerates content that should have been preserved, so the
dilation that helped there hurts here. Threshold and dilation need re-tuning for
this setting; the figure above is the untuned one and is reported as such.

## Note on a reporting bug caught here

The summary line originally printed "collateral 24.2% -> 39.1% (0.6x less)".
It is 1.62x MORE. The ratio was computed the wrong way round and then labelled
"less" unconditionally. Fixed in stage2.py; no published number depended on it,
because it was caught before the figure entered any document.

## Throttling observed during this run

Seconds of compute per sample, at a fixed workload, drifted 76 → 93 → 111 → 114
→ 121 over the first 25 samples: the machine was thermally throttling, roughly
1.6x slower. After a pause and active cooling it returned to 74 s/sample. Die
temperature is not readable without sudo; throughput drift at a fixed workload
is a usable proxy and needs no privileges. The battery sensor read ~31 °C
throughout and was not informative about the SoC.
