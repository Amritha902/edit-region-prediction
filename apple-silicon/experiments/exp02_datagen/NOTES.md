# Experiment 02 — dataset generation pilot

Stages 2–3 of Figure 2 on real data: apply the instruction with
InstructPix2Pix, then recover the edit mask from the pixel difference in
CIE L\*a\*b\*. Run on Apple M5, fp16 on MPS, 512 px short side, 20 steps,
`guidance_scale=7.5`, `image_guidance_scale=1.5`.

`samples/` holds triplets: **original | edited | recovered mask overlay**.

## What the pilot shows

**The method works when the edit is local.** `remove_001` ("remove the skis")
is the clean case: the editor removed the skis, poles, hat and gloves, and the
ΔE mask covers exactly those regions and nothing else. That mask is a usable
training target.

**The failure mode is global photometric drift, not the threshold.**
`insert_017` (a backlit river scene under a bridge) came back with the whole
frame brightened. InstructPix2Pix regenerates every pixel, so when it shifts
global exposure the ΔE signal is dominated by that shift: the recovered mask
scatters across the bridge girders, the swan and a bag strap, none of which
were the edit. No choice of threshold fixes this, because the drift and the
edit occupy the same ΔE range.

This is the risk the `mask_extraction` docstring flags — the threshold was
tuned on synthetic pairs where only the edit region changed. Real generated
pairs violate that assumption.

## Consequences for the full run

Two mitigations, to be validated before committing the ~10 h run:

1. **Photometric alignment before differencing.** Fit a per-channel linear
   correction (or histogram match) from edited to original over the whole
   frame, apply it, then difference. Global drift is removed by construction;
   a local edit survives because it is a small fraction of the pixels.
2. **Raise `image_guidance_scale`** from 1.5, which weights fidelity to the
   input image more heavily and reduces drift at the source.

A drift statistic should also be recorded per sample and used as a rejection
filter, so a pair whose mask is drift-dominated never reaches training.

## Timing

19 s per edit at 512 px / 20 steps — InstructPix2Pix runs three forward passes
per step for classifier-free guidance, so 20 steps is 60 UNet evaluations.
2000 pairs is therefore ~10.5 h. The generator is resumable: it skips any
sample whose output already exists and checkpoints every 50.
