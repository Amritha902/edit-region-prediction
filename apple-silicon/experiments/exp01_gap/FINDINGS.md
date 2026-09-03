# Experiment 01-02 — Does the inherited model predict the *edit region*?

Run on Apple M5 (10-core GPU, 24 GB unified, Metal 4), torch 2.13 / MPS,
ultralytics 8.3.209, checkpoint `best.pt` (71.8 M params, YOLOv8x-seg fine-tune).
Images: COCO val2017 `000000000785` (skier), `000000039769` (two cats),
`000000002149` (apples). Reproduce: `python experiments/exp01_gap/run_gap2.py`.

## Finding 1 — the inherited text conditioning was vacuous

`SegmentationModel.segment_image` calls `FastSAM.predict(texts=[...])` at
ultralytics' default confidence. Measured candidate-mask count on the skier
image with this checkpoint:

| conf  | 0.25 | 0.10 | 0.05 | 0.01 | 0.001 |
|-------|-----:|-----:|-----:|-----:|------:|
| masks |    2 |   11 |   25 |   77 |   300 |

At the default the pool is 1–2 masks, so CLIP's `argmax` over candidates has
nothing to choose between. Six semantically unrelated prompts — "the red
jacket", "the ski poles", "a backpack", "the snow", "the sky", and the nonsense
string "xyzzy nonsense qwerty" — produced only **two distinct masks**, with
IoU exactly 1.0000 between the first four.

The demo appeared to do text-conditioned segmentation. It never did. This is a
configuration defect in the inherited code, not a property of the method, and it
had to be fixed before any claim about the method could be made.

*(Note: `_clip_inference`'s docstring says it returns shape `(M, N)`; the
broadcast actually yields `(N_texts, M_crops)`. The docstring is wrong, the
code is correct — worth knowing when reading that file.)*

## Finding 2 — fixed, the baseline still fails every edit type: 0 / 7

With `conf=0.05, iou=0.9, imgsz=1024, retina_masks=True` the pool is 8–18 real
candidates and CLIP selection is meaningful. Result:

| # | type   | instruction               | cands | selected | margin  | outcome |
|---|--------|---------------------------|------:|---------:|--------:|---------|
| 1 | modify | the skier's red jacket    |    18 |   11.03% | +0.0579 | whole person, not the jacket |
| 2 | remove | the ski poles             |    18 |   11.03% | +0.0012 | whole person |
| 3 | insert | a backpack on the skier   |    18 |   11.03% | +0.0330 | whole person |
| 4 | insert | a pine tree on the left   |    18 |    1.24% | +0.0035 | the skis |
| 5 | remove | the cat on the left       |    10 |   19.99% | +0.0000 | the blanket |
| 6 | insert | a collar on the right cat |    10 |   19.99% | +0.0000 | the blanket |
| 7 | modify | the front apple           |     8 |   11.37% | +0.0073 | wrong apple |

Nothing is correct. Three distinct failure modes:

**Granularity.** Cases 1–3 return the identical whole-person mask for "jacket",
"poles" and "backpack". The candidate set contains no part-level region, so the
selected region is always the object, never the part being edited.

**Spatial language.** Cases 5 and 6 have margin *exactly* +0.0000 — the top two
candidates tie. Cropped-region CLIP sees each crop in isolation, so "left" and
"right" carry no information. The model cannot disambiguate instances by
position, which is precisely the failure the 2026 SOTA admits to when it keeps
only the largest connected component.

**Structure.** Cases 3, 4 and 6 are insertions. The edit region is empty space —
her back, the snow, the cat's neck. No such region exists in the candidate set,
so selection cannot reach it *even in principle*. Case 4 is the clearest: asked
for empty snow on the left, it returned the skis.

## Why this is the gap

Select-an-existing-mask is the wrong formulation. The edit region is not
generally an object: it is a part, an instance picked out by spatial language,
or a region containing nothing at all. It has to be **predicted**, conditioned
jointly on the image and the instruction — and to be predicted, it has to be
supervised, which is the thing no current dataset provides.

## What this licenses us to claim

Measured, on our hardware, with the inherited checkpoint. Not asserted from the
literature. That makes it defensible under questioning.

Limits, stated plainly: n = 7 probes on 3 images, chosen to span the edit types
rather than sampled. It establishes that the failure exists and is structural,
not its rate. A quantified baseline over a real eval split is Month 1 work.
