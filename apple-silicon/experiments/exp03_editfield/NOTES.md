# Experiment 03 — does an edit-ready field amortise, and can it reach empty space?

Run on Apple M5. Two COCO images, 20 instructions. `python run.py`.

## Amortisation

The instruction-independent analysis is computed once per image and reused.

| | |
|---|---|
| field build, 2 images, once | 0.98 s |
| 20 resolutions, total | **0.55 ms** |
| amortised per instruction | 49 ms |
| recomputing per instruction | 9.79 s — **10× more** |

Resolution touches no network: it indexes a cached structure. The 10× is the
ratio between re-deriving the scene 20 times and deriving it twice. It is a
real saving on repeated editing of one image, which is the normal interaction
pattern, and it is *not* a claim about beating RegionE or SpotEdit — those
accelerate the diffusion pass, which this does not touch.

## Reaching empty space

20/20 instructions resolved. `slots.png`, left to right:

- **"put a hat on the person"** → `on`, 2.77% — head and shoulders. Right area,
  too coarse for a hat.
- **"put a bird above the person"** → `above`, 5.53% — **empty sky above her
  head.** No object occupies it, so no mask exists to select. This is the case
  the 0/7 baseline in exp01 cannot reach in principle.
- **"add a bench beside the person"** → `beside`, 27.16% — flanking bands.
  Correct concept, but it returns *both* sides; the instruction does not say
  which, and the field does not yet choose.
- **"remove the person"** → the instance mask, 10.53%, tight and accurate.

Left/right disambiguation works: "remove the left cat" gives 16.95%, "remove
the right cat" 19.57% — different regions, where CLIP-crop selection tied at
margin exactly 0.0000.

## Bug found

`"put a bird above the person"` first resolved to the `on` slot. Cause:
`"on" in text` matches **pers·on**. This is the same substring-matching bug
already fixed once in the EDA earlier in this project, reintroduced in a new
file. Preposition, class-name and position-word matching are now all on word
boundaries via regex.

## Honest limits

Slots are axis-aligned rectangles intersected with free space — a geometric
prior, not a learned affordance. A hat gets the upper third of the person
rather than the crown of the head. The value here is that these are *cheap,
deterministic, supervised targets*: a model can be trained to predict the
refined region directly from (image, instruction), with these as the starting
supervision.

Detection also bounds the whole thing. At conf 0.25 the skier image yields only
`person` and `skis` — an instruction naming anything else cannot resolve.
