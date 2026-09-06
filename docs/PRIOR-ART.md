# Prior-art position, September 2026

Checked before committing engineering effort. Two directions were considered
and closed; one remains open.

## Closed — region-aware acceleration

| work | what it does | result |
|---|---|---|
| RegionE, ICLR 2026 ([2510.25590](https://arxiv.org/abs/2510.25590)) | detects edited vs unedited regions, skips denoising in unchanged areas, training-free | 2.06–2.57× on Step1X-Edit, FLUX.1 Kontext, Qwen-Image-Edit |
| SpotEdit ([2512.22323](https://arxiv.org/abs/2512.22323)) | SpotSelector skips stable regions by perceptual similarity, SpotFusion blends | 1.95×, training-free |

"Find the changed region, spend less compute elsewhere" is occupied.

## Closed — minimum sufficient context / adaptive framing

**LazyDiffusion**, Adobe Research + TAU, ECCV 2024
([2404.12382](https://arxiv.org/abs/2404.12382)). States the premise directly:
prior work either *"regenerate[s] the full canvas, wasting time and
computation, or confine[s] processing to a tight rectangular crop around the
mask, ignoring the global image context altogether."* Their answer is a context
encoder producing *"a compact global context tailored to the region to
generate"*, with a decoder that synthesises only the masked pixels — ~10×
speedup at a 10% mask.

So "don't crop, don't do full-frame, find the sufficient context" is not white
space. It is a two-year-old ECCV paper from Adobe.

## Closed — text-conditioned edit-region masks (patent)

Adobe **US12462449B2** (filed May 2023, granted Nov 2025) claims generating
masks that define local edit regions by conditioning on a text prompt — as
per-layer blending masks inside a generator. It does not claim prototype
coefficients, and it does not cover insertion into empty regions.

## Open — predicting the region from the instruction, including empty space

Every system above **consumes** a region; none produces one from language:

- LazyDiffusion requires a **user-drawn mask**.
- RegionE estimates the changed region from an initial denoising pass — i.e.
  after generation has begun, not from the instruction.
- Affordance insertion patents (US12561956) require *a marked region*.
  US20220335672A1 predicts placement, but from semantic maps, not instructions.

And the insertion case has no referent to segment: "put a hat on the man"
designates empty space above his head. Selection-based methods cannot reach it
in principle — measured at 0/7 in `apple-silicon/experiments/exp01_gap`.

**Position.** Predict the edit region from image + instruction, covering
add/modify/remove uniformly, supervised from generated before/after pairs.
That output is an *input* to RegionE, SpotEdit and LazyDiffusion rather than a
competitor to them. Claims about beating those systems on FLOPs are not
supportable from a laptop and are not made.
