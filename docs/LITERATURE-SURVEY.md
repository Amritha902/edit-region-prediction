# Literature Survey — Instruction-Guided Edit-Region Prediction

Amritha S (23BEC1368) · Yugeshwaran P (23BEC1404) — Foundations of Data Science, VIT Chennai.
Compiled September 2026. 42 works, weighted to 2025–2026. Every arXiv identifier
below was resolved against arXiv or the publisher before inclusion.

**Base paper:** AdaptEdit — *Edit Where You Mean: Region-Aware Adapter Injection
for Mask-Free Local Image Editing*, Cai et al., arXiv:2604.23763, April 2026.

---

## 0. Surveys and reference points

| # | Work | Year | Relevance |
|---|---|---|---|
| 1 | Instruction-based Image Editing: A Survey on Data, Models, Evaluation and Applications — arXiv:2607.25642 | 2026 | The field survey. Taxonomy across data, architecture, evaluation, commercial systems |
| 2 | Multimodal Referring Segmentation: A Survey — arXiv:2508.00265 | 2025 | The adjacent field we must distinguish ourselves from |
| 3 | Visual Affordance Prediction: Survey and Reproducibility — arXiv:2505.05074 | 2025 | Affordance as a prediction target |

---

## 1. Instruction-following editors — the task is solved *given a region*

| # | Work | Year | Contribution | What it assumes |
|---|---|---|---|---|
| 4 | Step1X-Edit — arXiv:2504.17761 | 2025 | MLLM parses the instruction into editing tokens, DiT decodes | Localization is internal and implicit |
| 5 | Qwen-Image-Edit (2509 / 2511 / 2.0) | 2025–26 | 20B foundation editor, monthly releases, strong text rendering | 20B parameters; no explicit region |
| 6 | FLUX.1 Kontext [dev] 12B / FLUX.2 32B | 2025 | Open-weights in-context editor | Region never exposed |
| 7 | Visual Autoregressive Modeling for Instruction-Guided Editing — arXiv:2508.15772 | 2025 | Autoregressive alternative to diffusion | Same implicit localization |
| 8 | Image Editing As Programs — arXiv:2506.04158 | 2025 | Edits as composable programs | Program is semantic, not spatial |
| 9 | EditMGT: Masked Generative Transformer — arXiv:2605.10859 | 2026 | Multi-layer attention consolidation for sharper localization | Localization read from attention only |
| 10 | GIDE: Diffusion LLMs for Training-Free Editing — arXiv:2603.21176 | 2026 | Diffusion-LLM editing, training-free | No supervised region |
| 11 | CannyEdit — arXiv:2508.06937 | 2025 | Selective Canny control, dual-prompt | Requires a control signal |
| 12 | NEP: Next Editing Token Prediction — arXiv:2508.06044 | 2025 | Conditions on text, source and **edit-region mask tokens** | The mask is an input, not an output |

**Reading.** Editing quality is no longer the bottleneck. Every system above
edits well *once told where*. Item 12 is the clearest statement of the problem:
it takes the edit-region mask as a conditioning input and does not produce one.

---

## 2. Edit localization — the closest work, and the base paper

| # | Work | Year | Method | Limitation |
|---|---|---|---|---|
| **13** | **AdaptEdit / Edit Where You Mean — arXiv:2604.23763** | **2026** | **Adapters into a frozen DiT; SpatialGate; a MaskPredictor head trained jointly that grounds the region from the instruction, removing the user-mask requirement** | **Masks come from MagicBrush human annotation and rendered text boxes. Its own limitations section names the insertion case as unsolved** |
| 14 | Rethinking Where to Edit — arXiv:2604.20258 | 2026 | Training-free, task-aware; attention cues + feature centroids partition tokens into edit / non-edit | Unsupervised; keeps the largest connected component, so repeated instances break it |
| 15 | WhereEdit — arXiv:2607.20883 | 2026 | AutoMask: sequence-matches changed tokens, aggregates cross-attention, connected-component selection | Derived during inference; does not address empty space |
| 16 | MaskFlow — arXiv:2608.06929 | 2026 | Mask enters the probability path and flow-matching objective | Assumes the mask already exists |
| 17 | Region in Context — arXiv:2510.16772 | 2025 | Human-like semantic reasoning over region and context | Region still tied to existing content |
| 18 | Training-free Geometric Image Editing — arXiv:2507.23300 | 2025 | Geometric edits without training | Geometry given, not inferred |
| 19 | MADiff — arXiv:2412.20062 | 2024 | Two phases: mask prediction, then editing | Fashion domain; predicts over existing garments |
| 20 | iEdit — arXiv:2305.05947 | 2023 | Weak supervision, CLIPSeg masks at test time | Inherits referring-expression semantics |

**Reading.** This is a genuinely active 2026 subfield, which is why the project
sits here. The base paper (13) is the only one that *trains* a predictor to
ground the region from language — our formulation exactly — and its supervision
comes from human annotation.

---

## 3. Benchmarks and metrics — what the field measures

| # | Work | Year | Note |
|---|---|---|---|
| 21 | LocateEdit-Bench — arXiv:2602.05577 | 2026 | 231K edited images, two protocols. Scores localization of edits that **have already occurred** — post-hoc forensics, not prediction |
| 22 | Balancing Preservation and Modification — arXiv:2506.13827 | 2025 | Region- and semantics-aware metric for IIE |
| 23 | GEdit-Bench (via Step1X-Edit) | 2025 | Per-category scores: subject-add 8.28, text_change 8.53, motion_change 3.63 |
| 24 | Adding Affordance Benchmark (via Add-it) | 2024 | Manually annotated suitable insertion areas; affordance 47% → 83% |

**Reading.** A localization benchmark now exists (21), but it evaluates
*detection after editing*. Nothing scores a region *predicted before editing*
from the instruction alone.

---

## 4. Planning and reasoning over instructions

| # | Work | Year | Note |
|---|---|---|---|
| 25 | X-Planner — arXiv:2507.05259 | 2025 | MLLM plans complex instructions before editing; planning is semantic |
| 26 | RePlan (ECCV 2026) | 2026 | MLLM planner + diffusion editor, region-aligned guidance |
| 27 | Counterfactual Segmentation Reasoning — arXiv:2506.21546 | 2025 | Diagnoses pixel-grounding hallucination |
| 28 | SegLLM — arXiv:2410.18923 | 2024 | Multi-round reasoning segmentation |

---

## 5. Object insertion and affordance — where our gap lives

| # | Work | Year | Contribution | Limitation |
|---|---|---|---|---|
| 29 | MADD: Affordance-Aware Object Insertion — arXiv:2412.14462 | 2024 | Dual-stream diffusion denoising RGB **and the insertion mask**; SAM-FB, 3M examples, 3000+ categories | Requires **a position prompt** — you supply the location |
| 30 | Add-it — arXiv:2411.07232 | 2024 | Training-free insertion via weighted extended attention | Training-free, unsupervised placement |
| 31 | SmartMask — arXiv:2312.05039 | 2023 | Mask generation for fine-grained insertion and layout | Uses semantic amodal data, not instructions |
| 32 | US12561956 (patent) — Affordance-based reposing | 2025 | Realistic object insertion into a scene | Requires **a marked region** |
| 33 | US20220335672A1 (patent) | 2022 | Learns joint distribution of object locations and shapes | From semantic maps, not language |

**Reading.** Insertion is well studied, and every method here is handed the
location. The question "where should this object go, given only a sentence?"
is not answered.

---

## 6. Efficiency via region awareness — occupied, and not our claim

| # | Work | Year | Result |
|---|---|---|---|
| 34 | RegionE — arXiv:2510.25590 (ICLR 2026) | 2026 | Skips denoising in unchanged regions; 2.57× / 2.41× / 2.06× on Step1X-Edit, FLUX.1 Kontext, Qwen-Image-Edit |
| 35 | SpotEdit — arXiv:2512.22323 | 2025 | SpotSelector + SpotFusion; 1.95×, training-free |
| 36 | LazyDiffusion — arXiv:2404.12382 (ECCV 2024) | 2024 | Context encoder gives compact global context, decodes only masked pixels; ~10× at a 10% mask |

**Reading.** We make no speed claim against these. All three *consume* a
region — RegionE infers it from an initial denoising pass, LazyDiffusion
requires a user-drawn mask. They are consumers of our output, not rivals.

---

## 7. Structured and persistent scene representations

| # | Work | Year | Note |
|---|---|---|---|
| 37 | I2E: From Image Pixels to Interactive Environments — arXiv:2601.03741 | 2026 | Amodal instance decomposition, depth-ordered object layers |
| 38 | Layer-wise Memory — arXiv:2505.01079 | 2025 | Stores latents and prompt embeddings across sequential edits |
| 39 | MUSE — arXiv:2606.14168 | 2026 | Persistent hierarchical scene memory, incremental update — **3D authoring** |
| 40 | EditSSC — arXiv:2606.09273 | 2026 | Editable semantic occupancy |
| 41 | Follow-Your-Shape (ICLR 2026) | 2026 | Training-free, mask-free trajectory-guided region control |

---

## 8. Text-guided segmentation — the wrong semantics, and a useful precedent

| # | Work | Year | Note |
|---|---|---|---|
| 42 | LISA — arXiv:2308.00692 | 2023 | `<SEG>` token prompts SAM; founded reasoning segmentation |
| 43 | VespaSeg — arXiv:2608.01077 | 2026 | MobileSAM decoder + LoRA — **lightweight text-guided segmentation is viable** |
| 44 | AnchorSeg — arXiv:2604.18562 | 2026 | Language-grounded query banks |
| 45 | SetCon — arXiv:2605.20110 | 2026 | Open-ended referring segmentation |
| 46 | DGSeg — arXiv:2607.04779 | 2026 | Dynamic gating of semantic-spatial predictions |
| 47 | FastSAM — arXiv:2306.12156 | 2023 | YOLOv8-seg backbone; YOLACT prototypes + coefficients. **Our architectural backbone** |

**The distinction that matters.** Referring segmentation returns *what a phrase
denotes*. For "put a hat on the man", the phrase denotes the man; the region
that must change is the empty space above his head. Referring segmentation is
therefore the wrong target, however good it gets. Item 43 matters separately: it
shows a lightweight text-conditioned segmenter is a legitimate design, which is
the regime we work in.

---

# The gap

Three findings, each supported above and each measured in this repository.

### G1 — Editing is solved given a region; producing the region is not

Sections 1 and 2. Every editor performs well when told where. The localization
subfield is barely eighteen months old, and splits into training-free attention
readers (14, 15, 9) that are unsupervised and collapse on repeated instances,
and trained predictors (13) whose supervision is hand-annotated.

*Measured here:* `apple-silicon/experiments/exp01_gap` — a FastSAM + CLIP
selection baseline scores **0/7** across modify, remove and insert, failing
three distinct ways: granularity (whole person returned for "the jacket"),
spatial language (top-two candidates tie at margin exactly **+0.0000** for
"the cat on the left"), and structure (asked for empty snow, it returned skis).

### G2 — The edit region is frequently not an object, and insertion has no referent

The base paper states this itself:

> "Geometry-changing edits with no localized source region ('put a hat on the
> dog' where the dog has no hat): the GT mask covers the new content's location
> but there is no clean source signal to 'protect' outside it."

Every insertion method in Section 5 is handed the position (29, 32) or places
it unsupervised (30). Selection-based methods cannot reach empty space *in
principle*: there is no mask to select and no token to attend to.

### G3 — The supervision exists, but it is roughly 9× too coarse

This is the finding that matters, and it is measured, not argued.

The base paper trains its MaskPredictor on MagicBrush human-annotated edit
regions. We downloaded the MagicBrush dev split (528 turns, 72 images) and
compared every human mask against the region that actually changed between the
source and target image, recovered by CIE76 ΔE in L\*a\*b\*.

| edit type | n | human mask | region that actually changed | ratio | recall | **precision** | **IoU** |
|---|---:|---:|---:|---:|---:|---:|---:|
| insert | 103 | 60.0% | 7.0% | **11.9×** | 97.8% | 11.9% | **0.12** |
| modify | 390 | 62.9% | 11.2% | 8.5× | 96.5% | 17.1% | 0.17 |
| remove | 35 | 63.6% | 8.9% | 7.1× | 96.2% | 13.4% | 0.13 |
| **all** | **528** | **62.4%** | **10.3%** | **9.1×** | **96.7%** | **15.8%** | **0.16** |

Recall is 96.7%: the annotations are *correct*, in that the edit almost always
falls inside them. Precision is **15.8%**: only about a sixth of the annotated
region actually changes. Median IoU against the true changed region is **0.16**,
and insertions are the worst case at 11.9× and IoU 0.12.

These are region-of-interest scribbles — a human roughly brushing "work
somewhere in here" for an inpainting tool — not edit regions. A predictor
supervised on them learns to emit regions an order of magnitude too large, which
is the most likely explanation for why the base paper's own limitations section
singles out insertion as unsolved.

*Correction of record:* an earlier draft of this survey claimed insertion
regions "cannot be annotated". That was wrong — MagicBrush annotates them. The
defensible claim is the measured one above: they are annotated ~9× too coarsely.
The first measurement was also run with the mask polarity inverted (MagicBrush
uses bright = preserve); the table above is the corrected version.

Reproduce: `apple-silicon/experiments/exp07_magicbrush/` ·
raw per-row data in `LITERATURE/magicbrush_mask_precision.csv`.

---

# How this project mitigates it

### M1 — Manufacture supervision an order of magnitude tighter than annotation

Apply the instruction with an editing model, then recover the changed region by
CIE76 ΔE differencing in L\*a\*b\*. Where a hat *landed* is ground truth for
where a hat *should go* — and it is the *precise* region, not a 9×-oversized
brush stroke. This is the same measurement used to expose G3, turned into a
supervision signal. Addresses **G3** and unlocks **G2**.

*Status:* 2000 grounded instruction specs over 1000 COCO images, balanced
669 insert / 668 remove / 663 modify, 887 carrying spatial qualifiers, each
tagged with the target instance box. Pilot run; failure mode identified and
quantified (below).

### M2 — Synthesise the region instead of selecting one

FastSAM's head produces 32 YOLACT-style prototypes; coefficients normally come
from a detection, which is precisely why output is object-bound. Predicting
those coefficients from the **instruction** instead lifts that constraint —
prototypes are not objects, so a text-conditioned combination can describe
empty space. Addresses **G2** at 71.8M parameters, against 0.6B–20B in
Sections 1–2, with (43) as precedent that this weight class is viable.

### M3 — Represent actionable empty space explicitly

An edit-ready field computed once per image holds instance regions *and* free
space, with affordance slots (on / above / beside / below). Addresses **G2**
directly and gives geometric weak supervision for M2.

*Measured:* `exp03` — 20/20 instructions resolved; "put a bird above the
person" returns empty sky, which the 0/7 baseline cannot reach. `exp04` —
analysis amortises across instructions, 164× at N=500, 48 µs per query,
break-even at N=3.

### M4 — Score the region, not only the image

Report IoU / Dice against the derived mask, instance-hit rate against the
recorded target box, and collateral change outside the region. LocateEdit-Bench
(21) scores edits post hoc; nothing scores a *predicted* region. Addresses
**G1**'s evaluation half.

---

# Honest limitations

Stated because a reviewer will find them.

1. **Modify edits leak badly.** Measured over `image_guidance_scale` 1.5→3.0:
   remove improves 99.8% → 10.8% leak, but recolouring stays ~91% at every
   setting, because the editor restyles globally. Modify needs mask-composited
   supervision rather than trusting the editor.
2. **The ΔE threshold was tuned synthetically.** On real pairs global
   photometric drift can dominate the signal; a drift filter is implemented
   (`topology.drift_score`) but the full run is not yet generated.
3. **Affordance slots are geometric priors**, axis-aligned rectangles
   intersected with free space — a hat gets the upper third of a person, not
   the crown. They are weak supervision, not the final output.
4. **No trained model yet.** M2 is designed and its supervision is being
   generated; the localizer is not trained, so no IoU against the base paper is
   claimed.
5. **The amortisation result covers scene analysis only**, not diffusion, so it
   is not comparable to RegionE or SpotEdit.
