# Literature — Instruction-Guided Edit-Region Prediction

Amritha S (23BEC1368) · Yugeshwaran P (23BEC1404) — Foundations of Data Science, VIT Chennai.
Compiled September 2026. 50 works, weighted to 2025–2026.

## Files here

| File | What it is |
|---|---|
| `Literature_Survey_Matrix.xlsx` | The main deliverable. 3 sheets: **Literature Matrix** (50 papers, what they built, backbone, params, what they improve, what's missing), **Gap and Mitigation** (G1–G4 and how we address each), **Base Paper** (full breakdown of AdaptEdit) |
| `pdfs/` | Downloaded PDFs, named `<arxivID>_<ShortName>.pdf` |
| `README.md` | This file — every link, in case a PDF failed to download |

## Base paper

**AdaptEdit — *Edit Where You Mean: Region-Aware Adapter Injection for Mask-Free Local
Image Editing*.** Cai et al., arXiv:2604.23763, 26 April 2026.
<https://arxiv.org/abs/2604.23763>

Chosen because it is the only work that **trains a predictor to ground the edit region
from the instruction** — our exact formulation — and because its own limitations section
names our gap:

> "Geometry-changing edits with no localized source region ('put a hat on the dog' where
> the dog has no hat): the GT mask covers the new content's location but there is no clean
> source signal to 'protect' outside it."

Its masks come from MagicBrush human annotation and rendered text boxes — explicitly not
from pixel differences. That is precisely why it cannot solve the insertion case, and
precisely what our ΔE-derived supervision provides.

**Backbone note:** AdaptEdit runs on a 12–20B diffusion transformer and cannot be trained
on a 24 GB laptop. We adopt its formulation and position against it; our implementation
uses FastSAM (71.8M).

---

## Cluster A — Instruction-following editors (task solved *given* a region)

- Step1X-Edit — <https://arxiv.org/abs/2504.17761>
- Qwen-Image-Edit 2509 / 2511 / 2.0 — <https://github.com/QwenLM/Qwen-Image>
- FLUX.1 Kontext [dev] / FLUX.2 — <https://blackforestlabs.ai/>
- Visual Autoregressive Modeling for Instruction-Guided Editing — <https://arxiv.org/abs/2508.15772>
- Image Editing As Programs — <https://arxiv.org/abs/2506.04158>
- EditMGT: Masked Generative Transformer — <https://arxiv.org/abs/2605.10859>
- GIDE: Diffusion LLMs for Training-Free Editing — <https://arxiv.org/abs/2603.21176>
- CannyEdit — <https://arxiv.org/abs/2508.06937>
- NEP: Next Editing Token Prediction — <https://arxiv.org/abs/2508.06044>

## Cluster B — Edit localization (the core, where the base paper sits)

- **AdaptEdit / Edit Where You Mean [BASE]** — <https://arxiv.org/abs/2604.23763>
- Rethinking Where to Edit: Task-Aware Localization — <https://arxiv.org/abs/2604.20258>
- WhereEdit: Mask-aware Local Latent Editing — <https://arxiv.org/abs/2607.20883>
- MaskFlow — <https://arxiv.org/abs/2608.06929>
- Region in Context — <https://arxiv.org/abs/2510.16772>
- Training-free Geometric Image Editing — <https://arxiv.org/abs/2507.23300>
- MADiff: Fashion Editing with Mask Prediction — <https://arxiv.org/abs/2412.20062>
- iEdit: Localised Text-guided Editing — <https://arxiv.org/abs/2305.05947>

## Cluster C — Benchmarks and metrics

- LocateEdit-Bench — <https://arxiv.org/abs/2602.05577>
- Balancing Preservation and Modification (metric) — <https://arxiv.org/abs/2506.13827>
- GEdit-Bench (via Step1X-Edit) — <https://arxiv.org/abs/2504.17761>
- Adding Affordance Benchmark (via Add-it) — <https://arxiv.org/abs/2411.07232>

## Cluster D — Planning and reasoning

- X-Planner — <https://arxiv.org/abs/2507.05259>
- RePlan (ECCV 2026) — <https://github.com/JIA-Lab-research/RePlan>
- Counterfactual Segmentation Reasoning — <https://arxiv.org/abs/2506.21546>
- SegLLM — <https://arxiv.org/abs/2410.18923>

## Cluster E — Insertion and affordance (where our gap lives)

- MADD: Affordance-Aware Object Insertion — <https://arxiv.org/abs/2412.14462>
- Add-it: Training-Free Object Insertion — <https://arxiv.org/abs/2411.07232>
- SmartMask — <https://arxiv.org/abs/2312.05039>
- US12561956 — Affordance-based reposing (patent) — <https://patents.google.com/?q=12561956>
- US20220335672A1 — Context-aware synthesis and placement — <https://patents.google.com/patent/US20220335672A1/en>
- US12462449B2 — Mask conditioned image transformation, Adobe — <https://patents.google.com/patent/US12462449B2/en>

## Cluster F — Region-aware efficiency (occupied; not our claim)

- RegionE (ICLR 2026) — <https://arxiv.org/abs/2510.25590> · code <https://github.com/Peyton-Chen/RegionE>
- SpotEdit — <https://arxiv.org/abs/2512.22323>
- LazyDiffusion (ECCV 2024) — <https://arxiv.org/abs/2404.12382>

## Cluster G — Structured / persistent scene representations

- I2E: From Image Pixels to Interactive Environments — <https://arxiv.org/abs/2601.03741>
- Layer-wise Memory — <https://arxiv.org/abs/2505.01079>
- MUSE: Agentic 3D Scene Authoring — <https://arxiv.org/abs/2606.14168>
- EditSSC — <https://arxiv.org/abs/2606.09273>
- Follow-Your-Shape (ICLR 2026) — <https://github.com/mayuelala/FollowYourShape>
- Free-Form Scene Editor — <https://arxiv.org/abs/2511.13713>

## Cluster H — Text-guided segmentation (wrong semantics, useful precedent)

- LISA: Reasoning Segmentation via LLM — <https://arxiv.org/abs/2308.00692>
- VespaSeg: Resource-Aware RES (lightweight precedent) — <https://arxiv.org/abs/2608.01077>
- AnchorSeg — <https://arxiv.org/abs/2604.18562>
- SetCon — <https://arxiv.org/abs/2605.20110>
- DGSeg — <https://arxiv.org/abs/2607.04779>
- RSAgent — <https://arxiv.org/abs/2512.24023>
- **FastSAM (our backbone)** — <https://arxiv.org/abs/2306.12156>

## Cluster I — Surveys

- Instruction-based Image Editing: A Survey — <https://arxiv.org/abs/2607.25642>
- Multimodal Referring Segmentation: A Survey — <https://arxiv.org/abs/2508.00265>
- Visual Affordance Prediction: Survey and Reproducibility — <https://arxiv.org/abs/2505.05074>

---

## The gap, in one page

**G1 — Editing is solved given a region; producing the region is not.**
Every editor in Cluster A performs well when told where. NEP takes the edit-region mask as
an *input*. Cluster B is barely 18 months old and splits into training-free attention
readers (unsupervised, collapse on repeated instances) and trained predictors whose
supervision is hand-annotated.
*Measured:* our FastSAM+CLIP baseline scores **0/7** — granularity, spatial language
(margin exactly **+0.0000** on "the cat on the left"), and structure.

**G2 — The edit region is often not an object, and insertion has no referent.**
The base paper says so itself. Every insertion method is *handed* the position: MADD needs
a position prompt, US12561956 needs a marked region, Add-it is unsupervised.
*Measured:* EditField resolves "put a bird above the person" to empty sky — a region the
0/7 baseline cannot reach in principle.

**G3 — Supervision, not architecture, is the binding constraint.**
The base paper's masks are human-annotated. Nobody can annotate where a hat *should* go on
a dog that has no hat. This is why G2 survives in a 2026 paper that already has adapters,
a SpatialGate and a trained MaskPredictor.

**G4 — Nothing scores a predicted edit region.**
LocateEdit-Bench scores edits that already happened — forensics, not prediction.

## How we mitigate it

1. **Manufacture the supervision** — apply the instruction, recover the changed region by
   CIE76 ΔE in L\*a\*b\*. Where a hat *landed* is ground truth for where a hat *should go*.
2. **Synthesise the region, don't select it** — predict FastSAM's prototype coefficients
   from the instruction. Prototypes aren't object-bound, so the output can be empty space.
3. **Represent actionable empty space** — affordance slots (on / above / beside / below)
   intersected with free space, computed once per image.
4. **Score the region** — IoU/Dice against the derived mask, instance-hit rate against the
   recorded target box, collateral change outside the region.

---

## Note on the PDFs

The 42 archived PDFs (630 MB) are deliberately not committed. Every one is on
arXiv and every link is in the table above and in the survey document, so they
re-download in minutes. The two .xlsx matrices ARE committed: the 50-paper
classification with what each work built and what it is missing was assembled by
hand and would take days to rebuild.
