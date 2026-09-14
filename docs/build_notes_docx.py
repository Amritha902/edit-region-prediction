import sys; sys.path.insert(0,".")
from mkdocx import *

d = base_doc()
for _ in range(2): d.add_paragraph()
para(d,"DA2 — writing notes", size=20, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space=6)
para(d,"Facts, numbers and structure for the three required sections", size=12.5,
     italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space=18, color=MUTE)
para(d,"These are notes to write FROM, not text to copy. Every bullet is a fact or a number "
       "from your own experiments — the sentences have to be yours. Turnitin checks both "
       "similarity and AI score, and both must be under 10%.",
     size=11, align=WD_ALIGN_PARAGRAPH.CENTER, space=6, color=MUTE)
para(d,"Deadline 17 Sep · 2 upload attempts · final document to Teams",
     size=11, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space=20, color=MUTE)
d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

def bullets(items, style="List Bullet"):
    for t in items:
        p=d.add_paragraph(t, style=style); p.paragraph_format.space_after=Pt(2)
        for r in p.runs: r.font.size=Pt(10.5)

def note(t):
    p=d.add_paragraph(); r=p.add_run(t); r.font.size=Pt(10); r.italic=True
    r.font.color.rgb=MUTE; p.paragraph_format.space_after=Pt(8)

# ── 1 ────────────────────────────────────────────────────────────────────
d.add_heading("1  Introduction — what to cover", level=1)
note("Aim for roughly 500–700 words. Order below works; the wording must not be mine.")
d.add_heading("The problem", level=2)
bullets([
 "An editor is told WHAT to change, never WHERE.",
 "Diffusion editors (InstructPix2Pix, Qwen-Image-Edit, FLUX) denoise the whole frame.",
 "Two failures follow: the intended edit comes out weak, and unrelated regions drift in colour/texture.",
 "Fix: predict the region first, gate the editor to it. Splits the task into localization, then editing.",
])
d.add_heading("The hard case — this is your hook", level=2)
bullets([
 "“Put a hat on the dog” names an object that is not in the image.",
 "There is no source region to point at.",
 "Every method that works by referring to visible content fails on this class.",
 "Referring segmentation returns what a phrase DENOTES — for that instruction it returns the dog; the correct answer is the empty space above it.",
])
d.add_heading("Base paper and the gap", level=2)
bullets([
 "AdaptEdit, arXiv:2604.23763, April 2026 — the only work that trains a predictor to ground the region from the instruction.",
 "Scale: 20.43 B frozen diffusion transformer + 4.67 B adapters + 351 M condition encoder.",
 "Cannot be reproduced on a 26 GB laptop.",
 "It reports NO mask IoU — the MaskPredictor is never scored directly.",
 "Its own limitations section names our gap: geometry-changing edits with no localized source region.",
 "Its masks come from MagicBrush human annotation and rendered text boxes — explicitly not pixel differences.",
])
d.add_heading("Why this is a data-science problem", level=2)
bullets([
 "No benchmark scores a PREDICTED edit region. LocateEdit-Bench scores edits that already happened.",
 "That absence is why a 2026 paper built around a MaskPredictor can omit a mask IoU without objection.",
])
d.add_heading("Objectives — state these as a numbered list", level=2)
bullets([
 "Test whether an off-the-shelf segmentation + retrieval pipeline suffices, and characterise how it fails.",
 "Construct supervision that does not depend on human region annotation.",
 "Train a predictor over frozen features within one laptop's compute.",
 "Evaluate the predicted region directly — the measurement the base paper omits.",
 "Determine whether predicting the region improves the resulting edit, and by how much.",
 "Establish the upper bound the representation imposes, and report the fraction reached.",
])

# ── 2 ────────────────────────────────────────────────────────────────────
d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
d.add_heading("2  Literature Survey — what to cover", level=1)
note("Roughly 900–1200 words. 45 works, all 2024–2026. Cite with the links from the survey file.")
d.add_heading("Open with the classification — this is the strongest framing", level=2)
bullets([
 "Sorting 45 works by what each DOES with the edit region beats a chronological reading.",
 "17 works CONSUME a region given to them — NEP takes the mask as an input; MaskFlow builds it into the objective; LazyDiffusion needs a user-drawn mask.",
 "11 works READ a region out of a running generator — WhereEdit aggregates cross-attention; RegionE infers it after denoising starts. Unsupervised, and blind to empty space.",
 "Only 2 works PREDICT a region from the instruction — AdaptEdit, and MADiff which is fashion-only.",
 "15 works score or frame the problem, and none scores a predicted region.",
 "One line to land: of 45 recent works, two produce the region from language.",
])
d.add_heading("Cluster notes — one short paragraph each", level=2)
bullets([
 "Instruction editors (9): excellent at editing, silent on localization. Step1X-Edit 19 B, Qwen 20 B, FLUX 12–32 B. NEP makes the asymmetry explicit by taking the mask as input.",
 "Edit localization (7): splits into attention-readers and region-consumers. WhereEdit states outright it does not handle adding into empty regions.",
 "Insertion and affordance (4): the field solves insertion GIVEN a location. MADD needs a position prompt; Add-it is training-free so placement emerges. Patents say the same in legal terms — US12561956 requires “an original image with a marked region”.",
 "Region-aware efficiency (3): RegionE, SpotEdit, LazyDiffusion all exploit a region for speed and all assume it arrives from elsewhere. Useful to you: this proves a cheap per-instruction query is a recognised structure.",
 "Text-guided segmentation (5): referring semantics are the wrong semantics. CLIPSeg gets 0.1849 overall but 0.054 on insertion. VespaSeg proves lightweight text-conditioned segmentation works.",
 "Benchmarks (4): LocateEdit-Bench scores 231 K already-edited images — post-hoc forensics, not prediction.",
 "Planning and reasoning (4): X-Planner and RePlan decompose instructions semantically, never to pixels, and need an MLLM in the loop.",
 "Structured scene representations (6): I2E and layer-wise memory represent things that EXIST. None makes empty space addressable.",
 "Surveys (3): the 2026 instruction-editing survey confirms localization is treated as a sub-component, never a first-class output.",
])
d.add_heading("The four gaps — state each with evidence AND your measurement", level=2)
bullets([
 "G1 Producing a region is unsolved while editing given one is not. Your evidence: FastSAM + CLIP selection scores 0 of 7, failing on granularity, spatial language, and insertion in principle.",
 "G2 The edit region is often not an object. Evidence: the base paper says so itself; WhereEdit says so. Your measurement: CLIPSeg 0.1849 overall but 0.054 on insertion.",
 "G3 Supervision binds harder than architecture. Your measurement: MagicBrush masks cover 62.4% of the frame against a true 10.3% — 9.1x too large, precision 15.8%, worst on insertions at 11.9x.",
 "G4 Nothing scores a predicted region. Evidence: LocateEdit-Bench, GEdit-Bench, the affordance benchmark — all score something downstream of the region.",
])
d.add_heading("What each line of work improves vs never reports", level=2)
bullets([
 "Instruction editors improve GEdit-Bench category scores; never report mask IoU.",
 "AdaptEdit improves text_change 8.53, subject-add 8.28; never scores its own MaskPredictor.",
 "Attention readers improve edit quality with no training; never report IoU against a true changed region.",
 "Region-aware efficiency improves wall-clock; never says where the region comes from.",
 "Referring segmenters improve RefCOCO-style IoU on denoted objects — the wrong target for editing.",
])
d.add_heading("Close with your position", level=2)
bullets([
 "Available to claim: predicting the region from the instruction is nearly vacant; deriving supervision from pixel differences appears unattempted here; reporting a mask IoU is novel by absence.",
 "NOT available to claim: the mechanism. FiLM and BCE+Dice are AdaptEdit's choices too. Say so — it is stronger than pretending otherwise.",
 "Honest limit: 39 of the 45 are from 2025–2026, so the area moves fast. Rest the claim on the measured bound, which is a property of the representation.",
])

# ── 3 ────────────────────────────────────────────────────────────────────
d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
d.add_heading("3  Methodology — what to cover", level=1)
note("Roughly 900–1200 words. This section is mostly facts, so it is the easiest to write in your own words.")
d.add_heading("Frozen feature extraction", level=2)
bullets([
 "YOLOv8x-seg (FastSAM) backbone, frozen, used only as a feature source.",
 "Gives 32 mask prototypes at stride 4 — 160 x 160 maps for a 640 x 640 input.",
 "Plus a 640-dimensional image context vector from the SPPF layer.",
 "Instruction encoded by a frozen CLIP text encoder to 512 dimensions.",
 "Prototypes are spatial basis functions: any mask is a linear combination of them through a sigmoid.",
 "Neither backbone trains. Only the head learns — that is what makes it run on one laptop.",
])
d.add_heading("Two heads", level=2)
bullets([
 "v1, global coefficients: one 32-d vector + bias per image-instruction pair. Text modulates image context by FiLM; an MLP emits coefficients. 1,399,073 parameters. Same combination applies everywhere in the frame.",
 "v2, spatial field: a 20 x 20 field of coefficients, bilinearly upsampled, plus a global branch. 4,693,633 parameters. Different locations can use different prototype mixtures.",
 "Why v2: a single vector forces one mixture everywhere, which suits objects and not empty space. An earlier experiment showed v1 saturating at 1.4 M, so width was never the constraint — expressiveness was.",
])
d.add_heading("Supervision — explain WHY, this is your contribution", level=2)
bullets([
 "Do NOT use MagicBrush's own masks as targets: 62.4% of frame against 10.3% true change.",
 "Instead derive the target from the real source/target image pair by CIE76 colour difference in L*a*b*.",
 "Threshold 12.0, chosen from a noise-robustness sweep (holds at sigma=8).",
 "Morphological open then close with a 5 x 5 kernel; drop connected components under 40 px.",
 "Reject samples whose changed area is outside 0.4%–45% — too small to learn, or a whole-frame restyle.",
 "The idea in one line: where a hat LANDED is ground truth for where a hat SHOULD GO.",
 "Worth mentioning: MagicBrush masks are bright = PRESERVE. Assuming the opposite gave IoU 0.001 before it was caught.",
])
d.add_heading("Loss and optimisation", level=2)
bullets([
 "BCE + 2 x Dice. Gradient-norm clipping at 1.0.",
 "AdamW, learning rate 1e-3 cosine to zero, weight decay 0.01, batch 32, 60 epochs.",
 "12 seeds (1368, 1–11), seeding torch, numpy, random and the Metal generator.",
])
d.add_heading("Dataset", level=2)
bullets([
 "MagicBrush: all 51 training shards, all 4 dev shards, 25 GB of Parquet.",
 "8,807 rows -> 8,306 usable after filtering (501 rejected as degenerate).",
 "Kinds: modify 6,045, insert 1,828, remove 433.",
 "Held-out dev: 528 turns -> 503 usable.",
])
d.add_heading("Split integrity — a strong thing to include", level=2)
bullets([
 "MagicBrush is multi-turn: one photograph yields several successive edits.",
 "If the split were by turn rather than by image, the same photo would sit on both sides.",
 "So every source and target image was hashed (SHA-1) on both sides and intersected.",
 "Result: 794 distinct dev images, 13,317 distinct train images, 0 overlap.",
 "Say that this was run BEFORE the results were trusted, not afterwards to defend them.",
])
d.add_heading("Protocol", level=2)
bullets([
 "Checkpoint epoch selected on a held-out 15% slice of TRAIN, never on dev.",
 "Dev evaluated exactly once per seed, at full resolution against the undilated target.",
 "Welch t-test on seed means; one-sample t-tests against fixed baselines.",
 "Frozen features precomputed once into memory-mapped arrays (13.8 GB) so each epoch touches only the head.",
])
d.add_heading("Why each design choice — examiners ask this", level=2)
bullets([
 "Predict coefficients rather than select a mask: selection can only return an object already present, so insertion is impossible in principle. Evidence: 0 of 7.",
 "Freeze both backbones: the backbone depends on the image alone, so pay for it once and run the head per instruction. Evidence: 5.30 ms vs CLIPSeg's 54.3 ms.",
 "Derive targets by colour difference: the dataset's masks are 9.1x too large, so training on them teaches the wrong region.",
 "Spatial field over global vector: one vector forces a single mixture everywhere. Evidence: insert 0.1186 vs 0.0972.",
])

d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
d.add_heading("Writing checklist", level=1)
bullets([
 "Write each section in your own words with these notes beside you — do not paste sentences from any document.",
 "Numbers are facts and cannot be plagiarised; the sentences around them must be yours.",
 "Quote the base paper's limitation directly and put it in quotation marks with the citation — a quoted, attributed line is not plagiarism.",
 "Cite all 45 works with links; a reference list raises similarity slightly but that is expected and excluded in most Turnitin settings.",
 "Upload once, read the report, fix what it flags, upload the second time. Only two attempts.",
 "Deadline 17 Sep. Final document goes to Teams, not Turnitin.",
], style="List Number")
d.save("DA2_writing_notes.docx"); print("wrote DA2_writing_notes.docx")
