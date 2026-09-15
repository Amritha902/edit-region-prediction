import sys; sys.path.insert(0,".")
from mkdocx import *

d = base_doc()
for _ in range(2): d.add_paragraph()
para(d,"DA2 writing notes", size=20, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space=6)
para(d,"Facts and numbers for the three sections", size=12.5,
     italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space=18, color=MUTE)
para(d,"Notes to write from, not text to copy. Every bullet is a fact or a number from your own "
       "runs. The sentences have to be yours. Turnitin checks similarity and AI score; both under 10%.",
     size=11, align=WD_ALIGN_PARAGRAPH.CENTER, space=6, color=MUTE)
para(d,"Due 15 Sep · 2 upload attempts · final file to Teams",
     size=11, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space=20, color=MUTE)
d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

def bullets(items, style="List Bullet"):
    for t in items:
        p=d.add_paragraph(t, style=style); p.paragraph_format.space_after=Pt(2)
        for r in p.runs: r.font.size=Pt(10.5)

def note(t):
    p=d.add_paragraph(); r=p.add_run(t); r.font.size=Pt(10); r.italic=True
    r.font.color.rgb=MUTE; p.paragraph_format.space_after=Pt(8)

# ── rubric map ───────────────────────────────────────────────────────────
d.add_heading("Rubric map", level=1)
note("The five evaluation criteria, and which of your own work answers each one. "
     "Every number here is measured, and the file that produced it is named.")

d.add_heading("1  Data Collection and Understanding", level=2)
bullets([
 "Dataset: MagicBrush. All 51 training shards and all 4 dev shards, 25 GB of Parquet.",
 "Source: released on HuggingFace, built on COCO images, with human annotators performing real edits.",
 "Why this one: it is the only public set with genuine source and target image pairs for instruction edits, and the base paper trains on it too, so the comparison is fair.",
 "Variables per row: source_img (bytes), target_img (bytes), mask_img (bytes), instruction (string), turn index, img_id.",
 "Scale: 8,807 training rows and 528 dev turns.",
 "Structure worth stating: it is multi-turn, so one photograph produces several edits in sequence, and turn 2's source is turn 1's target.",
 "Class balance: modify 6,045, insert 1,828, remove 433. Heavily imbalanced, and the rarest class is also the hardest one.",
 "Understanding that changed the design: the supplied human masks cover 62.4% of the frame against a true change of 10.3%, so they are 9.1 times too large, with precision 15.8% and IoU 0.16. Insertions are worst at 11.9 times.",
 "Second finding: the mask polarity is bright equals preserve. Assuming the opposite gave IoU 0.001 before it was caught.",
 "Code: datagen/mask_extraction.py, train/dataset.py, experiments/exp07_magicbrush.",
])

d.add_heading("2  Data Preprocessing", level=2)
bullets([
 "Duplicates and leakage. Every source and target image was hashed with SHA-1 on both sides and the two sets intersected. Result: 794 distinct dev images, 13,317 distinct train images, 0 overlap. Run before the results were trusted, not after.",
 "A second dedup on img_id was needed for the qualitative figure, because byte hashing fails on multi-turn rows where one image appears as both a target and the next source.",
 "Outliers. Samples whose changed area falls outside 0.4% to 45% are rejected. Too small to learn from, or a whole-frame restyle rather than a local edit.",
 "Result of that filter: 501 of 8,807 rows dropped, so 8,306 usable. That is 5.7% rejected.",
 "Noise removal. Morphological open then close with a 5 by 5 kernel, then connected components under 40 px are discarded.",
 "Transformation. Images converted BGR to L*a*b*, and the change map is CIE76 colour distance with a threshold of 12.0. That threshold came from a noise robustness sweep and holds at sigma 8.",
 "Resizing. Target resized to the source size before differencing. Source resized to 640 by 640 bilinear for the backbone. Masks downsampled to the 160 by 160 prototype grid.",
 "Encoding, text. Instructions go through CLIP BPE tokenisation to a 512-d vector, L2 normalised.",
 "Encoding, categorical. The edit kind is derived by rule from the leading verb: put or add gives insert, remove or delete gives remove, everything else gives modify.",
 "Storage. Features cast to float16 and written to memory-mapped arrays totalling 13.8 GB, so training never re-runs the backbone.",
 "Code: train/dataset.py, train/precompute_full.py, train/leakage_check.py.",
])

d.add_heading("3  Feature Engineering and Selection", level=2)
bullets([
 "The central feature decision: treat FastSAM's 32 mask prototypes as a spatial basis rather than as objects. Normally their coefficients come from a detection, which is why the output can only be something already in the image. We predict the coefficients from the instruction instead.",
 "Feature 1: 32 prototypes at stride 4, so 160 by 160 maps. Chosen because they are fine grained, linearly combinable, and already computed by the backbone, so they cost nothing extra.",
 "Feature 2: a 640-d global image descriptor, average pooled from the SPPF layer.",
 "Feature 3: a 512-d CLIP text embedding of the instruction.",
 "Engineered feature 4: 12 fixed geometric functions appended to the basis. They are 1, x, y, x squared, y squared, xy, x cubed, y cubed, and sin and cos of pi x and pi y, on a grid normalised to minus 1 through 1. They carry no parameters.",
 "Selection evidence: a least squares fit measured each function's contribution to the achievable ceiling. Adding them lifts insert from 0.197 to 0.324 and remove from 0.464 to 0.581.",
 "Control: the same code accepts a random basis of 12 Gaussian-blurred noise fields of matched size, so any gain has to beat extra capacity alone. The reported runs use no extra basis, so the headline result does not depend on this.",
 "Rejected alternatives, and why: raw pixels (no semantics, and far too many dimensions for a laptop), detection boxes (can only name existing objects), referring segmentation output (returns what a phrase denotes, which is the wrong target for insertion).",
 "Code: train/model_v2.py, experiments exp10 and exp11.",
])

d.add_heading("4  Model Implementation", level=2)
bullets([
 "Two heads were built and compared, not one. v1 predicts a single coefficient vector for the frame, 1,399,073 parameters. v2 predicts a coefficient field at 20 by 20, upsampled, 4,693,633 parameters.",
 "Justification for v2: an earlier run showed v1 saturating at 1.4 M parameters, so width was not the constraint. The constraint was that one vector forces the same mixture of prototypes everywhere, which suits objects and not empty space.",
 "Both backbones are frozen, so only the head learns. That is what makes the whole thing trainable on one laptop.",
 "Loss: binary cross-entropy plus 2 times Dice. Justification: edit regions average about 10% of the frame, so BCE alone collapses to predicting background everywhere. Dice is scale invariant and stops that.",
 "Optimisation: AdamW, learning rate 1e-3 cosine to zero, weight decay 0.01, batch 32, 60 epochs, gradient norm clipped at 1.0.",
 "Repetition: 12 seeds, seeding torch, numpy, random and the Metal generator, so the result is a distribution and not one lucky run.",
 "Model selection: the checkpoint epoch is chosen on a held-out 15% slice of train. Dev is evaluated once per seed and never used for selection.",
 "Baselines implemented for comparison: CLIPSeg, FastSAM with CLIP selection, MagicBrush human masks, and a random control.",
 "Result: 0.2065 IoU plus or minus 0.0072 against CLIPSeg's 0.1849 and human masks at 0.1511, with p = 4.1e-09.",
 "Statistics: Welch t-tests on seed means, one-sample t-tests against fixed baselines, Cohen's d, and 95% confidence intervals.",
 "Stage 2 closes the loop: the predicted region gates Stable Diffusion inpainting, and a 20-cell sweep over threshold and dilation found the operating point rather than guessing it.",
 "Code: train/model.py, train/model_v2.py, train/train_clean.py, train/clean_eval_full.py, train/metrics_full.py, train/stage2_sd_sweep.py.",
])

d.add_heading("5  Code Quality and Documentation", level=2)
bullets([
 "Repository: github.com/Amritha902/edit-region-prediction, with every experiment, checkpoint and figure committed.",
 "Every module opens with a docstring that states why the thing exists, not just what it does.",
 "Reproducibility: seeds are fixed and recorded inside each checkpoint alongside n_train and dev_iou, so a result can always be traced to the run that produced it.",
 "Long jobs are resumable. The precompute checkpoints after every batch. The sweep runs one cell per job with a resume guard, after three runs died at session teardown.",
 "Baselines and controls live in the repo next to the model, so the comparison can be re-run and not just quoted.",
 "Documentation: an execution record describing how every stage was run, a questions and answers document, and this notes file.",
 "An honest point worth including: the demo prints a warning if another process is on the GPU, because an early timing run reported 96 ms against CLIPSeg's 54.3 ms purely from first-call graph compilation.",
])

d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

# ── 1 ────────────────────────────────────────────────────────────────────
d.add_heading("1  Introduction", level=1)
note("About 500 to 700 words.")
d.add_heading("The problem", level=2)
bullets([
 "An editor is told what to change. It is never told where.",
 "Diffusion editors such as InstructPix2Pix, Qwen-Image-Edit and FLUX denoise the whole frame.",
 "So the edit comes out weak, and parts of the image that should not change drift in colour and texture.",
 "Our fix is to predict the region first, then gate the editor to it.",
])
d.add_heading("The hard case", level=2)
bullets([
 "“Put a hat on the dog” names an object that is not in the picture.",
 "There is no source region to point at.",
 "Any method that works by referring to visible content fails here.",
 "Referring segmentation returns what the phrase names. For that instruction it returns the dog. The right answer is the empty space above it.",
])
d.add_heading("Base paper and the gap", level=2)
bullets([
 "AdaptEdit, arXiv:2604.23763, April 2026. The only work that trains a predictor to ground the region from the instruction.",
 "Its scale is 20.43 B frozen diffusion transformer, 4.67 B adapters, 351 M condition encoder.",
 "That cannot be reproduced on a 26 GB laptop.",
 "It reports no mask IoU. The MaskPredictor is never scored on its own.",
 "Its limitations section names our gap: edits that change geometry and have no localized source region.",
 "Its masks come from MagicBrush human annotation and rendered text boxes, not from pixel differences.",
])
d.add_heading("Why this is a data science problem", level=2)
bullets([
 "No benchmark scores a predicted edit region. LocateEdit-Bench scores edits that already happened.",
 "That is why a 2026 paper built around a MaskPredictor can leave out a mask IoU and nobody objects.",
])
d.add_heading("Objectives", level=2)
note("Write these as a numbered list.")
bullets([
 "Test whether an off-the-shelf segmentation and retrieval pipeline is enough, and describe how it fails.",
 "Build supervision that does not need human region annotation.",
 "Train a predictor over frozen features within one laptop's compute.",
 "Score the predicted region directly, which is the measurement the base paper leaves out.",
 "Find out whether predicting the region improves the edit, and by how much.",
 "Establish the upper bound the representation imposes, and report how much of it we reach.",
])

# ── 2 ────────────────────────────────────────────────────────────────────
d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
d.add_heading("2  Literature Survey", level=1)
note("About 900 to 1200 words. 45 works, all 2024 to 2026. Cite with the links from the survey file.")
d.add_heading("Open with the classification", level=2)
bullets([
 "Sorting 45 works by what each one does with the edit region reads better than sorting by year.",
 "17 works consume a region they are given. NEP takes the mask as input. MaskFlow builds it into the objective. LazyDiffusion needs the user to draw one.",
 "11 works read a region out of a running generator. WhereEdit aggregates cross-attention. RegionE infers it once denoising has started. These are unsupervised and blind to empty space.",
 "2 works predict a region from the instruction: AdaptEdit, and MADiff, which is fashion only.",
 "15 works score or frame the problem. None of them scores a predicted region.",
 "The line to land: of 45 recent works, two produce the region from language.",
])
d.add_heading("Cluster notes", level=2)
note("One short paragraph each.")
bullets([
 "Instruction editors (9). Very good at editing, silent on localization. Step1X-Edit 19 B, Qwen 20 B, FLUX 12 to 32 B. NEP makes the asymmetry plain by taking a mask as input and never producing one.",
 "Edit localization (7). Splits into attention readers and region consumers. WhereEdit says outright that it does not handle adding into empty regions.",
 "Insertion and affordance (4). The field solves insertion once a location is given. MADD needs a position prompt. Add-it is training free, so placement just emerges.",
 "Region-aware efficiency (3). RegionE, SpotEdit and LazyDiffusion all use a region for speed and all assume it arrives from somewhere else. Useful to us: it shows a region is worth money in compute.",
 "Text-guided segmentation (5). Referring semantics are the wrong semantics. CLIPSeg gets 0.1849 overall and 0.054 on insertion. VespaSeg shows lightweight heads on frozen features can work.",
 "Benchmarks (4). LocateEdit-Bench scores 231 K already-edited images. That is forensics after the fact, not prediction.",
 "Planning and reasoning (4). X-Planner and RePlan break instructions down semantically, never to pixels, and need an MLLM in the loop.",
 "Structured scene representations (6). I2E and layer-wise memory represent things that exist. None of them makes empty space addressable.",
 "Surveys (3). The 2026 instruction-editing survey confirms localization is treated as a sub-component and never as an output in its own right.",
])
d.add_heading("The four gaps", level=2)
note("Give each one the published evidence and then your own measurement.")
bullets([
 "G1. Producing a region is unsolved, while editing given one is not. Our evidence: FastSAM with CLIP selection scores 0 of 7, failing on granularity, spatial relations and insertion.",
 "G2. The edit region is often not an object. The base paper says so. WhereEdit says so. Our measurement: CLIPSeg 0.1849 overall but 0.054 on insertion.",
 "G3. Supervision binds harder than architecture. Our measurement: MagicBrush masks cover 62.4% of the frame against a true 10.3%, so they are 9.1 times too large and precision collapses.",
 "G4. Nothing scores a predicted region. LocateEdit-Bench, GEdit-Bench and the affordance benchmark all score something downstream of the region.",
])
d.add_heading("What each line of work improves, and what it never reports", level=2)
bullets([
 "Instruction editors improve GEdit-Bench category scores. They never report mask IoU.",
 "AdaptEdit improves text_change to 8.53 and subject-add to 8.28. It never scores its own MaskPredictor.",
 "Attention readers improve edit quality with no training. They never report IoU against a true changed region.",
 "Region-aware efficiency improves wall-clock time. It never says where the region comes from.",
 "Referring segmenters improve RefCOCO-style IoU on named objects, which is the wrong target for editing.",
])
d.add_heading("Close with your position", level=2)
bullets([
 "What we can claim: predicting the region from the instruction is close to vacant, and deriving supervision from pixel differences looks unattempted in this setting.",
 "What we cannot claim: the mechanism. FiLM and BCE with Dice are AdaptEdit's choices too. Saying so is stronger than pretending otherwise.",
 "Honest limit: 39 of the 45 works are from 2025 and 2026, so the area moves fast. Rest the claim on the measured bound, which is a property of the representation.",
])

# ── 3 ────────────────────────────────────────────────────────────────────
d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
d.add_heading("3  Methodology", level=1)
note("About 900 to 1200 words. This section is mostly facts, so it is the easiest to write.")
d.add_heading("Frozen feature extraction", level=2)
bullets([
 "YOLOv8x-seg (FastSAM) backbone, frozen, used only as a feature source.",
 "It gives 32 mask prototypes at stride 4, so 160 by 160 maps for a 640 by 640 input.",
 "It also gives a 640-dimensional image context vector from the SPPF layer.",
 "The instruction goes through a frozen CLIP text encoder to 512 dimensions.",
 "The prototypes are spatial basis functions. Any mask is a linear combination of them through a sigmoid.",
 "Neither backbone trains. Only the head learns. That is what makes it run on one laptop.",
])
d.add_heading("Two heads", level=2)
bullets([
 "v1, global coefficients. One 32-d vector and a bias per image and instruction pair. Text modulates image context by FiLM, then an MLP emits the coefficients. 1,399,073 parameters.",
 "v2, spatial field. A 20 by 20 field of coefficients, bilinearly upsampled, plus a global branch. 4,693,633 parameters. Different locations can use different mixtures.",
 "Why v2: a single vector forces one mixture everywhere, which suits objects and not empty space. An earlier run showed v1 saturating at 1.4 M, so the limit was the form and not the size.",
])
d.add_heading("Supervision", level=2)
note("Explain why. This is the contribution.")
bullets([
 "Do not use MagicBrush's own masks as targets. They cover 62.4% of the frame against a true 10.3%.",
 "Derive the target from the real source and target image pair, by CIE76 colour difference in L*a*b*.",
 "Threshold 12.0, chosen from a noise robustness sweep. It holds at sigma 8.",
 "Morphological open then close with a 5 by 5 kernel. Drop connected components under 40 px.",
 "Reject samples whose changed area falls outside 0.4% to 45%, so nothing too small to learn and no whole-frame restyle.",
 "The idea in one line: where a hat landed is ground truth for where a hat should go.",
 "MagicBrush masks are bright equals preserve. Assuming the opposite gave IoU 0.001 before it was caught.",
])
d.add_heading("Architecture, layer by layer", level=2)
note("Exact shapes and parameter counts. Draw this as a block diagram in the report.")

para(d,"Stage 1, the full forward pass", size=10.5, bold=True, space=4)
bullets([
 "Input: an RGB image at 640 by 640, and one instruction string.",
 "Frozen branch A, image. YOLOv8x-seg. Two tensors are tapped by forward hooks.",
 "   model[-1].proto gives the prototypes, shape (B, 32, 160, 160).",
 "   model[9], the SPPF layer, is average pooled to a context vector, shape (B, 640).",
 "Frozen branch B, text. CLIP ViT-B/32 text encoder, L2 normalised, shape (B, 512).",
 "Trainable head. It maps (text 512, context 640) to coefficients over the 32 prototypes.",
 "Output: logits = sum over i of c_i times proto_i, plus a bias, shape (B, 1, 160, 160).",
 "A sigmoid and a threshold turn the logits into the mask.",
 "Frozen weight: 71.75 M for the backbone and prototypes, plus the CLIP text encoder. None of it trains.",
])

para(d,"v1, TextCoeffHead. One coefficient vector for the whole image.", size=10.5, bold=True, space=4)
bullets([
 "text_proj: LayerNorm(512), Linear(512 to 512), GELU. 263,680 parameters.",
 "img_proj: LayerNorm(640), Linear(640 to 512), GELU. 329,472 parameters.",
 "FiLM: two Linear(512 to 512) layers producing gamma and beta. 262,656 each.",
 "Fusion is v = gamma(t) * v + beta(t). Multiplicative, not concatenation, so the MLP cannot ignore the text.",
 "mlp: LayerNorm, Linear(512 to 512), GELU, Dropout(0.1), Linear(512 to 33). 280,609 parameters.",
 "The 33 outputs are 32 coefficients and 1 bias. The last layer is zero initialised, so training starts from a flat mask.",
 "Total 1,399,073 parameters.",
 "The limitation: one 32-vector applies to the entire frame, so the mask is the same mixture of prototypes everywhere.",
])

para(d,"v2, SpatialCoeffHead. A coefficient field.", size=10.5, bold=True, space=4)
bullets([
 "text_proj, img_proj and the FiLM pair are identical to v1, so 1,118,464 parameters are shared.",
 "Global branch, glob: LayerNorm, Linear(512 to 512), GELU, Dropout, Linear(512 to 33). 280,609 parameters. This is v1's MLP kept as an overall prior.",
 "Spatial branch, field: Linear(512 to 6400), GELU, reshaped to (B, 16, 20, 20). 3,283,200 parameters.",
 "field_conv: Conv2d(16 to 64, 3 by 3, padding 1), GELU, Conv2d(64 to 32, 1 by 1). 11,360 parameters.",
 "The 20 by 20 field is bilinearly upsampled to the prototype size of 160 by 160.",
 "The two branches add: c(x,y) = c_global + c_field(x,y). So different places in the frame can use different mixtures.",
 "Both output layers are zero initialised, so v2 starts as a flat mask, exactly like v1.",
 "Total 4,693,633 parameters. That is 3.35 times v1 and still 0.018% of AdaptEdit's 25.4 B.",
])

para(d,"The optional geometric basis", size=10.5, bold=True, space=4)
bullets([
 "12 fixed smooth functions appended to the 32 prototypes, so the basis becomes 44.",
 "They are 1, x, y, x squared, y squared, xy, x cubed, y cubed, sin(pi x), sin(pi y), cos(pi x), cos(pi y), on a coordinate grid normalised to minus 1 through 1.",
 "They carry no parameters at all. Only the 12 extra coefficients per branch are learned, which is 6,936 more weights.",
 "Why: a least squares fit showed adding them lifts the ceiling on insert from 0.197 to 0.324, and on remove from 0.464 to 0.581.",
 "Control: the same code takes basis=rand, which substitutes 12 Gaussian-blurred noise fields of matched size. Any gain has to beat that control or it is just extra capacity.",
 "Reported runs use basis=none, so the headline number does not depend on this.",
])

table(d,"Parameter budget.",
 ["Component","Parameters","Trains"],
 [["YOLOv8x-seg backbone and Proto","71.75 M","no"],
  ["CLIP ViT-B/32 text encoder","63.43 M","no"],
  ["v1 TextCoeffHead","1,399,073","yes"],
  ["v2 SpatialCoeffHead","4,693,633","yes"],
  ["v2 with geometric basis","4,700,569","yes"],
  ["AdaptEdit, for comparison","25.45 B","adapters only"]],
 widths=[3.2,1.5,1.0], align=["left","right","right"])

para(d,"Stage 2, the editing pipeline", size=10.5, bold=True, space=4)
bullets([
 "Stage 1 gives a probability map. Threshold it, then dilate it, to get the gate.",
 "The gated region goes to Stable Diffusion inpainting as the mask, with the instruction as the prompt.",
 "Four arms are compared on the same dev samples.",
 "   A. whole frame, InstructPix2Pix, no mask. This is the status quo.",
 "   B. inside our predicted mask.",
 "   C. inside the ground-truth mask. This is the ceiling for perfect localization.",
 "   D. inside MagicBrush's human mask. This is what the base paper's supervision would give.",
 "Metrics are all measured outside the true edit region: collateral fraction, PSNR_out, and CLIP_in inside the region.",
 "The operating point came from a 20-cell sweep over threshold and dilation, not from one guess.",
])

para(d,"Baseline architectures, for the comparison table", size=10.5, bold=True, space=4)
bullets([
 "CLIPSeg. A frozen CLIP backbone with a small transformer decoder, running at 352 by 352. The nearest comparison, because it is also a light head on frozen features.",
 "FastSAM with CLIP selection. Segment everything, then pick the region whose crop best matches the instruction. Selection only, so it can never return empty space.",
 "AdaptEdit. 20.43 B frozen diffusion transformer, 4.67 B adapters, 351 M condition encoder. Its MaskPredictor is the part comparable to ours, and it is never scored.",
 "Human masks. MagicBrush's own annotations, used as a baseline rather than as targets.",
])

d.add_heading("Loss and optimisation", level=2)
bullets([
 "BCE plus 2 times Dice. Gradient norm clipped at 1.0.",
 "AdamW, learning rate 1e-3 cosine to zero, weight decay 0.01, batch 32, 60 epochs.",
 "12 seeds (1368, and 1 to 11), seeding torch, numpy, random and the Metal generator.",
])
d.add_heading("Dataset", level=2)
bullets([
 "MagicBrush. All 51 training shards and all 4 dev shards, 25 GB of Parquet.",
 "8,807 rows, of which 8,306 are usable after filtering. 501 were rejected as degenerate.",
 "Kinds: modify 6,045, insert 1,828, remove 433.",
 "Held-out dev: 528 turns, 503 usable.",
])
d.add_heading("Split integrity", level=2)
bullets([
 "MagicBrush is multi-turn, so one photograph yields several edits in a row.",
 "If the split went by turn instead of by image, the same photo would sit on both sides.",
 "So every source and target image was hashed with SHA-1 on both sides, and the two sets were intersected.",
 "Result: 794 distinct dev images, 13,317 distinct train images, 0 overlap.",
 "Say that this ran before the results were trusted, not afterwards to defend them.",
])
d.add_heading("Protocol", level=2)
bullets([
 "The checkpoint epoch was picked on a held-out 15% slice of train, never on dev.",
 "Dev was evaluated once per seed, at full resolution, against the undilated target.",
 "Welch t-test on seed means. One-sample t-tests against fixed baselines.",
 "Frozen features were precomputed once into memory-mapped arrays of 13.8 GB, so each epoch only touches the head.",
])
d.add_heading("Why each design choice", level=2)
note("Examiners ask this.")
bullets([
 "Predict coefficients instead of selecting a mask. Selection can only return an object that is already there, so insertion is impossible in principle. Evidence: the 0 of 7 pilot.",
 "Freeze both backbones. The backbone depends on the image alone, so you pay for it once and run the head per instruction. Evidence: 5.30 ms against CLIPSeg's 54.3 ms.",
 "Derive targets by colour difference. The dataset's masks are 9.1 times too large, so training on them teaches the wrong region.",
 "Spatial field over a global vector. One vector forces a single mixture everywhere. Evidence: insert 0.1186 against 0.0972.",
])

# ── checklist ────────────────────────────────────────────────────────────
d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
d.add_heading("Checklist", level=1)
bullets([
 "Write each section in your own words with these notes beside you. Do not paste sentences from any document.",
 "Numbers are facts and cannot be plagiarised. The sentences around them have to be yours.",
 "Quote the base paper's limitation directly, in quotation marks, with the citation. A quoted and attributed line is not plagiarism.",
 "Cite all 45 works with links. A reference list raises similarity a little, which is expected and usually excluded.",
 "Upload once, read the report, fix what it flags, then upload the second time. Two attempts only.",
 "Due 15 Sep. The final file goes to Teams, not Turnitin.",
])

d.save("DA2_writing_notes.docx")
print("saved")
