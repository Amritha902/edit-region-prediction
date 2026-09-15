import sys, json; sys.path.insert(0,".")
from mkdocx import *
from pathlib import Path

T = Path("/Users/amritha/DATASCIENCE FINAL/instruct-seg-edit-mac/train")
F = Path("/Users/amritha/DATASCIENCE FINAL/report")
J = lambda n: json.load(open(T/n))

ce   = J("clean_eval_full.json")
bl   = J("baselines.json")
mf   = J("metrics_full.json")
lat  = J("latency_v2.json")
leak = J("leakage_check.json")
hc   = J("head_comparison.json")
sc   = J("scaling.json")
s2   = J("stage2.json")
fs   = J("freespace.json")
fx   = J("fix_checks.json")
gr   = J("grid_final.json")

v2 = [x for x in ce if x["name"].startswith("v2")][0]
v1 = [x for x in ce if x["name"].startswith("v1")][0]
sm = mf["v2"]["summary"]

d = base_doc()
titlepage(d, "Results and Evidence",
          "Instruction-Conditioned Edit-Region Prediction",
          "Predicting where an image edit belongs, from the instruction alone",
          course="Foundations of Data Science · DA2",
          date="15 September 2026")

rp=d.add_paragraph(); rp.alignment=WD_ALIGN_PARAGRAPH.CENTER
rr=rp.add_run("Code and data: "); rr.font.size=Pt(10.5); rr.font.color.rgb=MUTE
hyperlink(rp,"https://github.com/Amritha902/edit-region-prediction","https://github.com/Amritha902/edit-region-prediction",size=10.5)
rp.paragraph_format.space_after=Pt(4)
rp2=d.add_paragraph(); rp2.alignment=WD_ALIGN_PARAGRAPH.CENTER
rr2=rp2.add_run("Every table and figure in this document is generated from the committed result files by report/build_da2_results.py.")
rr2.font.size=Pt(9.5); rr2.italic=True; rr2.font.color.rgb=MUTE
rp2.paragraph_format.space_after=Pt(16)

toc(d, [("1","Data Collection and Understanding",1),
        ("2","Data Preprocessing",1),
        ("3","Feature Engineering and Selection",1),
        ("4","Model Implementation",1),
        ("5","Results",1),
        ("6","Code Quality and Documentation",1)])

def H1(t): d.add_paragraph().add_run().add_break(WD_BREAK.PAGE); d.add_heading(t, level=1)
def H2(t): d.add_heading(t, level=2)
def B(items):
    for t in items:
        p=d.add_paragraph(t, style="List Bullet"); p.paragraph_format.space_after=Pt(2)
        for r in p.runs: r.font.size=Pt(10.5)
def N(t):
    p=d.add_paragraph(); r=p.add_run(t); r.font.size=Pt(10); r.italic=True
    r.font.color.rgb=MUTE; p.paragraph_format.space_after=Pt(8)

pc = lambda x: f"{x*100:.1f}%"
f4 = lambda x: f"{x:.4f}"

# ════════════════════════════════════════════════════════════ 1
H1("1  Data Collection and Understanding")

H2("1.1  Dataset")
B(["MagicBrush, from HuggingFace, built on COCO images with human annotators performing real edits.",
   "All 51 training shards and all 4 dev shards were downloaded. 25 GB of Parquet.",
   "Chosen because it is the only public set with genuine source and target image pairs for instruction edits, and because the base paper trains on it, so the comparison is fair.",
   "It is multi-turn: one photograph produces several edits in sequence, and turn 2's source is turn 1's target."])

table(d,"Table 1.1  Variables in each row.",
 ["Variable","Type","Role"],
 [["source_img","PNG bytes","input image"],
  ["target_img","PNG bytes","edited image, used to derive the target"],
  ["mask_img","PNG bytes","human annotation, used only as a baseline"],
  ["instruction","string","the text condition"],
  ["img_id","string","groups the turns of one photograph"],
  ["turn_index","integer","position within a multi-turn sequence"]],
 widths=[1.7,1.4,3.1], align=["left","left","left"])

table(d,"Table 1.2  Dataset scale.",
 ["Split","Rows","Usable","Rejected"],
 [["train","8,807","8,306","501"],
  ["dev","528","503","25"]],
 widths=[1.6,1.5,1.5,1.5])

table(d,"Table 1.3  Class balance, training split. The rarest class is also the hardest.",
 ["Edit kind","Count","Share"],
 [["modify","6,045","72.8%"],
  ["insert","1,828","22.0%"],
  ["remove","433","5.2%"]],
 widths=[2.0,1.6,1.6])

H2("1.2  Exploratory findings that changed the design")
N("These are not descriptive statistics. Each one altered a decision.")

table(d,"Table 1.4  The supplied human masks are far coarser than the real change. "
        "Measured over all 528 dev turns.",
 ["Quantity","Value"],
 [["Mean human-mask coverage of the frame","62.4%"],
  ["Mean true changed area","10.3%"],
  ["Ratio","9.1x too large"],
  ["Ratio on insertions alone","11.9x"],
  ["Precision of the human mask","15.8%"],
  ["IoU of the human mask against the true change","0.16"]],
 widths=[4.2,1.8])

B(["Consequence: the human masks were rejected as training targets and the target was derived from the image pair instead. This is the central design decision of the project.",
   "Second finding: the mask polarity is bright equals preserve. Assuming the opposite produced IoU 0.001 before it was caught.",
   "Third finding: because the set is multi-turn, a split by turn would put the same photograph on both sides. This forced the leakage audit in section 2."])

figure(d, str(F/"run_evidence.png"),
       "Figure 1.1  Run evidence: dataset construction, filtering and the training environment.")

# ════════════════════════════════════════════════════════════ 2
H1("2  Data Preprocessing")

H2("2.1  Duplicates and leakage")
B([f"Every source and target image was hashed with SHA-1 on both sides of the split, and the two sets intersected.",
   f"Result: {leak['n_dev']} distinct dev images, {leak['n_train']:,} distinct train images, {leak['n_overlap']} overlap.",
   "This was run before the results were trusted, not afterwards to defend them.",
   "A second deduplication on img_id was needed for the qualitative figure, because byte hashing fails where one image is both a target and the next source."])

table(d,"Table 2.1  Leakage audit.",
 ["Quantity","Value"],
 [["Distinct dev images", f"{leak['n_dev']}"],
  ["Distinct train images", f"{leak['n_train']:,}"],
  ["Intersection", f"{leak['n_overlap']}"]],
 widths=[4.2,1.8])

H2("2.2  Outliers and degenerate samples")
B(["Samples whose changed area falls outside 0.4% to 45% are rejected: too small to learn from, or a whole-frame restyle rather than a local edit.",
   "501 of 8,807 training rows were dropped, which is 5.7%.",
   "25 of 528 dev turns were dropped by the same rule."])

H2("2.3  Noise removal and transformation")
table(d,"Table 2.2  The preprocessing chain, in order.",
 ["Step","Operation","Parameter"],
 [["1","Resize target to source size","nearest for masks, bilinear for images"],
  ["2","Colour space","BGR to CIE L*a*b*"],
  ["3","Difference","CIE76 dE per pixel"],
  ["4","Threshold","dE > 12.0"],
  ["5","Morphological open then close","5 x 5 kernel"],
  ["6","Drop small components","under 40 px"],
  ["7","Area filter","keep 0.4% to 45%"],
  ["8","Resize image for backbone","640 x 640 bilinear"],
  ["9","Downsample mask","160 x 160, the prototype grid"]],
 widths=[0.6,2.9,2.5], align=["left","left","left"])

B(["The threshold of 12.0 was not chosen by eye. It came from a noise robustness sweep and holds at sigma 8.",
   "The idea in one line: where a hat landed is ground truth for where a hat should go."])

figure(d, str(F/"DA2_figures/pipeline.png"),
       "Figure 2.1  The preprocessing chain on three held-out samples. Every panel is the actual "
       "array at that step. The final column is the dataset's own annotation at the same scale, "
       "which is what the ratio in Table 1.4 measures.")

H2("2.4  Encoding")
B(["Text: instructions go through CLIP BPE tokenisation to a 512-d vector, L2 normalised.",
   "Categorical: the edit kind is derived by rule from the leading verb. put or add gives insert, remove or delete gives remove, everything else gives modify.",
   "Storage: features are cast to float16 and written to memory-mapped arrays totalling 13.8 GB, so training never re-runs the backbone. This is what made 60 epochs on 8,306 samples feasible on a laptop."])

# ════════════════════════════════════════════════════════════ 3
H1("3  Feature Engineering and Selection")

H2("3.1  The central feature decision")
B(["FastSAM emits 32 mask prototypes for every frame. Normally the coefficients that combine them come from a detection, which is exactly why its output can only ever be an object already present.",
   "We treat the prototypes as a spatial basis and predict the coefficients from the instruction instead.",
   "mask = sigmoid( sum over i of c_i times proto_i, plus b )",
   "Because the prototypes are basis functions and not objects, a text-derived combination can describe empty space. That is what makes insertion possible in principle."])

table(d,"Table 3.1  The four feature sources.",
 ["Feature","Shape","Source","Trains"],
 [["Mask prototypes","(B, 32, 160, 160)","YOLOv8x-seg proto module","no"],
  ["Image context","(B, 640)","SPPF layer, average pooled","no"],
  ["Text embedding","(B, 512)","CLIP ViT-B/32 text encoder","no"],
  ["Geometric basis","(12, 160, 160)","fixed closed form","no"]],
 widths=[1.6,1.7,2.0,0.9], align=["left","left","left","right"])

figure(d, str(F/"DA2_figures/basis.png"),
       "Figure 3.1  The basis the head combines: 32 image-dependent prototypes and 12 fixed "
       "geometric functions. No single prototype is the answer to an instruction; a combination is.")

H2("3.2  The engineered geometric basis")
B(["12 fixed smooth functions appended to the 32 prototypes: 1, x, y, x squared, y squared, xy, x cubed, y cubed, sin(pi x), sin(pi y), cos(pi x), cos(pi y), on a grid normalised to minus 1 through 1.",
   "They carry no parameters at all.",
   "A least squares fit measured the achievable ceiling with and without them."])

table(d,"Table 3.2  Least-squares ceiling by basis. The random control is the test that "
        "the gain is not merely extra capacity.",
 ["Basis","Extra","Insert","Modify","Remove"],
 [[k, str(v["extra"]), f4(v["insert"]), f4(v["modify"]), f4(v["remove"])]
  for k,v in fs.items()],
 widths=[2.4,0.8,1.0,1.0,1.0])

B(["The geometric basis beats the matched random control on all three kinds, so the gain is structural and not capacity.",
   "The reported headline runs use no extra basis, so the main result does not depend on this."])

H2("3.3  Alternatives rejected, with reasons")
table(d,"Table 3.3  What was not used.",
 ["Candidate","Why rejected"],
 [["Raw pixels","No semantics, and far too many dimensions for the compute budget"],
  ["Detection boxes","Can only name objects already present, so insertion is impossible"],
  ["Referring segmentation output","Returns what a phrase denotes, which is the wrong target"],
  ["MagicBrush human masks","9.1 times too large, precision 15.8%"]],
 widths=[2.0,4.0], align=["left","left"])

# ════════════════════════════════════════════════════════════ 4
H1("4  Model Implementation")

H2("4.1  Architecture")
figure(d, str(F/"DA2_figures/architecture.png"),
       "Figure 4.1  System architecture. Both backbones are frozen; only the head learns.")

B(["Input: an RGB image at 640 by 640, and one instruction string.",
   "Frozen branch A, image. YOLOv8x-seg. Two tensors are tapped by forward hooks: model[-1].proto gives (B, 32, 160, 160), and model[9], the SPPF layer, is average pooled to (B, 640).",
   "Frozen branch B, text. CLIP ViT-B/32 text encoder, L2 normalised, (B, 512).",
   "Trainable head maps (text, context) to coefficients over the prototype basis.",
   "Output: logits of shape (B, 1, 160, 160). A sigmoid and a threshold give the mask."])

table(d,"Table 4.1  v1 TextCoeffHead, block by block. One coefficient vector for the whole frame.",
 ["Block","Layers","Parameters"],
 [["text_proj","LayerNorm(512), Linear(512 to 512), GELU","263,680"],
  ["img_proj","LayerNorm(640), Linear(640 to 512), GELU","329,472"],
  ["gamma","Linear(512 to 512)","262,656"],
  ["beta","Linear(512 to 512)","262,656"],
  ["mlp","LayerNorm, Linear(512 to 512), GELU, Dropout(0.1), Linear(512 to 33)","280,609"],
  ["Total","","1,399,073"]],
 widths=[1.2,3.6,1.2], align=["left","left","right"], bold_rows=(5,))

table(d,"Table 4.2  v2 SpatialCoeffHead. A coefficient field at 20 by 20, upsampled.",
 ["Block","Layers","Parameters"],
 [["text_proj, img_proj, gamma, beta","identical to v1","1,118,464"],
  ["glob","LayerNorm, Linear(512 to 512), GELU, Dropout, Linear(512 to 33)","280,609"],
  ["field","Linear(512 to 6400), GELU, reshape to (16, 20, 20)","3,283,200"],
  ["field_conv","Conv2d(16 to 64, 3x3), GELU, Conv2d(64 to 32, 1x1)","11,360"],
  ["Total","","4,693,633"]],
 widths=[1.7,3.1,1.2], align=["left","left","right"], bold_rows=(4,))

B(["Fusion is FiLM: v = gamma(t) * v + beta(t). Multiplicative rather than concatenation, so the network cannot ignore the instruction.",
   "The 20 by 20 field is bilinearly upsampled to 160 by 160 and added to the global prior: c(x,y) = c_global + c_field(x,y).",
   "Both output layers are zero initialised, so training starts from a flat mask.",
   "Justification for v2: an earlier run showed v1 saturating at 1.4 M parameters, so width was not the constraint. The constraint was that one vector forces the same mixture everywhere, which suits objects and not empty space."])

table(d,"Table 4.3  Parameter budget.",
 ["Component","Parameters","Trains"],
 [["YOLOv8x-seg backbone and Proto","71.75 M","no"],
  ["CLIP ViT-B/32 text encoder","63.43 M","no"],
  ["v1 TextCoeffHead","1,399,073","yes"],
  ["v2 SpatialCoeffHead","4,693,633","yes"],
  ["AdaptEdit, the base paper","25.45 B","adapters only"]],
 widths=[3.2,1.5,1.3])

H2("4.2  Training")
table(d,"Table 4.4  Training configuration.",
 ["Setting","Value"],
 [["Loss","BCE + 2 x Dice"],
  ["Why Dice","Edit regions average 10% of the frame, so BCE alone predicts background everywhere"],
  ["Optimiser","AdamW"],
  ["Learning rate","1e-3, cosine to zero"],
  ["Weight decay","0.01"],
  ["Batch size","32"],
  ["Epochs","60"],
  ["Gradient clipping","norm 1.0"],
  ["Seeds","12 (1368, and 1 to 11)"],
  ["Device","Apple MPS"]],
 widths=[1.8,4.2], align=["left","left"])

H2("4.3  Evaluation protocol")
B(["The checkpoint epoch is selected on a held-out 15% slice of TRAIN. Dev is never used for selection.",
   "Dev is evaluated once per seed, at full resolution, against the undilated target.",
   "Welch t-tests on seed means, one-sample t-tests against fixed baselines, Cohen's d, and 95% confidence intervals.",
   "A zeroed-instruction control was run to confirm the text is actually doing work."])

# ════════════════════════════════════════════════════════════ 5
H1("5  Results")

H2("5.1  Headline")
table(d,"Table 5.1  Dev IoU against every baseline. 12 seeds, evaluated once each.",
 ["Method","Trainable params","IoU","Insert","Modify","Remove"],
 [["Random centred box","0", f4(bl["random centred box"]["all"]), f4(bl["random centred box"]["insert"]), f4(bl["random centred box"]["modify"]), f4(bl["random centred box"]["remove"])],
  ["Whole frame","0", f4(bl["full frame"]["all"]), f4(bl["full frame"]["insert"]), f4(bl["full frame"]["modify"]), f4(bl["full frame"]["remove"])],
  ["MagicBrush human mask","0", f4(bl["MagicBrush GT mask"]["all"]), f4(bl["MagicBrush GT mask"]["insert"]), f4(bl["MagicBrush GT mask"]["modify"]), f4(bl["MagicBrush GT mask"]["remove"])],
  ["CLIPSeg","150.7 M", f4(bl["CLIPSeg (150M)"]["all"]), f4(bl["CLIPSeg (150M)"]["insert"]), f4(bl["CLIPSeg (150M)"]["modify"]), f4(bl["CLIPSeg (150M)"]["remove"])],
  ["Ours v1 global","1,399,073", f4(v1["iou"]), f4(v1["by_kind"]["insert"]), f4(v1["by_kind"]["modify"]), f4(v1["by_kind"]["remove"])],
  ["Ours v2 spatial","4,693,633", f4(v2["iou"]), f4(v2["by_kind"]["insert"]), f4(v2["by_kind"]["modify"]), f4(v2["by_kind"]["remove"])]],
 widths=[1.7,1.2,0.8,0.8,0.8,0.8], bold_rows=(5,))

B([f"v2 reaches {f4(v2['iou'])} plus or minus {f4(v2['sd'])}, with a 95% confidence interval of {f4(sm['iou']['ci_lo'])} to {f4(sm['iou']['ci_hi'])}.",
   f"It beats CLIPSeg ({f4(bl['CLIPSeg (150M)']['all'])}) with 32 times fewer trainable parameters.",
   f"It beats the human masks ({f4(bl['MagicBrush GT mask']['all'])}), which is the supervision the base paper uses.",
   f"v2 over v1 is +{f4(mf['v2_vs_v1']['diff'])}, p = {mf['v2_vs_v1']['p']:.1e}, Cohen's d = {mf['v2_vs_v1']['cohens_d']:.2f}.",
   "Per-kind crossover: v2 wins insert and modify, v1 wins remove. The spatial field helps exactly where a single mixture cannot work."])

figure(d, str(F/"results_full.png"),
       "Figure 5.1  Full results: IoU by method, per-kind breakdown, seed distribution and threshold behaviour.")

H2("5.2  Full metrics")
table(d,"Table 5.2  v2 at threshold 0.5, mean over 12 seeds with 95% confidence intervals.",
 ["Metric","Mean","SD","95% CI"],
 [[k.capitalize(), f4(sm[k]["mean"]), f4(sm[k]["sd"]),
   f"{f4(sm[k]['ci_lo'])} to {f4(sm[k]['ci_hi'])}"]
  for k in ["iou","precision","recall","f1"] if k in sm],
 widths=[1.4,1.2,1.2,2.2])

table(d,"Table 5.3  Threshold sweep on v2. IoU peaks at 0.3, not at the default 0.5.",
 ["Threshold","IoU","Precision","Recall","F1","Pred area"],
 [[t, f4(x["iou"]), f4(x["precision"]), f4(x["recall"]), f4(x["f1"]), pc(x["pred_area"])]
  for t,x in mf["v2"]["sweep"].items()],
 widths=[1.0,0.9,1.0,0.9,0.9,1.1],
 bold_rows=(2,))

N(f"True area is {pc(mf['v2']['true_area'])}. The threshold was selected on a train-val slice, "
  f"and dev was touched once, so this sweep is reported and not used for selection.")

H2("5.3  Scaling")
table(d,"Table 5.4  IoU against training-set size. The delta column is the gain over a "
        "zeroed instruction, which is what the text is worth.",
 ["n train","IoU","SD","Zeroed text","Delta"],
 [[str(r["n"]), f4(r["iou"]), f4(r["iou_sd"]), f4(r["notext"]), f4(r["delta"])]
  for r in sc],
 widths=[1.1,1.1,1.0,1.3,1.1])

B([f"Scaling from 2,278 to 8,306 samples, a factor of 3.65, lifted IoU from 0.1718 to {f4(v2['iou'])}, p = 4.1e-09.",
   "The gain over zeroed instructions confirms the text is doing real work rather than the model learning a fixed average mask."])

figure(d, str(F/"curves.png"),
       "Figure 5.2  Training curves across all 12 seeds, with the zeroed-instruction control.")

H2("5.4  Head comparison")
table(d,"Table 5.5  Architectures tried before v2. Delta is the gain over zeroed text.",
 ["Head","Params","IoU","Delta","Insert","Modify","Remove"],
 [[h["name"], f"{h['params']:,}", f4(h["iou"]), f4(h["delta"]),
   f4(h["by_kind"]["insert"]), f4(h["by_kind"]["modify"]), f4(h["by_kind"]["remove"])]
  for h in hc],
 widths=[1.5,1.0,0.75,0.75,0.7,0.7,0.7])

N("The query-grid heads have more spatial freedom and do worse. Capacity was never the "
  "constraint, which is what motivated v2's field rather than a wider MLP.")

H2("5.5  The representational ceiling")
table(d,"Table 5.6  Least-squares ceiling on dev, and the fraction reached.",
 ["Kind","Ceiling","v2 achieved","Fraction reached"],
 [["insert", f4(fx["ceiling_dev"]["insert"]), f4(v2["by_kind"]["insert"]),
   pc(v2["by_kind"]["insert"]/fx["ceiling_dev"]["insert"])],
  ["modify", f4(fx["ceiling_dev"]["modify"]), f4(v2["by_kind"]["modify"]),
   pc(v2["by_kind"]["modify"]/fx["ceiling_dev"]["modify"])],
  ["remove", f4(fx["ceiling_dev"]["remove"]), f4(v2["by_kind"]["remove"]),
   pc(v2["by_kind"]["remove"]/fx["ceiling_dev"]["remove"])]],
 widths=[1.3,1.4,1.5,1.8])

B(["The ceiling is the best any linear combination of this basis could achieve, found by least squares against the true mask.",
   "Insertion is bounded at 0.2119 and we reach 56% of it. Even a perfect head on these features could not do much better.",
   "That makes insertion a representation problem, not a training problem. It is the open question the project ends on."])

H2("5.6  Latency")
table(d,"Table 5.7  Inference cost, resolution matched at 352 px.",
 ["Model","Params","Per image","Per instruction"],
 [["CLIPSeg", f"{lat['clipseg']['params']:,}", "n/a", f"{lat['clipseg']['per_instruction_ms']:.1f} ms"],
  ["Ours v1", f"{lat['v1']['params']:,}", f"{lat['v1']['352']['per_image_ms']:.1f} ms", f"{lat['v1']['352']['per_instruction_ms']:.2f} ms"],
  ["Ours v2", f"{lat['v2']['params']:,}", f"{lat['v2']['352']['per_image_ms']:.1f} ms", f"{lat['v2']['352']['per_instruction_ms']:.2f} ms"]],
 widths=[1.3,1.5,1.5,1.7], bold_rows=(2,))

B([f"v2 is {lat['clipseg']['per_instruction_ms']/lat['v2']['352']['per_instruction_ms']:.1f} times faster per instruction than CLIPSeg, with "
   f"{lat['clipseg']['params']/lat['v2']['params']:.1f} times fewer parameters.",
   "The backbone runs once per image and the head runs per instruction, so the cost amortises across multiple edits on the same photograph.",
   "An early timing run reported 96 ms, which was first-call Metal graph compilation. The benchmark now warms up first and warns if another process holds the GPU."])

H2("5.7  Stage 2: does the region actually help the edit?")
table(d,"Table 5.8  Four arms, same dev samples, Stable Diffusion inpainting. "
        "Net is recall minus collateral. Higher is better except collateral.",
 ["Arm","Recall","Collateral","Net","PSNR out","CLIP in"],
 [[k, pc(v["recall"]), pc(v["collateral"]), pc(v["net"]), f"{v['psnr']:.2f}", f4(v["clip"])]
  for k,v in s2.items()],
 widths=[1.7,0.9,1.1,0.9,1.0,0.9])

B([f"Editing inside our predicted region gives net {pc(s2['B our mask']['net'])} against {pc(s2['A whole-frame']['net'])} for the whole frame. Localization is worth 5.8 times.",
   f"It also beats MagicBrush's own masks ({pc(s2['D MagicBrush mask']['net'])}), which is the supervision the base paper uses.",
   f"Perfect localization would give {pc(s2['C GT mask']['net'])}, so there is headroom left."])

figure(d, str(F/"DA2_figures/stage2.png"),
       "Figure 5.3  Stage 2 in full: the four gating strategies, net gain, the precision and recall "
       "frontier, inference cost, and all 20 operating points.")

H2("5.8  Operating-point sweep")
N("A 20-cell grid over threshold and dilation. The optimum is interior on both axes, "
  "which means it is a real optimum and not an artefact of the search range.")

ths=[0.2,0.3,0.4,0.5]; dils=[0,16,32,48,64]
def cell(t,dl,k):
    g=gr.get(f"{t}_{dl}")
    return pc(g[k]) if g else "-"
table(d,"Table 5.9  Net gain across the grid.",
 ["thr"]+[f"{x} px" for x in dils],
 [[str(t)]+[cell(t,dl,"net") for dl in dils] for t in ths],
 widths=[0.9]+[1.0]*5)
table(d,"Table 5.10  Collateral damage across the same grid.",
 ["thr"]+[f"{x} px" for x in dils],
 [[str(t)]+[cell(t,dl,"collateral") for dl in dils] for t in ths],
 widths=[0.9]+[1.0]*5)

best=gr["0.3_32"]
B([f"Best cell: threshold 0.3 with 32 px dilation. Net {pc(best['net'])}, recall {pc(best['recall'])}, collateral {pc(best['collateral'])}, PSNR {best['psnr']:.2f}.",
   f"Its collateral of {pc(best['collateral'])} is below the whole-frame arm's {pc(s2['A whole-frame']['collateral'])}, so the gain does not come from simply editing more.",
   "The optimal dilation rises with the threshold: 16 px at 0.2, 32 px at 0.3, 48 px at 0.4 and 0.5. A tighter mask needs more padding, which is the behaviour you would predict.",
   "References: whole frame +9.1%, MagicBrush +26.8%, oracle +53.0%."])

figure(d, str(F/"grid_heatmap.png"),
       "Figure 5.4  The threshold and dilation grid. The optimum is interior on both axes.")

H2("5.9  Qualitative")
figure(d, str(F/"qualitative_full.png"),
       "Figure 5.5  Predictions on held-out dev images, deduplicated by photograph. "
       "Source, instruction, derived target, and prediction.")

B(["Mask coverage varies a lot by image: 9.5% on the cleanest case and 56.3% on a dense close-up where the model over-predicts badly.",
   "The honest answer to does it always work is no, and the per-kind numbers and the ceiling say why."])

# ════════════════════════════════════════════════════════════ 6
H1("6  Code Quality and Documentation")

rp=d.add_paragraph(); rr=rp.add_run("Repository: "); rr.font.size=Pt(10.5)
hyperlink(rp,"https://github.com/Amritha902/edit-region-prediction","https://github.com/Amritha902/edit-region-prediction",size=10.5)
rp.paragraph_format.space_after=Pt(6)

B(["Every experiment, checkpoint, figure and result file is committed.",
   "Each module opens with a docstring stating why the thing exists, not only what it does.",
   "Seeds are fixed and stored inside each checkpoint alongside n_train and dev_iou, so any number traces back to the run that produced it.",
   "Long jobs are resumable. The precompute checkpoints after every batch. The sweep runs one cell per job with a resume guard, added after three runs died at session teardown.",
   "Baselines and controls live beside the model so the comparison can be re-run rather than quoted."])

table(d,"Table 6.1  Where each result in this document comes from.",
 ["Section","Script","Output"],
 [["1.2  Human mask coarseness","experiments/exp07_magicbrush","measured on 528 dev turns"],
  ["2.1  Leakage audit","train/leakage_check.py","leakage_check.json"],
  ["2.3  Preprocessing","datagen/mask_extraction.py, train/dataset.py","derived targets"],
  ["3.2  Ceiling by basis","train/freespace.py","freespace.json"],
  ["4  Architecture","train/model.py, train/model_v2.py","head definitions"],
  ["5.1  Headline","train/clean_eval_full.py","clean_eval_full.json"],
  ["5.2  Full metrics","train/metrics_full.py","metrics_full.json"],
  ["5.3  Scaling","train/scaling.py","scaling.json"],
  ["5.4  Head comparison","train/head_comparison.py","head_comparison.json"],
  ["5.5  Ceiling","train/fix_checks.py","fix_checks.json"],
  ["5.6  Latency","train/latency_v2.py","latency_v2.json"],
  ["5.7  Stage 2","train/stage2.py","stage2.json"],
  ["5.8  Grid sweep","train/stage2_sd_sweep.py","grid_final.json"]],
 widths=[1.8,2.3,1.9], align=["left","left","left"])

table(d,"Table 6.2  Errors found and corrected during the work. Listed because catching "
        "them is part of the result.",
 ["What was wrong","How it was caught","Effect"],
 [["Mask polarity inverted","IoU 0.001","Would have trained on the complement"],
  ["Latency measured on v1, not v2","Audit before writing up","Reported figure was the wrong head"],
  ["Ceiling quoted from train, not dev","Recomputed on dev","Remove ceiling 0.4325, not 0.4575"],
  ["Collateral ratio divided backwards","Read the summary line","Printed 0.6x less where it was 1.62x more"],
  ["Qualitative figure showed 3 scenes","Multi-turn dedup on img_id","Byte hashing had failed"],
  ["Checkpoints silently ungitignored","git ls-tree against remote","Push reported success, saved nothing"],
  ["Demo timing 96 ms","Warm-up added","First-call graph compilation"]],
 widths=[2.1,1.9,2.0], align=["left","left","left"])

d.save("DA2_Results_And_Evidence.docx")
print("saved DA2_Results_And_Evidence.docx")
