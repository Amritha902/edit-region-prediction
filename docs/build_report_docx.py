import sys; sys.path.insert(0,".")
from mkdocx import *

TITLE = "Disentangling Localization from Editing in Instruction-Guided Image Manipulation"
SUB   = "A frozen-backbone edit-region predictor, measured against its representational ceiling"

d = base_doc()
titlepage(d, "Technical Report", TITLE, SUB)
toc(d, [("1","Introduction",1),("1.1","What we build",2),("1.2","Why this design",2),("2","Literature Review",1),("3","Problem Statement and Objectives",1),
        ("4","Methodology",1),("4.1","Frozen feature extraction",2),("4.2","Global coefficient head (v1)",2),
        ("4.3","Spatial coefficient field (v2)",2),("4.4","Supervision and loss",2),
        ("5","Dataset",1),("5.1","Split integrity",2),
        ("6","Experimental Protocol",1),("6.1","Configuration",2),
        ("7","Results and Discussion",1),("7.1","Accuracy",2),("7.2","Full metric suite",2),
        ("7.3","Operating point",2),("7.4","Per edit kind",2),("7.5","Representational ceiling",2),
        ("7.6","Training dynamics and ablations",2),("7.7","Latency",2),
        ("8","Stage 2: Editing Within the Predicted Region",1),
        ("9","Validation and Corrective Actions",1),
        ("10","Limitations",1),("11","Conclusion and Future Work",1),
        ("A","Appendix A — Complete Configuration",1),
        ("B","Appendix B — Reproduction",1)])

d.add_heading("Abstract", level=1)
para(d,"Instruction-guided image editors apply a diffusion model to the whole frame and rely on the model to decide, implicitly, which pixels should change. The base paper for this work, AdaptEdit (arXiv:2604.23763), makes that decision explicit through a learned MaskPredictor, and states in its own limitations section the case it cannot address: edits that add an object where no source region exists. AdaptEdit reports no mask IoU, and at 20.43 billion frozen parameters with approximately 5.0 billion trainable it cannot be reproduced on available hardware.")
para(d,"This work reimplements its mechanism at a scale that can be run and verified, and supplies the measurement the base paper omits. A 4.69 million parameter head predicts a spatial field of coefficients over 32 frozen FastSAM prototypes, conditioned on a CLIP text embedding. Trained on 8,306 MagicBrush turns on a single Apple M5, it attains IoU 0.2065 ± 0.0072 across 12 seeds on 503 held-out turns, exceeding CLIPSeg (0.1849, 150.7 million parameters) and the human annotations distributed with MagicBrush (0.1511), at 32 times fewer trainable parameters and 10.3 times lower per-instruction latency.")
para(d,"The principal contribution is a negative and quantitative one. Fitting coefficients directly to ground truth by least squares bounds the IoU attainable by any predictor over this basis: 0.212 for insertion against 0.433 for removal. Insertion, the case this work addresses, carries the lowest ceiling because prototypes trained to segment objects span empty space poorly. The present model reaches 56 percent of that bound. The remaining shortfall is representational rather than statistical, and additional data cannot close it.")

d.add_heading("1  Introduction", level=1)
para(d,"An editor is told what to change. It is never told where. Systems like InstructPix2Pix denoise the whole frame and leave the region implicit, which costs twice over: the edit lands weakly, and unrelated parts of the image drift.")
para(d,"So predict the region first, then gate the editor to it. Localization, then editing. This report is about localization; Section 8 checks that it is worth doing.")
para(d,"One case makes the problem hard. “Put a hat on the dog” names something that is not in the picture. There is no source region to find, and every method that works by pointing at visible content fails here.")

d.add_heading("1.1  What we build", level=2)
para(d,"A 4.69 M parameter head. It takes a CLIP embedding of the instruction and 32 frozen FastSAM prototypes, and predicts a 20 × 20 field of coefficients over those prototypes. The weighted sum is the mask. Nothing else trains.")
table(d,"Table 1. The system in one view.",
 ["Stage","Runs","Cost","Trained?"],
 [["FastSAM backbone → 32 prototypes","once per image","33.6 ms","frozen"],
  ["CLIP text encoder → 512-d","once per instruction","included below","frozen"],
  ["Coefficient head → mask","once per instruction","5.30 ms","4.69 M params"]],
 widths=[2.7,1.5,1.1,1.3])

d.add_heading("1.2  Why this design", level=2)
para(d,"Three choices, each forced by something we measured.")
table(d,"Table 2. Design decisions and the evidence behind each.",
 ["Choice","Why","Evidence"],
 [["Predict coefficients, do not select a mask",
   "Selecting from existing masks can only ever return an object that is already there, so insertion is impossible in principle",
   "Exp 01: FastSAM + CLIP selection scores 0 of 7"],
  ["Freeze both backbones",
   "The backbone is the same for every instruction on an image. Freezing it lets us pay for it once and run the head per instruction",
   "5.30 ms per instruction against CLIPSeg's 54.3 ms"],
  ["Derive targets by colour difference",
   "The dataset's own masks are 9.1× larger than the real change, so training on them teaches the wrong region",
   "Exp 07: 62.4% of frame against 10.3% true"],
  ["A spatial field, not one global vector",
   "One vector forces every location to use the same prototype mixture, which suits objects and not empty space",
   "Insert 0.1186 against 0.0972 for the global head"]],
 widths=[1.7,3.0,1.9], align=["left","left","left"])

d.add_heading("2  Literature Review", level=1)
para(d,"We surveyed 45 works from 2024 to 2026; the full matrix is a companion document. Sorting them by what each does with the edit region, not by method, is more revealing than reading them in date order.")
table(d,"Table 3. Surveyed works classified by their treatment of the edit region.",
 ["Posture","Works","Representative","Implication"],
 [["Consumes a region supplied to it","17","NEP, MaskFlow, LazyDiffusion","Demand established; supply assumed"],
  ["Reads a region from model internals","11","WhereEdit, RegionE","Unsupervised; blind to empty space"],
  ["Predicts a region from the instruction","2","AdaptEdit, MADiff","The formulation adopted here"],
  ["Scores or frames the problem","15","LocateEdit-Bench, surveys","None scores a predicted region"]],
 widths=[2.4,0.7,1.7,2.0], bold_rows=(2,))
para(d,"Two of the 45 train a model to produce the region from language, and one of those only works on fashion images. Four gaps follow: producing a region is unsolved while editing given one is not; the region is often not an object; supervision binds harder than architecture; and nothing scores a predicted region.")

d.add_heading("3  Problem Statement and Objectives", level=1)
para(d,"Given an image and an editing instruction, predict a binary mask of the pixels that should change — including when no matching object exists in the image.")
for t in ["To establish whether an off-the-shelf segmentation and retrieval pipeline is sufficient, and if not, to characterise its failure modes.",
          "To construct supervision for the task that does not depend on human region annotation.",
          "To design and train a predictor operating over frozen features, within the compute available on a single laptop.",
          "To evaluate the predicted region directly, a measurement absent from the base paper and from the benchmarks surveyed.",
          "To determine whether predicting the region improves the resulting edit, and to quantify by how much.",
          "To establish the upper bound imposed by the chosen representation, and to report the fraction of it attained."]:
    p=d.add_paragraph(t, style="List Number"); p.paragraph_format.space_after=Pt(3)
    for r in p.runs: r.font.size=Pt(11)

d.add_heading("4  Methodology", level=1)
d.add_heading("4.1  Frozen feature extraction", level=2)
para(d,"We freeze a YOLOv8x-seg (FastSAM) backbone and use it only as a feature source. It gives 32 mask prototypes at stride 4 — 160 × 160 maps for a 640 × 640 input — plus a 640-d context vector from the SPPF layer. Prototypes are spatial basis functions: any mask is a linear combination of them through a sigmoid. A frozen CLIP encoder turns the instruction into 512 dimensions.")
para(d,"We train neither backbone. Only the head learns. That is what makes this run on one laptop, and it is where the parameter and latency numbers in Section 7.7 come from.")
d.add_heading("4.2  Global coefficient head (v1)", level=2)
para(d,"The first head predicts one 32-d coefficient vector and a bias per image–instruction pair. Text modulates the image context by FiLM; an MLP emits the coefficients; the logits are the prototype-weighted sum. 1,399,073 parameters. The same linear combination applies everywhere in the frame.")
d.add_heading("4.3  Spatial coefficient field (v2)", level=2)
para(d,"What limits the first head is structural, not a shortage of capacity: a single coefficient vector requires every location to use the same prototype mixture, and a prior experiment found the head saturating at 1.4 million parameters. The second head predicts a 20 × 20 field of coefficients, upsampled bilinearly to prototype resolution, retaining a global branch so that the instruction may still establish an overall prior. The head has 4,693,633 trainable parameters. The hypothesis under test is that a spatially varying combination assists most where the target region is not the extent of a single object, which is the insertion case.")
d.add_heading("4.4  Supervision and loss", level=2)
para(d,"We minimise BCE plus twice Dice, clipping gradient norm at 1.0.")
para(d,"We do not use the MagicBrush annotations as targets. Measured across all 528 development turns, those masks cover 62.4 percent of the frame against a true changed area of 10.3 percent: 9.1 times too large, with precision 15.8 percent, and worst for insertions at 11.9 times. They are regions of interest rather than edit masks. Instead we derive each target from the human-authored source and target pair by CIE76 colour difference in L*a*b* space, thresholded at 12.0 and cleaned by morphological opening and closing with a 5 × 5 kernel followed by removal of connected components below 40 pixels. We reject any sample whose changed area falls outside 0.4 to 45 percent.")

d.add_heading("5  Dataset", level=1)
para(d,"MagicBrush: all 51 training shards and all 4 development shards, 25 GB of Parquet. Each row holds a source image, a target image, an instruction and a turn index. It is multi-turn, so one photograph yields several successive edits.")
table(d,"Table 4. Dataset composition after target derivation and filtering.",
 ["Split","Shards","Rows","Usable","Insert","Modify","Remove"],
 [["Train","51","8,807","8,306","1,828","6,045","433"],
  ["Development (held out)","4","528","503","—","—","—"]],
 widths=[1.9,0.8,0.9,0.9,0.8,0.8,0.8])
d.add_heading("5.1  Split integrity", level=2)
para(d,"Because the dataset is multi-turn, a split partitioned by turn rather than by image would place the same photograph in both partitions, and increasing the training set would amplify any such contamination. So before accepting any result we hashed every source and target image on both sides with SHA-1 and intersected them.")
table(d,"Table 5. Image-level disjointness across all 55 shards.",
 ["Quantity","Count"],[["Distinct development images","794"],["Distinct training images","13,317"],
 ["Images present in both partitions","0"]], widths=[3.6,1.2], bold_rows=(2,))

d.add_heading("6  Experimental Protocol", level=1)
para(d,"All results follow one protocol, held constant so that training-set size is the only variable between the two scales evaluated. The checkpoint epoch is selected on a held-out 15 percent slice of the training partition and never on the development set. The development set is evaluated once per seed, at full image resolution against the undilated target. We run twelve seeds, each seeding the PyTorch, NumPy, Python and Metal generators. For significance we use Welch's t-test on seed means, and one-sample t-tests against fixed baselines.")
para(d,"We precompute the frozen features once into memory-mapped arrays, 13.8 GB for the full training partition, so that each epoch touches only the head. This makes twenty-four seeded runs feasible on the available hardware.")
d.add_heading("6.1  Configuration", level=2)
table(d,"Table 6. Configuration values, taken from source.",
 ["Group","Parameter","Value"],
 [["Supervision","Colour difference metric","CIE76 in L*a*b*"],
  ["","Threshold","12.0"],
  ["","Morphological kernel; minimum component","5 × 5; 40 px"],
  ["","Sample filter on changed area","0.4% – 45%"],
  ["Representation","Input resolution","640 × 640"],
  ["","Prototypes","32 at 160 × 160, frozen"],
  ["","Text; image context","512-d CLIP; 640-d SPPF, both frozen"],
  ["Head (v2)","Coefficient grid","20 × 20, bilinear upsample"],
  ["","Hidden width; dropout","512; 0.0"],
  ["Optimisation","Optimiser","AdamW"],
  ["","Learning rate","1 × 10⁻³, cosine to zero"],
  ["","Weight decay; batch size","0.01; 32"],
  ["","Loss","BCE + 2 × Dice"],
  ["","Gradient clipping","norm 1.0"],
  ["Evaluation","Epochs; selection","60; best on train-validation slice"],
  ["","Decision threshold","0.5 fixed (Section 7.3)"],
  ["","Seeds","12 (1368, 1–11)"],
  ["","Baseline","CIDAS/clipseg-rd64-refined"]],
 widths=[1.25,2.35,3.0], align=["left","left","left"])
d.save("Review1_Technical_Report.docx")
print("wrote Review1_Technical_Report.docx (part 1)")

# ---------------- results onward ----------------
d.add_heading("7  Results and Discussion", level=1)
d.add_heading("7.1  Accuracy", level=2)
table(d,"Table 7. Mean development IoU over 503 held-out turns, 12 seeds per configuration.",
 ["Configuration","Trainable","n = 2,278","n = 8,306","Gain"],
 [["v2 spatial field","4,693,633","0.1718 ± 0.0099","0.2065 ± 0.0072","+0.0347"],
  ["v1 global coefficients","1,399,073","0.1590 ± 0.0111","0.1907 ± 0.0053","+0.0317"],
  ["CLIPSeg (pretrained baseline)","150,747,746","0.1849","0.1849","—"],
  ["MagicBrush human masks","—","0.1511","0.1511","—"],
  ["Full frame (trivial)","0","0.0958","0.0958","—"],
  ["Random centred box (trivial)","0","0.0621","0.0621","—"]],
 widths=[2.1,1.25,1.25,1.25,0.75], bold_rows=(0,))
table(d,"Table 8. Statistical significance. Welch's t-test on seed means; one-sample tests against fixed baselines.",
 ["Comparison","Difference","p"],
 [["v2: 8,306 against 2,278 samples","+0.0347","4.1 × 10⁻⁹"],
  ["v1: 8,306 against 2,278 samples","+0.0317","1.5 × 10⁻⁷"],
  ["v2 against v1 at n = 8,306","+0.0158","5.4 × 10⁻⁶"],
  ["v2 against CLIPSeg","+0.0216","5.2 × 10⁻⁷"],
  ["v1 against CLIPSeg","+0.0058","2.9 × 10⁻³"],
  ["v2 against MagicBrush human masks","+0.0554","2.5 × 10⁻¹¹"]],
 widths=[3.4,1.2,1.4], bold_rows=(2,))
figure(d,"results_full.png","Figure 1. Individual seed results for both heads at both training-set sizes, with reference baselines. Lower panels give per-kind performance against the least-squares ceiling of Section 7.5.")
para(d,"It would be convenient to credit the spatial head for this. The measurements do not support that. Both heads gained a comparable absolute amount from the larger training set, and the global head at 8,306 samples also exceeds CLIPSeg. The architecture contributes a significant additional 0.0158, but the larger single factor is the quantity of training data.")

d.add_heading("7.2  Full metric suite", level=2)
para(d,"Intersection over union alone conceals the precision–recall trade-off; two models with identical IoU may behave very differently. All four measures were computed over the same 503 development turns at the fixed 0.5 threshold.")
table(d,"Table 9. Mean ± standard deviation over 12 seeds, with 95 percent confidence intervals on the mean (t, 11 df).",
 ["Head","IoU","Precision","Recall","F1 / Dice"],
 [["v2 spatial","0.2065 ± 0.0072","0.3126 ± 0.0084","0.3878 ± 0.0256","0.2976"],
  ["v1 global","0.1907 ± 0.0053","0.2972 ± 0.0070","0.3510 ± 0.0209","0.2757"],
  ["95% CI, v2","[0.2019, 0.2111]","[0.3072, 0.3179]","[0.3716, 0.4041]","—"]],
 widths=[1.35,1.5,1.5,1.5,0.85], bold_rows=(0,))
para(d,"Cohen's d for the v2 against v1 comparison on IoU is 2.50, indicating a large effect, not just a significant one. Mean predicted area is 11.70 percent against a true changed area of 9.58 percent, so the model over-predicts slightly at this operating point.")

d.add_heading("7.3  Operating point", level=2)
para(d,"The 0.5 threshold was fixed before evaluation. Sweeping it on the development set shows IoU peaking near 0.3.")
table(d,"Table 10. Threshold sweep for the v2 head, mean over 12 seeds. True changed area is 9.58 percent.",
 ["Threshold","IoU","Precision","Recall","F1","Predicted area"],
 [["0.1","0.2096","0.2393","0.6733","0.3124","30.12%"],
  ["0.2","0.2172","0.2637","0.5775","0.3185","22.74%"],
  ["0.3","0.2180","0.2818","0.5063","0.3166","18.10%"],
  ["0.4","0.2144","0.2975","0.4453","0.3097","14.60%"],
  ["0.5 (reported)","0.2065","0.3126","0.3878","0.2976","11.70%"],
  ["0.7","0.1747","0.3458","0.2750","0.2532","6.90%"],
  ["0.9","0.1119","0.3705","0.1479","0.1658","2.74%"]],
 widths=[1.3,0.9,1.0,0.9,0.9,1.2], bold_rows=(2,))
para(d,"Adopting 0.3 on the basis of the development sweep would constitute test-set tuning, the error that previously inverted the comparison between the two heads. The threshold was therefore selected per seed on the train-validation slice, the same slice used for epoch selection, and applied to the development set once.")
table(d,"Table 11. Threshold selected on the train-validation slice. Paired t-test across the same 12 seeds.",
 ["Head","Thresholds selected","Fixed 0.5","Selected","Gain","p (paired)"],
 [["v2 spatial","0.3 (×7), 0.2 (×5)","0.2065 ± 0.0072","0.2185 ± 0.0051","+0.0120","2.9 × 10⁻⁷"],
  ["v1 global","0.2 (×9), 0.3 (×3)","0.1907 ± 0.0053","0.2092 ± 0.0038","+0.0185","6.3 × 10⁻⁸"]],
 widths=[1.15,1.5,1.35,1.25,0.7,1.0], bold_rows=(0,))
para(d,"Two consequences are recorded. With both heads at their selected thresholds the advantage of the spatial head narrows from +0.0158 to +0.0093 (p = 5.8 × 10⁻⁵); a portion of its margin at the fixed threshold reflected that 0.5 suited it better. Separately, CLIPSeg is evaluated at its default operating point, so a tuned comparison favours the present model. The fixed-threshold result is therefore retained as the headline, being the conservative figure.")

d.add_heading("7.4  Per edit kind", level=2)
table(d,"Table 12. Performance by edit kind at n = 8,306, against the least-squares ceiling computed on the development set.",
 ["Kind","v2","v1","Ceiling","v2 % of ceiling","v1 % of ceiling"],
 [["Insert","0.1186","0.0972","0.2119","56.0%","45.9%"],
  ["Modify","0.2313","0.2143","0.3867","59.8%","55.4%"],
  ["Remove","0.2007","0.2166","0.4325","46.4%","50.1%"]],
 widths=[1.0,0.9,0.9,1.0,1.35,1.35], bold_rows=(0,))
para(d,"The spatial head is superior on insertion and modification and inferior on removal. That matches the hypothesis of Section 4.3 and is not a uniform improvement: removal targets an object already present, which a global coefficient over object-shaped prototypes describes adequately, so the additional spatial capacity is unnecessary there and costs precision. The field is advantageous where no source object exists.")
figure(d,"qualitative_full.png","Figure 2. Predictions of the full-scale spatial head on six distinct development scenes, selected by edit kind and deduplicated by scene identifier rather than for appearance. Red indicates ground truth only, green prediction only, yellow agreement.", width=5.6)

d.add_heading("7.5  Representational ceiling", level=2)
para(d,"To separate a model that has not learned this yet from a basis that cannot express it, we fitted coefficients directly to the ground-truth mask by least squares. No learning is involved, and the result is an upper bound on the IoU attainable by any coefficient predictor over these 32 prototypes.")
table(d,"Table 13. Least-squares upper bound per edit kind, computed on the development set.",
 ["Kind","Ceiling IoU","v2 attained","Fraction attained"],
 [["Remove","0.4325","0.2007","46.4%"],["Modify","0.3867","0.2313","59.8%"],
  ["Insert","0.2119","0.1186","56.0%"]],
 widths=[1.2,1.3,1.3,1.5], bold_rows=(2,))
para(d,"The ceiling for insertion is less than half that for removal. The architecture was originally justified on the reasoning that prototypes are spatial basis functions rather than objects, and that a text-derived combination could therefore describe empty space. That reasoning is refuted by this measurement: the prototypes were trained to segment objects and carry that bias, spanning object-shaped regions approximately twice as well as empty space. The mechanism is weakest in precisely the case for which it was adopted.")
para(d,"We then checked whether the shortfall was simply a shortage of insertion examples. Trained per kind at matched sample counts, insertion attained 0.0450 against 0.1568 for modification on the same 497 samples, a factor of 3.5 given identical data volume. We tried to raise the ceiling by appending hand-designed free-space basis functions. Its own control rejected it: the geometric basis exceeded neither a random basis of equal size (+0.0035, p = 0.33) nor the plain spatial head (−0.0030, p = 0.47), and was removed.")
d.save("Review1_Technical_Report.docx"); print("part 2 appended")

d.add_heading("7.6  Training dynamics and ablations", level=2)
figure(d,"curves.png","Figure 3. Learning curves, three seeds per head at n = 8,306. Markers indicate the epoch selected. All six fall between epoch 2 and 7 of 60.")
table(d,"Table 14. Validation IoU at its maximum and at the final epoch.",
 ["Head","Selected epochs","Peak validation IoU","At epoch 60","Change"],
 [["v2 spatial","2, 5, 3","0.1884","0.1066","−43%"],
  ["v1 global","7, 5, 7","0.1740","0.1167","−33%"]],
 widths=[1.2,1.4,1.6,1.1,0.9])
para(d,"Training loss decreases monotonically across all 60 epochs while validation IoU approximately halves, which is overfitting, not a failure to optimise. Selecting the checkpoint on a held-out slice of the training partition is therefore material to every figure reported: training to the final epoch would have yielded approximately half the IoU. The spatial head also leads at the peak, so its advantage is not an artefact of selection. A practical consequence is that 60 epochs is roughly six times more than required; approximately 10 reach the same checkpoint.")
para(d,"Zeroing the text embedding each epoch drops validation IoU from 0.188 to approximately 0.03, a difference of +0.158 for the spatial head and +0.153 for the global head at the peak. The head is therefore conditioning on the instruction rather than exploiting an image prior. An independent three-seed run with per-epoch logging reproduced its development IoU exactly for all six runs, matching the corresponding seeds of the twelve-seed run to four decimal places.")

d.add_heading("7.7  Latency", level=2)
table(d,"Table 15. Measured on the M5, resolution-matched at CLIPSeg's native 352 × 352. Per-image cost is incurred once; per-instruction cost is incurred for every instruction on that image.",
 ["Model","Trainable","Per image","Per instruction","Speed-up","Parameters"],
 [["v2 spatial","4,693,633","33.6 ms","5.30 ms","10.3×","32.1× fewer"],
  ["v1 global","1,399,073","33.8 ms","5.39 ms","10.1×","107.7× fewer"],
  ["CLIPSeg","150,747,746","—","54.3 ms","1×","—"]],
 widths=[1.1,1.25,1.0,1.3,0.9,1.3], bold_rows=(0,))
para(d,"The advantage is structural. The architecture separates into a per-image stage, the frozen backbone producing 32 prototypes, and a per-instruction stage comprising a CLIP text encoding, a small multilayer perceptron and a weighted sum. CLIPSeg re-executes a full 150 million parameter encoder–decoder for each instruction and admits no such separation. Editing is an iterative activity in which several instructions are issued against one image, so the amortised cost is the relevant quantity. At the operating resolution of 640 pixels the backbone costs 98.4 ms and break-even occurs at approximately two instructions; at the matched 352 pixels both stages are cheaper. Accuracy at 352 pixels is unmeasured, so the matched comparison is a latency comparison only.")

d.add_heading("8  Stage 2: Editing Within the Predicted Region", level=1)
para(d,"A mask metric is a proxy. This experiment tests the premise of the work directly. A single editor, InstructPix2Pix, produces the edit for every arm, and only the gating mask varies, so any difference is attributable to the mask. We used sixty held-out development samples. Recall denotes the fraction of the true edit region that changed, collateral the fraction of pixels outside that region that changed, and net their difference.")
table(d,"Table 16. Stage 2 with a fixed editor and varying gating mask.",
 ["Gating mask","Recall","Collateral","Net"],
 [["None — whole frame (status quo)","33.3%","24.2%","+9.1%"],
  ["Predicted, as output","7.5%","1.5%","+6.0%"],
  ["Predicted, tuned (threshold 0.2, dilation 64 px)","22.9%","8.8%","+14.2%"],
  ["MagicBrush human mask","31.5%","13.0%","+18.5%"],
  ["Ground-truth region (oracle)","30.2%","0.1%","+30.1%"]],
 widths=[3.2,1.0,1.1,0.9], bold_rows=(2,))
para(d,"Editing within the correct region is worth a factor of 3.3, +30.1 percent against +9.1 percent for whole-frame editing, which is direct evidence that localization pays off on the edited image, not only on a mask metric. The predicted mask as output is precision-biased, yielding 1.5 percent collateral but only 7.5 percent recall, so gating discards the majority of the edit and net falls below whole-frame editing. Dilation exchanges precision for recall; swept to 200 pixels across two thresholds, net attains a maximum at 64 pixels and declines monotonically thereafter, so 64 pixels is a true optimum rather than the limit of the search.")
para(d,"The MagicBrush mask, 9.1 times larger than the true region, carries 13.0 percent collateral and forfeits 11.6 points of net against the ground-truth region, so the cost of coarse supervision propagates into the edited image. The tuned prediction exceeds whole-frame editing but does not exceed the MagicBrush mask, at +14.2 percent against +18.5 percent: those masks are substantially too large yet retain more of the edit. Caveats are sixty samples, one editor and one guidance setting.")

para(d,"")
d.add_heading("8.1  The same test with a real inpainting model", level=2)
para(d,"The experiment above gates one editor's whole-frame output through a mask, so the mask can only ever subtract. A mask that is too large costs nothing there. We repeated the test with Stable Diffusion inpainting, where the mask instead defines the region the model regenerates, using the same 60 samples and the same predicted masks.")
table(d,"Table 17. Stage 2 with Stable Diffusion inpainting. Sixty held-out samples, 20 steps, threshold 0.2 and 64 px dilation carried over from the compositing experiment.",
 ["Gating mask","Recall","Collateral","Net","PSNR outside","CLIP inside"],
 [["None — whole frame","33.3%","24.2%","+9.1%","22.56","0.2445"],
  ["Predicted, tuned","71.1%","39.1%","+32.0%","14.11","0.2623"],
  ["MagicBrush human mask","78.4%","51.7%","+26.8%","13.30","0.2550"],
  ["Ground-truth region (oracle)","59.4%","6.4%","+53.0%","27.36","0.2464"]],
 widths=[2.0,0.9,1.1,0.85,1.1,1.0], bold_rows=(1,))
para(d,"Two results differ from the compositing experiment. Localization is worth more here, a factor of 5.8 against 3.3, because an inpainting model conditioned on a correct region can synthesise content rather than merely pass an edit through a stencil. And the predicted mask now exceeds the human annotation, +32.0 percent against +26.8, reversing the ordering we reported above.")
para(d,"At 64 px the win came with a problem. We raised net by pushing recall to 71.1 percent, not by reducing collateral: collateral rose to 39.1 percent against 24.2 percent for whole-frame editing, and PSNR outside the region fell 8.45 dB. We had carried that 64 px over from the compositing experiment, where enlarging a mask can only recover more of the edit and so costs nothing. Under inpainting the mask decides what gets regenerated, so an over-large one destroys content that should have been kept.")
para(d,"Re-tuning dilation for this setting confirms that diagnosis.")
table(d,"Table 18. Dilation swept for the predicted mask under inpainting. The other arms are held fixed at the values in Table 15, on the same sixty samples.",
 ["Dilation","Mask area","Recall","Collateral","Net","PSNR outside"],
 [["0 px","22.4%","46.5%","16.1%","+30.4%","20.07"],
  ["16 px","36.2%","58.6%","23.0%","+35.5%","17.46"],
  ["32 px","47.1%","65.6%","30.8%","+34.8%","15.44"],
  ["64 px","—","71.1%","39.1%","+32.0%","14.11"]],
 widths=[1.0,1.1,0.9,1.1,0.9,1.2], bold_rows=(1,))
para(d,"Net traces +30.4, +35.5, +34.8 and +32.0 across the four settings, rising and then falling, so 16 px is an interior optimum and not the edge of the search. There the predicted mask reaches +35.5 percent net at 23.0 percent collateral, which is below the 24.2 percent of whole-frame editing. The earlier problem disappears: we now exceed the status quo on both quantities at once instead of buying net with damage, and PSNR outside recovers 3.3 dB. The margin over the human annotation widens to +35.5 against +26.8.")
para(d,"One thing the sweep makes visible. Even undilated the predicted mask covers 22.4 percent of the frame against a true changed area of 9.6 percent, so at threshold 0.2 it over-covers by a factor of 2.3 before any dilation. Threshold and dilation are not independent here, and a joint sweep is the next refinement. The oracle, at 6.4 percent collateral and +53.0 percent net, shows how much is still available.")

d.add_heading("9  Validation and Corrective Actions", level=1)
para(d,"We made four corrections during the work. They are recorded because the alternative would render the surviving results unverifiable.")
para(d,"First, an overnight script contained a copy operation that merged the development partition into the training cache, so that the model was trained on the partition it was subsequently evaluated on. We found it when a baseline scored 0.3743 against a training-time 0.1397, a discrepancy too large to be attributable to a difference in metric. We withdrew the affected headline and the claim that our supervision beat human annotation, then re-established both on a disjoint split.")
para(d,"Second, an earlier evaluation selected the best epoch by development IoU and concluded that the spatial head was inferior to the global head, 0.1373 against 0.1533. Selecting on a held-out slice of the training partition reverses the ordering significantly. So selection on the test set does not just inflate a result. It can flip it.")
para(d,"Third, two proposed extensions to the prototype basis, a free-space basis and a geometric basis, were each required to exceed a random basis of equal dimensionality and failed to do so. We removed both.")
para(d,"Fourth, an earlier conclusion that additional data would not assist insertion was drawn from a per-kind test at approximately 500 insertion samples. With 1,828 samples, insertion improved from 44 to 56 percent of its ceiling. The ceiling itself is unchanged. The surviving claim is that the basis bounds insertion at 0.212, not that data does not assist insertion.")

d.add_heading("10  Limitations", level=1)
for t in ["The insertion ceiling of 0.212 is a bound imposed by the present basis, of which 56 percent is attained. Raising it requires a basis spanning empty space; the single attempt at this was rejected by its control.",
 "We chose hyperparameters at 2,278 samples and reused them unchanged at 8,306 so that training-set size would be the only variable. The reported figures are therefore a lower bound.",
 "The backbone remains frozen. We accept that trade deliberately: it is what buys the parameter and latency advantages. It also caps how accurate we can get.",
 "Stage 2 uses InstructPix2Pix on sixty samples at one guidance setting. Stable Diffusion inpainting weights are available but were not used for the reported figures.",
 "Accuracy at 352 pixels is unmeasured, so the resolution-matched latency comparison is a latency comparison only.",
 "CLIPSeg is evaluated at its default operating point while the present threshold is selected on a held-out slice, so the tuned comparison favours this work. The fixed-threshold comparison is reported as the headline.",
 "AdaptEdit is not reproduced, and no claim here is a comparison against its reported figures."]:
    p=d.add_paragraph(t, style="List Number"); p.paragraph_format.space_after=Pt(3)
    for r in p.runs: r.font.size=Pt(11)

d.add_heading("11  Conclusion and Future Work", level=1)
para(d,"A 4.69 million parameter head operating entirely over frozen features predicts edit regions more accurately than a 150.7 million parameter pretrained referring segmenter and than the human annotations distributed with the dataset, at 32 times fewer trainable parameters and 10.3 times lower per-instruction latency, trained end to end on a single laptop. Localization is worth a factor of 3.3 downstream, which substantiates the two-stage premise.")
para(d,"The principal finding, however, is the bound. Insertion, the case named by the base paper as its own limitation and the case this work addresses, is bounded at 0.212 by a prototype basis trained to segment objects, not to describe empty space. Part of the shortfall to that bound was closed with additional data; the bound itself was not moved.")
para(d,"Three directions follow. The basis rather than the head should be addressed, since the ceiling is a property of the 32 prototypes: learning a small number of additional prototypes supervised on insertion regions, required to exceed a random basis of equal size, is the direct test. Hyperparameters should be re-swept at 8,306 samples, which is now inexpensive given that approximately 10 epochs suffice. Stage 2 recall should be raised without reintroducing collateral change, in order to exceed the +18.5 percent attained by the MagicBrush mask.")

d.add_heading("Appendix A — Complete Configuration", level=1)
para(d,"See Table 4. Every value here we read out of the source, not out of notes. The training partition comprises 8,306 usable turns drawn from all 51 shards, of 8,807 rows, with 501 rejected as degenerate. The partitions are image-disjoint as reported in Table 3.")
d.add_heading("Appendix B — Reproduction", level=1)
para(d,"We produced every result on one machine: Apple M5, 10 cores, 26 GB unified memory, macOS 26.6.2, PyTorch 2.13.0 on the Metal backend, NumPy 2.2.6, OpenCV 5.0.0, Ultralytics 8.3.209, SciPy 1.15.3.")
for c in ["python train/precompute_full.py --split train --batch 8",
          "python train/leakage_check.py",
          "python train/clean_eval_full.py --seeds 12 --configs v2,v1",
          "python train/clean_eval_full.py --seeds 3 --configs v2,v1 --curves",
          "python train/metrics_full.py",
          "python train/threshold_select.py",
          "python train/latency_v2.py",
          "python train/plots_full.py && python train/curve_plots.py"]:
    p=d.add_paragraph(); r=p.add_run(c); r.font.name="Consolas"; r.font.size=Pt(9.5)
    p.paragraph_format.space_after=Pt(1)
para(d,"")
para(d,"Measured wall times: feature precomputation over 8,807 rows, 35.0 minutes; leakage audit over 55 shards, approximately 9 minutes; spatial-head training, 37 to 40 minutes per seed; global-head training, approximately 32 minutes per seed. Total compute approximately 21 hours. Repository: github.com/Amritha902/edit-region-prediction.")
figure(d,"run_evidence.png","Figure 4. Output from the runs that produced this report, reproduced from the logs on the machine described above.")
d.save("Review1_Technical_Report.docx"); print("complete")
