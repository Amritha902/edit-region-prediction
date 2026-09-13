import sys; sys.path.insert(0,".")
from mkdocx import *

d = base_doc()
titlepage(d,"Consolidated Project Report",
 "Disentangling Localization from Editing in Instruction-Guided Image Manipulation",
 "Complete record of experimental work, results and corrective actions")
toc(d,[("1","Introduction",1),("2","Project Overview",1),("3","Experimental Work",1),
 ("3.1","Summary of experiments",2),("3.2","Experiments in detail",2),
 ("4","Consolidated Results",1),("5","Validation and Corrective Actions",1),
 ("6","Repository Structure",1),("7","Environment and Reproduction",1),
 ("8","Open Problems and Future Work",1),("9","Companion Documents",1)])

d.add_heading("1  Introduction", level=1)
para(d,"An instruction-guided image editor is informed what to change but not where. Contemporary editors denoise the entire frame and determine the affected region implicitly, which produces two failures simultaneously: the intended edit is applied weakly, and unrelated regions drift. The fix is to predict the edit region first and gate the editor to it.")
para(d,"The base paper for this work, AdaptEdit, performs exactly this and names in its own limitations section the case it cannot address: an instruction such as “put a hat on the dog” where the dog has no hat, for which no source region exists. This project constructs that predictor at a scale that can be executed and verified, and supplies the measurement the base paper omits.")
para(d,"This document consolidates the complete record: every experiment conducted, what each established, the corrective actions taken, and the problems that remain open. The technical report argues the result and the literature survey positions it; this document is the audit trail.")

d.add_heading("2  Project Overview", level=1)
table(d,"Table 1. Project at a glance.",["Item","Detail"],
 [["Problem","Predict the pixels an editing instruction requires to change, including where no source object exists"],
  ["Base paper","AdaptEdit, arXiv:2604.23763, April 2026"],
  ["Approach","A 4.69 M parameter head predicting a spatial coefficient field over 32 frozen FastSAM prototypes, conditioned on a CLIP text embedding"],
  ["Dataset","MagicBrush, 51 training shards and 4 development shards; 8,306 usable training turns, 503 held-out"],
  ["Supervision","CIE76 colour difference in L*a*b* between authentic source and target pairs"],
  ["Headline result","IoU 0.2065 ± 0.0072 over 12 seeds, exceeding CLIPSeg (0.1849) and the dataset's human annotations (0.1511)"],
  ["Efficiency","32× fewer trainable parameters and 10.3× lower per-instruction latency than CLIPSeg"],
  ["Principal limitation","Insertion is bounded at 0.212 by the prototype basis; 56 percent attained"],
  ["Hardware","Apple M5, 26 GB unified memory; approximately 21 hours of compute"]],
 widths=[1.5,5.1], align=["left","left"])

d.add_heading("3  Experimental Work", level=1)
d.add_heading("3.1  Summary of experiments", level=2)
para(d,"We ran thirteen numbered experiments in sequence. Several came back against the working hypothesis. In those cases we changed the direction of the project, not the number.")
table(d,"Table 2. All experiments, the question addressed and the outcome.",
 ["No.","Question","Result","Outcome"],
 [["01","Is an off-the-shelf FastSAM and CLIP selector sufficient?","0 of 7","Fails on granularity, spatial language and insertion"],
  ["02","Can supervision be manufactured by editing and differencing?","Confirmed","Colour difference recovers the changed region reliably"],
  ["03","Can empty space be resolved from language?","20 of 20","Resolves to regions no object occupies"],
  ["04","Does a per-image and per-instruction split amortise?","Confirmed","Backbone once, head per instruction"],
  ["07","Are the MagicBrush masks usable as targets?","9.1× too large","No; 62.4% of frame against 10.3% true change"],
  ["08","Why does insertion not improve with data?","Ceiling 0.212","A representational bound established by least squares"],
  ["09","Are the headline figures clean?","Contaminated","Development data had been merged into training; retracted"],
  ["10","Do hand-designed free-space basis functions assist?","p = 0.33","Rejected by a random-basis control of equal size"],
  ["11","Is head width the constraint?","Saturates","No; expressiveness rather than capacity"],
  ["12","Does a spatial coefficient field exceed a global one?","p = 0.0070","Yes at 2,278 samples; geometric basis removed"],
  ["13","Does predicting the region improve the edit?","3.3×","Yes; the premise holds downstream"],
  ["14","Does a 3.65-fold increase in data assist?","0.1718 → 0.2065","Yes, p = 4.1 × 10⁻⁹; now exceeds CLIPSeg"],
  ["15","What do precision, recall and threshold indicate?","0.2185","Threshold selected on train-validation; spatial lead narrows"]],
 widths=[0.45,2.3,1.15,2.7], align=["center","left","left","left"], bold_rows=(6,11))
para(d,"Experiments 05 and 06, concerning edit cost and incremental editing, were exploratory and produced no claim used in the submission. They are retained in the repository for completeness.")

d.add_heading("3.2  Experiments in detail", level=2)
para(d,"Experiment 01 established the baseline. Selecting a FastSAM mask by CLIP similarity was evaluated on seven representative instructions and succeeded on none. Three distinct failure modes were identified: granularity, in which the whole person is returned for an instruction concerning a jacket; spatial language, in which the ranking is insensitive to positional qualifiers; and insertion, which fails in principle because no candidate mask corresponds to empty space.")
para(d,"Experiment 07 measured the supervision available from the dataset. Across all 528 development turns the distributed masks cover 62.4 percent of the frame against a true changed area of 10.3 percent, with precision 15.8 percent and an IoU of 0.16 against the actual change. Insertions are worst at a factor of 11.9. This experiment also turned up the fact that the distributed masks use the convention that bright pixels indicate preservation rather than editing; we had assumed the opposite and got a recall of 0.3 percent before catching it.")
para(d,"Experiment 08 is the principal negative result. We fitted coefficients directly to the ground-truth mask by least squares, with no learning involved, establishing an upper bound on the IoU attainable by any coefficient predictor over the 32 prototypes: 0.2119 for insertion, 0.3867 for modification and 0.4325 for removal, computed on the development set. Insertion, the case the project addresses, carries the lowest bound, because prototypes trained to segment objects span empty space poorly.")
para(d,"Experiment 13 closed the loop. A single editor produced the edit for every arm and only the gating mask varied, so differences are attributable to the mask alone. Editing within the ground-truth region yields a net improvement of 30.1 percent against 9.1 percent for whole-frame editing, a factor of 3.3.")
para(d,"Experiment 14 scaled the training set from 2,278 to 8,306 samples with every other variable held constant, including hyperparameters selected at the smaller scale. Both heads improved by a comparable absolute amount, and the global head also came to exceed CLIPSeg, indicating that the improvement is attributable first to data and second to architecture.")

d.add_heading("4  Consolidated Results", level=1)
table(d,"Table 3. All headline figures. Development set comprises 503 held-out MagicBrush turns, 12 seeds, threshold 0.5 unless stated.",
 ["Quantity","Value","Reference"],
 [["Development IoU, spatial head","0.2065 ± 0.0072","CLIPSeg 0.1849; human masks 0.1511"],
  ["Development IoU, global head","0.1907 ± 0.0053","Also exceeds CLIPSeg"],
  ["Precision / Recall / F1","0.3126 / 0.3878 / 0.2976","Predicted area 11.70% against true 9.58%"],
  ["With threshold selected on train-validation","0.2185 ± 0.0051","Paired p = 2.9 × 10⁻⁷"],
  ["Spatial against global head","+0.0158, p = 5.4 × 10⁻⁶","Cohen's d = 2.50"],
  ["Insertion","0.1186","CLIPSeg 0.054; ceiling 0.2119"],
  ["Per-instruction latency","5.30 ms","CLIPSeg 54.3 ms; 10.3× at 352 px"],
  ["Trainable parameters","4,693,633","CLIPSeg 150,747,746; 32× fewer"],
  ["Stage 2, value of localization","+30.1% against +9.1%","Factor of 3.3 over whole-frame editing"],
  ["Training set","8,306 turns","51 shards; no image shared with the held-out set"]],
 widths=[2.5,1.7,2.4], align=["left","left","left"], bold_rows=(0,))

d.add_heading("5  Validation and Corrective Actions", level=1)
para(d,"We made four corrections. We record them in full because the alternative would render the surviving results unverifiable.")
para(d,"5.1  Contamination of the training cache. An overnight script contained a copy operation merging the development partition into the training cache, so that the model was trained on the partition subsequently used for evaluation. We found it when a baseline scored 0.3743 against a training-time figure of 0.1397, a discrepancy too large to be attributable to a difference in metric. We withdrew the affected headline and the claim that our supervision beat human annotation, then re-established both on a disjoint split.")
para(d,"5.2  Selection on the evaluation set. An earlier evaluation selected the best epoch by development IoU and concluded that the spatial head was inferior to the global head, 0.1373 against 0.1533. Selecting on a held-out slice of the training partition reverses the ordering significantly. So selection on the evaluation set does not just inflate a result. It can flip it.")
para(d,"5.3  Extensions rejected by their own controls. Two proposed additions to the prototype basis, a free-space basis and a geometric basis, were each required to exceed a random basis of equal dimensionality. The geometric basis exceeded neither the random control (+0.0035, p = 0.33) nor the plain spatial head (−0.0030, p = 0.47). We removed both, and the gap they addressed is still open.")
para(d,"5.4  A claim subsequently qualified. Experiment 08 concluded that additional data would not assist insertion, on the basis of a per-kind test at approximately 500 insertion samples. With 1,828 samples insertion improved from 44 to 56 percent of its ceiling. The ceiling itself is unchanged. The surviving claim is that the basis bounds insertion at 0.212, not that data does not assist insertion.")
para(d,"We applied two further checks before accepting any result. The partitions were verified image-disjoint by computing the SHA-1 digest of every encoded source and target image on both sides, yielding zero of 794 development images present among 13,317 training images. Separately, the latency comparison was re-measured at matched resolution after it was found that the original figure compared the present system at 640 pixels against CLIPSeg at its native 352, and had been measured on the global head rather than the head reported.")

d.add_heading("6  Repository Structure", level=1)
table(d,"Table 4. Principal files. Repository: github.com/Amritha902/edit-region-prediction",
 ["Path","Contents"],
 [["train/model.py","Global-coefficient head and frozen backbone wrapper"],
  ["train/model_v2.py","Spatial coefficient field"],
  ["train/dataset.py","Dataset loader, target derivation, degeneracy filter"],
  ["train/precompute_full.py","Frozen-feature cache over all 51 shards, memory-mapped and resumable"],
  ["train/clean_eval_full.py","The twelve-seed protocol with per-epoch curve logging"],
  ["train/leakage_check.py","Image-level partition disjointness audit"],
  ["train/metrics_full.py","Precision, recall, F1, threshold sweep, confidence intervals, effect size"],
  ["train/threshold_select.py","Threshold selected on train-validation and applied once"],
  ["train/latency_v2.py","Both heads, resolution-matched against the baseline"],
  ["train/stage2*.py","Gated editing and the dilation sweep"],
  ["experiments/exp*/FINDINGS.md","One write-up per experiment, including those that failed"],
  ["PROVENANCE.md","Machine, library versions, exact commands and measured timings"]],
 widths=[2.2,4.4], align=["left","left"])

d.add_heading("7  Environment and Reproduction", level=1)
para(d,"We produced every result on a single machine: Apple M5, 10 cores, 26 GB unified memory, macOS 26.6.2, PyTorch 2.13.0 on the Metal backend, NumPy 2.2.6, OpenCV 5.0.0, Ultralytics 8.3.209, SciPy 1.15.3, Python 3.10.20.")
para(d,"Measured wall times: feature precomputation over 8,807 rows, 35.0 minutes; partition audit over 55 shards, approximately 9 minutes; spatial-head training, 37 to 40 minutes per seed; global-head training, approximately 32 minutes per seed. Total compute is approximately 21 hours.")

d.add_heading("8  Open Problems and Future Work", level=1)
for t in ["The insertion ceiling. The bound of 0.212 is a property of the present basis, of which 56 percent is attained. Raising it requires a basis capable of spanning empty space, and the single attempt at this was rejected by its control. This is the research problem and the basis of any claim to novelty.",
 "Hyperparameter selection. All values were selected at 2,278 samples and reused unchanged at 8,306 so that training-set size would be the only variable. The reported figures are therefore a lower bound. A re-sweep is now inexpensive, since approximately 10 epochs suffice rather than 60.",
 "Stage 2 recall. The tuned prediction attains +14.2 percent net against +18.5 percent for the MagicBrush mask. Those masks are substantially too large yet retain more of the edit.",
 "Baseline parity. CLIPSeg is evaluated at its default operating point. A like-for-like comparison would require a held-out partition on which to tune it.",
 "Accuracy at 352 pixels is unmeasured, so the resolution-matched latency figure is a latency comparison only."]:
    p=d.add_paragraph(t, style="List Number"); p.paragraph_format.space_after=Pt(4)
    for r in p.runs: r.font.size=Pt(11)

d.add_heading("9  Companion Documents", level=1)
table(d,"Table 5. The submission set.",["Document","Contents"],
 [["Technical Report","Method, protocol, complete results, statistics, limitations and reproduction"],
  ["Literature Survey","Forty-five works from 2024 to 2026, nine clusters, the four gaps and comparable-baseline positioning"],
  ["Consolidated Project Report","This document: experimental record, corrective actions, repository structure and open problems"],
  ["Presentation","Thirteen slides with speaker notes"]],
 widths=[2.0,4.6], align=["left","left"])
d.save("Review1_Consolidated_Report.docx"); print("consolidated complete")
