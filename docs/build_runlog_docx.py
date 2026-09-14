import sys; sys.path.insert(0,".")
from mkdocx import *

d = base_doc()
titlepage(d,"Execution Record",
 "Disentangling Localization from Editing in Instruction-Guided Image Manipulation",
 "Every run, every failure, and how each was resolved")
toc(d,[("1","What This Project Is",1),("1.1","The base paper and the gap",2),
 ("1.2","What was built",2),("1.3","What it achieves",2),
 ("2","Hardware and Environment",1),("3","Runs in Order",1),
 ("4","Failures Encountered and Their Resolution",1),
 ("4.1","Data and supervision",2),("4.2","Protocol and measurement",2),
 ("4.3","Infrastructure",2),("4.4","Hardware",2),
 ("5","Results Retracted or Rejected",1),("6","Measured Timings",1),
 ("7","Reproduction",1)])

d.add_heading("1  What This Project Is", level=1)
para(d,"An instruction-guided image editor is told what to change. It is never told where. Systems such as InstructPix2Pix denoise the entire frame and leave the affected region implicit, which costs twice over: the intended edit is applied weakly, and unrelated parts of the image drift.")
para(d,"The fix is to predict the region that should change and gate the editor to it, splitting the task into localization and then editing. This project builds the first stage and measures it.")
para(d,"One case makes the problem hard. An instruction such as \u201cput a hat on the dog\u201d names an object that is not in the picture. There is no source region to find, and every method that works by pointing at visible content fails on it.")
d.add_heading("1.1  The base paper and the gap", level=2)
para(d,"AdaptEdit (arXiv:2604.23763, April 2026) is the only work that trains a predictor to ground the edit region from the instruction. Two facts about it shape this project. It cannot be reproduced on available hardware: 20.43 billion frozen parameters plus roughly 5 billion trainable. And it reports no mask IoU at all, so the quality of its localization is unmeasured. Its own limitations section names the case above as unsolved.")
para(d,"This project therefore reimplements its mechanism at a scale that can be run and verified, and supplies the measurement the base paper omits.")
d.add_heading("1.2  What was built", level=2)
para(d,"A 4,693,633 parameter head. It takes a CLIP embedding of the instruction and 32 frozen FastSAM prototypes, and predicts a 20 by 20 field of coefficients over those prototypes; the weighted sum is the mask. Nothing else trains.")
table(d,"Table 1. The system. The backbone depends on the image alone, so it is computed once and reused for every instruction on that image.",
 ["Stage","Runs","Cost","Trained"],
 [["FastSAM backbone to 32 prototypes","once per image","33.6 ms","frozen"],
  ["CLIP text encoder to 512-d","per instruction","included below","frozen"],
  ["Coefficient head to mask","per instruction","5.30 ms","4,693,633 params"]],
 widths=[2.7,1.5,1.2,1.5])
d.add_heading("1.3  What it achieves", level=2)
table(d,"Table 2. Headline results. Development set is 503 held-out MagicBrush turns, 12 seeds.",
 ["Quantity","Value","Against"],
 [["Dev IoU","0.2065 +/- 0.0072","CLIPSeg 0.1849; human masks 0.1511"],
  ["Significance vs CLIPSeg","p = 5.2 x 10^-7","Cohen's d 2.50 against the global head"],
  ["Precision / Recall / F1","0.3126 / 0.3878 / 0.2976","at threshold 0.5"],
  ["Per-instruction latency","5.30 ms","CLIPSeg 54.3 ms, a factor of 10.3"],
  ["Trainable parameters","4,693,633","CLIPSeg 150,747,746, 32x fewer"],
  ["Stage 2, value of localization","+53.0% against +9.1%","5.8x over whole-frame editing"],
  ["Stage 2, tuned operating point","+39.5% net at 22.2% collateral","below whole-frame's 24.2%"],
  ["Insertion ceiling","0.2119","a least-squares bound; 56% reached"]],
 widths=[2.0,1.7,2.9], align=["left","left","left"])
para(d,"The last row is the result the project regards as most useful and it is a negative one. Fitting coefficients directly to the ground-truth mask by least squares bounds the IoU any predictor can reach over this basis. Insertion, the case the project exists for, carries the lowest bound, because prototypes trained to segment objects span empty space poorly. More data cannot pass it.")

d.add_heading("2  Hardware and Environment", level=1)
table(d,"Table 3. The single machine every result was produced on.",
 ["Component","Detail"],
 [["Machine","Apple M5, 10 cores, 26 GB unified memory"],
  ["OS","macOS 26.6.2 (build 25G83)"],
  ["Compute backend","PyTorch 2.13.0, Metal Performance Shaders (MPS)"],
  ["Python","3.10.20"],
  ["Libraries","NumPy 2.2.6, OpenCV 5.0.0, pandas 2.3.3, SciPy 1.15.3, Pillow 12.3.0"],
  ["Models","Ultralytics 8.3.209, diffusers 0.39.0, transformers 5.15.0"],
  ["Storage used","26 GB dataset, 13.8 GB feature cache, 4.8 GB diffusion weights"],
  ["Total compute","approximately 21 hours"]],
 widths=[1.7,4.9], align=["left","left"])

d.add_heading("3  Runs in Order", level=1)
table(d,"Table 4. Every run, in sequence, with its outcome.",
 ["#","Run","Purpose","Wall time","Outcome"],
 [["1","precompute_full.py","Cache frozen features for all 51 shards","35.0 min","8,306 of 8,807 usable, 13.8 GB"],
  ["2","leakage_check.py","Hash every image on both sides of the split","~9 min","0 of 794 dev images in train"],
  ["3","clean_eval_full.py","12 seeds x 2 heads at full scale","~9 h","v2 0.2065, v1 0.1907"],
  ["4","clean_eval_full.py --curves","3 seeds per head with per-epoch logging","~4 h","best epoch 2-7 of 60"],
  ["5","metrics_full.py","Precision, recall, F1, threshold sweep, CIs","~13 min","P 0.3126 R 0.3878 F1 0.2976"],
  ["6","threshold_select.py","Threshold chosen on train-val, applied once","~5 min","0.2185, +0.0120 over fixed 0.5"],
  ["7","latency_v2.py","Both heads, resolution-matched","~2 min","5.30 ms/instruction, 10.3x"],
  ["8","stage2.py","Stage 2 with SD inpainting, 4 arms","107 min","localization worth 5.8x"],
  ["9","mask_grid.py","Test a free geometric proxy for net","~3 min","REJECTED, r = 0.318"],
  ["10","stage2_sd_sweep.py","Dilation sweep at threshold 0.2","62 min","16 px optimal on that slice"],
  ["11","stage2_sd_sweep.py (joint)","20-cell threshold x dilation grid","~6 h","optimum 0.3 / 32 px, +39.5%"],
  ["12","plots_full / curve_plots / grid_figure","Figures","~2 min","5 figures"]],
 widths=[0.35,1.6,2.2,0.85,1.8], align=["center","left","left","left","left"])

d.add_heading("4  Failures Encountered and Their Resolution", level=1)
d.add_heading("4.1  Data and supervision", level=2)
table(d,"Table 5.",["Problem","How it surfaced","Resolution"],
 [["MagicBrush mask polarity inverted","Recall 0.3%, IoU 0.001 on the first run",
   "Their masks are bright = PRESERVE, not bright = edit. All numbers recomputed."],
  ["Their masks unusable as targets","Measured 62.4% of frame against a true 10.3%",
   "Targets derived by CIE76 dE in L*a*b* from real source/target pairs instead."],
  ["Feature cache too large for RAM","13.8 GB plus a same-sized staging copy on a 26 GB machine",
   "Rewrote the cache as flat memmaps; peak RSS stays at one batch. Verified bit-identical."]],
 widths=[1.7,2.3,2.6], align=["left","left","left"])

d.add_heading("4.2  Protocol and measurement", level=2)
table(d,"Table 6.",["Problem","How it surfaced","Resolution"],
 [["Dev merged into train","A baseline scored 0.3743 against a training-time 0.1397",
   "An overnight script copied the combined cache over the train cache. Headline retracted, re-run on disjoint shards."],
  ["Epoch selected on the evaluation set","v2 appeared worse than v1, 0.1373 against 0.1533",
   "Selection moved to a held-out slice of train. The ordering reversed."],
  ["Seeding incomplete","Two identical commands gave 0.1215 and 0.1073",
   "Data permutation was seeded but weight init was not. All four generators now seeded."],
  ["Checkpoint chosen by IoU","Epoch-1 blob predictors scored ~0.14 by covering 20% of the frame",
   "Selection switched to the language delta, which those predictors fail."],
  ["Ceiling compared across splits","A train-set bound was quoted against a dev score",
   "Recomputed on dev. The remove ceiling moved 0.4575 to 0.4325."],
  ["Latency not resolution-matched","Ours measured at 640 px against CLIPSeg at its native 352",
   "Re-measured matched, and on the reported head rather than v1."],
  ["Collateral ratio printed inverted","Output read '0.6x less' where it was 1.62x more",
   "Ratio computed the wrong way and labelled unconditionally. Fixed before it entered a document."]],
 widths=[1.7,2.3,2.6], align=["left","left","left"])

d.add_heading("4.3  Infrastructure", level=2)
table(d,"Table 7.",["Problem","How it surfaced","Resolution"],
 [["Long jobs killed repeatedly","Three sweeps died with no error, each at a session boundary",
   "Split the work into one cell per job, each writing its JSON on completion, with a resume guard that skips banked cells."],
  ["Results written only at the end","A 10-cell run died having saved nothing",
   "Incremental writes after every cell. The completed cells were recovered from the log."],
  ["Checkpoints silently not committed","git add -A reported success; the remote held 0 .pt files",
   ".gitignore had *.pt for the backbone weights. Added an exception; now verified against the remote tree, never the push output."],
  ["Waiter loops matching themselves","A watcher spun for three hours doing nothing",
   "Waiters now key on an exact PID rather than a process-name pattern."]],
 widths=[1.7,2.3,2.6], align=["left","left","left"])

d.add_heading("4.4  Hardware", level=2)
table(d,"Table 8.",["Problem","How it surfaced","Resolution"],
 [["Thermal throttling","Compute per sample drifted 76 to 121 s at a fixed workload",
   "Confirmed by throughput drift, which needs no privileges; the battery sensor read 31 C throughout and was not informative about the SoC. Recovered to 74 s after cooling."],
  ["Battery drain under load","The machine discharged on AC while training",
   "A 35 W adapter supplies 34.8 W at its ceiling; sustained GPU load consumes it. Charging resumes when the load pauses."],
  ["System sleeping mid-run","One seed took 378.9 min instead of 37",
   "caffeinate's system-sleep assertion only holds on AC power. The machine slept once it was on battery."]],
 widths=[1.7,2.3,2.6], align=["left","left","left"])

d.add_heading("5  Results Retracted or Rejected", level=1)
for t in ["The IoU 0.1397 headline and the claim that our supervision beat human annotation, both withdrawn when the training cache was found to contain the evaluation split. Re-established later on a genuinely disjoint split.",
 "The conclusion that the spatial head was inferior to the global head, which was an artefact of selecting the epoch on the evaluation set.",
 "A free-space basis and a geometric basis appended to the prototype set. Each was required to exceed a random basis of equal dimensionality; the geometric basis beat neither the random control (p = 0.33) nor the plain head (p = 0.47). Both removed.",
 "The claim that additional data would not help insertion, drawn at roughly 500 insertion samples. With 1,828 it moved from 44 to 56 percent of its ceiling. The ceiling itself is unchanged; the surviving claim is the bound, not the prediction about data.",
 "A geometric proxy for the Stage 2 operating point, intended to avoid hours of diffusion. Correlation with measured net was 0.318 and the ranking was not preserved, so the sweep was paid for in full."]:
    p=d.add_paragraph(t, style="List Number"); p.paragraph_format.space_after=Pt(5)
    for r in p.runs: r.font.size=Pt(10.5)

d.add_heading("6  Measured Timings", level=1)
table(d,"Table 9. Wall times on the machine in Table 1.",
 ["Stage","Time"],
 [["Feature precompute, 8,807 rows","35.0 min"],
  ["Leakage audit, 55 shards","~9 min"],
  ["v2 training, per seed, on AC power","37-40 min"],
  ["v2 training, per seed, on battery (throttled)","44-53 min"],
  ["v1 training, per seed","~32 min"],
  ["Stage 2 inpainting, 60 samples x 4 arms","107 min"],
  ["One grid cell, 40 samples","13-18 min"],
  ["Full 20-cell grid","~6 h"]],
 widths=[3.6,1.4])

d.add_heading("7  Reproduction", level=1)
for c in ["python train/precompute_full.py --split train --batch 8",
          "python train/leakage_check.py",
          "python train/clean_eval_full.py --seeds 12 --configs v2,v1",
          "python train/clean_eval_full.py --seeds 3 --configs v2,v1 --curves",
          "python train/metrics_full.py",
          "python train/threshold_select.py",
          "python train/latency_v2.py",
          "python train/stage2.py --n 60 --steps 20",
          "./run_cells.sh          # the 20-cell grid, resumable",
          "python train/assemble_grid.py",
          "python train/plots_full.py && python train/curve_plots.py && python train/grid_figure.py"]:
    p=d.add_paragraph(); r=p.add_run(c); r.font.name="Consolas"; r.font.size=Pt(9.5)
    p.paragraph_format.space_after=Pt(1)
para(d,"")
para(d,"All 24 trained checkpoints are committed, so every evaluation above re-runs without retraining. The dataset, the feature cache and the diffusion weights are not committed; RESTORE.md carries the commands to fetch them. Repository: github.com/Amritha902/edit-region-prediction")
d.save("Review1_Execution_Record.docx"); print("wrote Review1_Execution_Record.docx")
