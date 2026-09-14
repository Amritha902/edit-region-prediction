import sys; sys.path.insert(0,".")
from mkdocx import *

d = base_doc()
for _ in range(2): d.add_paragraph()
para(d,"Demonstration run-sheet", size=20, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space=6)
para(d,"What to run, in what order, and what to say", size=12.5, italic=True,
     align=WD_ALIGN_PARAGRAPH.CENTER, space=8, color=MUTE)
para(d,"5 of the 10 DA2 marks · from 20 September", size=11, italic=True,
     align=WD_ALIGN_PARAGRAPH.CENTER, space=20, color=MUTE)

d.add_heading("Before she arrives", level=1)
for t in ["Kill every background job: pkill -f stage2_sd_sweep. Under GPU contention the demo "
          "has reported 10x the benchmark timings, which makes the model look slower than the "
          "baseline it beats.",
          "Plug in. On battery macOS throttles and the numbers drift.",
          "Run ./run_demo.sh once beforehand so the models are in the page cache — the first "
          "load takes ~15 s and the first MPS call compiles Metal graphs.",
          "Have the repo open: github.com/Amritha902/edit-region-prediction"]:
    p=d.add_paragraph(t, style="List Number"); p.paragraph_format.space_after=Pt(4)
    for r in p.runs: r.font.size=Pt(10.5)

d.add_heading("The run, in order", level=1)
table(d,"Roughly 8 minutes of running, plus questions.",
 ["#","Command","What to say while it runs"],
 [["1","./run_demo.sh (case 1)",
   "The mask lands on the paper on the wall and covers 9.5% of the frame. Point at the "
   "per-image versus per-instruction split in the output — that split is the architecture."],
  ["2","(case 2)",
   "Larger region, background edit. Shows the mask scaling with the instruction rather than "
   "being a fixed blob."],
  ["3","(case 3, insertion)",
   "This is the hard case and the region will be imperfect. Say so first, before she does: "
   "the prototype basis caps ANY predictor at 0.2119 here, and we reach 56% of that."],
  ["4","(case 4, --edit)",
   "End to end: predict the region, then inpaint inside it. About 40 s on first load."],
  ["5","python train/metrics_full.py",
   "Optional. Recomputes precision, recall and F1 from the committed checkpoints — nothing "
   "is hard-coded."]],
 widths=[0.4,1.9,4.3], align=["center","left","left"])

d.add_heading("The numbers to have ready", level=1)
table(d,"Quote these, do not recompute them live.",
 ["Quantity","Value","Against"],
 [["Dev IoU, 12 seeds","0.2065 ± 0.0072","CLIPSeg 0.1849, human masks 0.1511"],
  ["Significance vs CLIPSeg","p = 5.2 x 10^-7","Cohen's d = 2.50 vs the global head"],
  ["Precision / Recall / F1","0.3126 / 0.3878 / 0.2976","at threshold 0.5"],
  ["Per-instruction latency","5.30 ms","CLIPSeg 54.3 ms — 10.3x"],
  ["Trainable parameters","4,693,633","CLIPSeg 150,747,746 — 32x fewer"],
  ["Training set","8,306 turns","0 images shared with the held-out split"],
  ["Localization value, Stage 2","+53.0% vs +9.1%","5.8x over whole-frame editing"],
  ["Insertion ceiling","0.2119","we reach 56% of it"]],
 widths=[1.9,1.5,3.2], align=["left","left","left"])

d.add_heading("Questions she is likely to ask", level=1)
for q,a in [
 ("Why not compare directly against the base paper?",
  "It reports no mask IoU at all, and it is 20.43 billion frozen parameters plus about 5 "
  "billion trainable — it cannot be run here. That absence is the gap we fill: we "
  "reimplement its mechanism at a scale we can run and supply the measurement it omits."),
 ("Is 0.2065 a good number?",
  "Against the right references, yes. It beats a 150.7 million parameter pretrained "
  "segmenter and the dataset's own human annotations. Against the ceiling we are at 56%, "
  "and we say so rather than hiding it."),
 ("Why is insertion still weak?",
  "It is a representational bound, not a training failure. We fitted coefficients directly "
  "to ground truth by least squares — that bounds any predictor over this basis at 0.2119 "
  "for insertion against 0.4325 for removal. The prototypes were trained to segment objects, "
  "and objects are not empty space."),
 ("Does it always work?",
  "No. Mask coverage varies from 9.5% of the frame on a good image to 56.3% on a dense "
  "close-up where it over-predicts badly. The per-kind numbers are in the report."),
 ("How do I know the results are not overfitted or leaked?",
  "Three things. The checkpoint epoch is chosen on a held-out slice of train, never on the "
  "evaluation set. The evaluation set is touched once per seed. And every image was hashed "
  "on both sides — 794 dev images, 13,317 train images, zero overlap."),
 ("What did not work?",
  "Two proposed basis extensions failed against a random control of equal size and were "
  "dropped. An early headline was retracted after we found dev had been merged into train. "
  "Both are written up in the repository.")]:
    d.add_heading(q, level=2)
    p=d.add_paragraph(a); p.paragraph_format.space_after=Pt(8)
    for r in p.runs: r.font.size=Pt(10.5)

d.add_heading("If something breaks", level=1)
table(d,"",["Symptom","Cause","Fix"],
 [["Timings ~10x too slow","another job on the GPU","the demo prints a warning; kill it and rerun"],
  ["~1 s per instruction","first MPS call compiling","run once beforehand to warm up"],
  ["Mask covers most of the frame","dense close-up image","use the demo_images provided"],
  ["Model file not found","wrong directory","run from instruct-seg-edit-mac/"]],
 widths=[1.9,1.7,2.9], align=["left","left","left"])
d.save("DA2_demo_runsheet.docx"); print("wrote DA2_demo_runsheet.docx")
