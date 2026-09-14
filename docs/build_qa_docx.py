import sys; sys.path.insert(0,".")
from mkdocx import *

QA = [
("The problem and the motivation", [
 ("What is this project actually doing?",
  "Predicting which pixels an editing instruction requires to change. Editors are told what to change, never where, so they denoise the whole frame and the region stays implicit. We predict it explicitly and gate the editor to it."),
 ("Why does that matter?",
  "Two failures follow from leaving it implicit: the intended edit is applied weakly, and unrelated parts of the image drift. We measured the second half — editing inside the correct region is worth 5.8 times, +53.0% net against +9.1% for whole-frame editing."),
 ("Why is insertion the hard case?",
  "“Put a hat on the dog” names an object that is not in the picture. There is no source region to point at. Every method that works by referring to visible content fails on it, which is why referring segmenters do worst there — CLIPSeg gets 0.054 on insertion against 0.1849 overall."),
 ("Who else has tried this?",
  "Of 45 works from 2024 to 2026, seventeen consume a region given to them, eleven read one out of a running generator, fifteen score something downstream — and two predict one from language. One of those two is restricted to fashion images."),
]),
("The base paper", [
 ("Why this base paper?",
  "AdaptEdit is the only 2026 work that trains a predictor to ground the edit region from the instruction, which is exactly our formulation. Its limitations section also names our gap in its own words."),
 ("Why not just extend it?",
  "20.43 billion frozen parameters plus roughly 5 billion trainable. It cannot be run or retrained on a 26 GB laptop."),
 ("Then how do you compare against it?",
  "We do not, and cannot. It reports no mask IoU anywhere — its MaskPredictor is evaluated only through downstream image quality. That absence is the gap: we reimplement the mechanism at a scale we can run and supply the measurement it omits."),
 ("Is that not convenient for you?",
  "It would be if we had quietly skipped the comparison. We state it explicitly in the report and in the survey, and the reason no benchmark caught it either: LocateEdit-Bench scores edits that have already happened, not regions predicted beforehand."),
]),
("The method", [
 ("What is a prototype?",
  "YOLACT-style segmentation does not output masks directly. It outputs 32 grayscale spatial patterns at 160x160, and a mask is a weighted sum of them through a sigmoid. The weights are the coefficients."),
 ("So what did you change?",
  "Normally a detection supplies the coefficients, which is why FastSAM can only return objects it detected. The instruction supplies them instead. The mask is synthesised rather than selected, which is what makes empty space reachable in principle."),
 ("Why freeze the backbone?",
  "Its output depends only on the image, not the instruction. Freezing it means computing it once per photograph and reusing it for every instruction — 33.6 ms paid once, then 5.30 ms each. That split is where the latency advantage comes from."),
 ("What does freezing cost you?",
  "Accuracy. It is a deliberate trade and we say so: it buys 32 times fewer trainable parameters and a factor of 10.3 in per-instruction latency, and it caps how accurate the system can get."),
 ("Why a spatial field instead of one coefficient vector?",
  "One vector forces every location to use the same prototype mixture, which suits an object but not a region defined by where something should go. It is not a capacity argument — the global head saturates at 1.4 million parameters."),
 ("Did the spatial field actually help?",
  "+0.0158 IoU, p = 5.4 x 10^-6, Cohen's d 2.50. And it helps where the hypothesis says it should: insertion +22% and modification +8%, while removal is slightly worse, because removal targets an object that already exists."),
]),
("Supervision", [
 ("Why not use the dataset's own masks?",
  "We measured them across all 528 dev turns: they cover 62.4% of the frame against a true changed area of 10.3%. That is 9.1 times too large, precision 15.8%, and worst on insertions at 11.9 times. They are regions of interest, not edit masks."),
 ("So what do you train on?",
  "The real before and after images a human made. We compute CIE76 colour difference in L*a*b* per pixel, threshold at 12, clean up morphologically. Where a hat actually landed is ground truth for where a hat should go."),
 ("Why threshold 12?",
  "From a noise-robustness sweep. It holds at Gaussian noise sigma = 8, where lower thresholds start reporting noise as change."),
 ("Does that supervision actually beat the human masks?",
  "Yes, by 0.0554 IoU, p = 2.5 x 10^-11. That claim was retracted once when we found a contaminated cache, then re-established on a genuinely disjoint split."),
]),
("Results and validity", [
 ("Is 0.2065 a good number?",
  "Against the right references, yes: it beats CLIPSeg at 150.7 million parameters and the dataset's own human annotations. Against the ceiling we are at 56%, and we report that too."),
 ("How do I know it is not overfitted?",
  "The checkpoint epoch is chosen on a held-out slice of train, never on the evaluation set. The evaluation set is touched once per seed. Twelve seeds, all four random generators seeded."),
 ("How do I know there is no leakage?",
  "The dataset is multi-turn, so one photograph yields several edits — a split by turn would put the same photo on both sides. We hashed every source and target image on both sides: 794 distinct dev images, 13,317 train, zero overlap. Run before the results were trusted, not after."),
 ("Why twelve seeds?",
  "Because two identical commands once gave 0.1215 and 0.1073. The data permutation was seeded and the weight initialisation was not. Single-run numbers on this system are not trustworthy."),
 ("Why is the threshold 0.5 when the sweep says 0.3 is better?",
  "Choosing 0.3 because the evaluation set says so would be test-set tuning, which already inverted one of our conclusions. Selecting the threshold on a train-validation slice gives 0.2185. We quote the fixed 0.5 as the headline because CLIPSeg is at its default operating point, so the tuned comparison would tilt toward us."),
]),
("Limitations", [
 ("Why is insertion still weak?",
  "It is a representational bound, not a training failure. We fitted coefficients directly to the ground truth by least squares — that caps any predictor over this basis at 0.2119 for insertion against 0.4325 for removal. FastSAM's prototypes were trained to segment objects, and objects are not empty space."),
 ("Does it always work?",
  "No. Mask coverage varies from 9.5% of the frame on a clean case to 56.3% on a dense close-up where it over-predicts badly. The per-kind numbers are in the report."),
 ("What did you try that failed?",
  "Two extensions to the prototype basis, free-space and geometric. Our rule is that any added basis must beat a random basis of equal size; the geometric one beat neither the random control (p = 0.33) nor the plain head (p = 0.47). Both dropped. We also tried to predict the Stage 2 operating point from mask geometry to avoid hours of compute — it correlated 0.318 and was abandoned."),
 ("What would you do next?",
  "Attack the basis, not the head. The ceiling is a property of the 32 prototypes, so learning a small number of extra prototypes supervised on insertion regions — required to beat a random basis of equal size — is the direct test."),
]),
("Novelty", [
 ("What is genuinely new here?",
  "Three things. Predicting the edit region from the instruction is nearly vacant in the literature — two of 45 works. Deriving supervision from pixel differences rather than human annotation appears unattempted in this setting, and we measured that it beats the human annotation the field trains on. And reporting a mask IoU at all is novel by absence."),
 ("What is not new?",
  "The mechanism. FiLM modulation and a BCE plus Dice objective are AdaptEdit's design choices as well as ours. We reached them independently, but that is not a claim. A lightweight text-conditioned segmenter is also precedented by VespaSeg."),
 ("Is this patentable?",
  "The supervision method is the strongest candidate — recovering the edit region from a before and after pair rather than from annotation. The architecture is not: too much of it is shared with the base paper and with standard segmentation practice."),
]),
]

d = base_doc()
titlepage(d,"Review Questions",
 "Disentangling Localization from Editing in Instruction-Guided Image Manipulation",
 "Anticipated questions, with the measured answer to each")
toc(d,[(str(i+1),sec,1) for i,(sec,_) in enumerate(QA)])
para(d,"Every answer here rests on a number from our own experiments. Where the "
       "honest answer is a limitation, it is given as one — those are the answers "
       "that hold up under follow-up.", size=10.5, italic=True, space=16, color=MUTE)

for i,(sec,items) in enumerate(QA, start=1):
    d.add_heading(f"{i}  {sec}", level=1)
    for q,a in items:
        p=d.add_paragraph(); r=p.add_run(q); r.font.size=Pt(11); r.bold=True
        p.paragraph_format.space_after=Pt(2); p.paragraph_format.space_before=Pt(9)
        p2=d.add_paragraph(); r2=p2.add_run(a); r2.font.size=Pt(10.5)
        p2.paragraph_format.left_indent=Inches(0.25); p2.paragraph_format.space_after=Pt(6)
        p2.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
d.save("Review1_Questions_And_Answers.docx")
print("wrote Review1_Questions_And_Answers.docx")
print(f"  {sum(len(x[1]) for x in QA)} questions across {len(QA)} sections")
