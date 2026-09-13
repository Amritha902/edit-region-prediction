import sys, json; sys.path.insert(0,".")
from mkdocx import *

rows=[r for r in json.load(open("/tmp/lit.json")) if int(r["Year"])>=2024]
# Four entries carried no URL in the matrix but have an official one that is
# deterministic from their identifier. RePlan and Follow-Your-Shape are
# conference papers with no arXiv id in the matrix; guessing a URL for them
# would be worse than leaving the citation as text.
URL_FILL={
 "US12561956":"https://patents.google.com/patent/US12561956",
 "US12462449B2":"https://patents.google.com/patent/US12462449B2",
 "Qwen-Image-Edit":"https://huggingface.co/Qwen/Qwen-Image-Edit",
 "FLUX.1 Kontext":"https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev",
}
for r in rows:
    if not (r["PDF link"] and str(r["PDF link"]).startswith("http")):
        for key,u in URL_FILL.items():
            if key in str(r["Paper"]): r["PDF link"]=u; break
CL=["B. Edit localization (CORE)","A. Instruction editors","E. Insertion & affordance",
    "F. Region-aware efficiency","H. Text-guided segmentation","C. Benchmarks & metrics",
    "D. Planning & reasoning","G. Structured scene representations","I. Surveys"]
NAME={"B. Edit localization (CORE)":"Edit localization","A. Instruction editors":"Instruction editors",
 "E. Insertion & affordance":"Insertion and affordance","F. Region-aware efficiency":"Region-aware efficiency",
 "H. Text-guided segmentation":"Text-guided segmentation","C. Benchmarks & metrics":"Benchmarks and metrics",
 "D. Planning & reasoning":"Planning and reasoning","G. Structured scene representations":"Structured scene representations",
 "I. Surveys":"Surveys"}
SYN={
"B. Edit localization (CORE)":"This is the cluster in which the present work sits, and it divides in two. One group reads a region out of a generator that is already running: Rethinking Where to Edit takes attention cues from the source and target streams, and WhereEdit matches changed prompt tokens and aggregates cross-attention. Both are training-free, and both inherit the limits of what attention happens to emphasise; WhereEdit states that it does not address addition into empty regions. The second group consumes a region produced elsewhere: MaskFlow incorporates the mask into the probability path of a flow-matching objective, and MADiff predicts a mask and then edits through it, though restricted to fashion and to garments already present. AdaptEdit is the only work that trains a predictor to ground the region from the instruction itself, which is the formulation adopted here. The opening it leaves is its supervision, not its architecture.",
"A. Instruction editors":"Nine works, with a uniform pattern: strong editing performance and no treatment of localization. Step1X-Edit parses the instruction into editing tokens and decodes with a diffusion transformer; Qwen-Image-Edit and FLUX.1 Kontext provide 12 to 32 billion parameter open-weight editors. In each, the decision of which pixels to change is internal and unscored. NEP makes the asymmetry explicit by conditioning on edit-region mask tokens as an input: it consumes a region and does not produce one.",
"E. Insertion & affordance":"Insertion is the case this project addresses, and the cluster shows the field has converged on solving it given a location. MADD denoises the image and the insertion mask jointly but requires a position prompt. Add-it places objects through weighted extended attention without training, so placement emerges instead of being predicted. The patent literature states the same requirement in legal terms: US12561956 requires an original image with a marked region, and US12462449B2 generates per-layer blending masks within a generator. Across research and patents, the location is an input.",
"F. Region-aware efficiency":"Three works exploit regions to reduce computation, and each assumes the region originates elsewhere. RegionE skips denoising in unchanged areas but infers the region from an initial denoising pass. SpotEdit skips stable regions by perceptual similarity. LazyDiffusion synthesises only the masked region from a compact global context and requires a user-drawn mask. The cluster establishes that a reusable per-image representation with an inexpensive per-instruction query is a recognised structure, which is the structure of the present architecture.",
"H. Text-guided segmentation":"Referring segmentation returns the region a phrase denotes. For an instruction such as “put a hat on the man” the phrase denotes the man, whereas the region requiring change is the space above him. The referring-segmentation survey makes this distinction explicit. We measured the consequence instead of assuming it: CLIPSeg attains 0.1849 IoU overall on this task but 0.054 on insertions. VespaSeg is a precedent in the opposite direction, demonstrating that lightweight text-conditioned segmentation is viable.",
"C. Benchmarks & metrics":"Four works, none of which scores a predicted region. LocateEdit-Bench scores localization of edits that have already occurred, which is post-hoc analysis of a finished edit, not an evaluation of a region predicted beforehand. GEdit-Bench and the region-and-semantics metric score the output image. The affordance benchmark measures plausibility, not predicted-region IoU. This absence explains how a 2026 paper built around a MaskPredictor reports no mask IoU.",
"D. Planning & reasoning":"These works decompose complex instructions prior to editing. X-Planner divides instructions into steps and RePlan adds region-aligned guidance through a multimodal planner. That planning is semantic: what to do, and in what order. It stops short of where, in pixels. Each also places a multi-billion parameter model in the loop for a decision the present head performs in 5.30 milliseconds.",
"G. Structured scene representations":"Six works construct persistent structure over a scene. I2E performs instance-level amodal decomposition with depth-aware ordering, and layer-wise memory stores latents across sequential edits. The common limitation is that they represent entities that exist; none renders empty space addressable, and none measures reuse across multiple instructions on a single image.",
"I. Surveys":"Three surveys frame the area. The 2026 instruction-editing survey confirms that localization is treated as a sub-component of editing, never as a first-class output with its own metric. The referring-segmentation survey supplies the semantic distinction relied upon here, and the affordance survey shows that affordance prediction grew out of robotics, not edit localization."}

d = base_doc()
titlepage(d,"Literature Survey",
 "Disentangling Localization from Editing in Instruction-Guided Image Manipulation",
 "A survey of forty-five works, 2024–2026, on predicting where an instruction requires an edit")
ent=[("1","Scope and Selection Criteria",1),("2","Classification of the Field",1)]
ent+=[(str(i+3),NAME[c],1) for i,c in enumerate(CL)]
ent+=[("12","Identified Research Gaps",1),("13","Proposed Mitigations and Outcomes",1),
      ("14","Quantitative Positioning",1),("15","Where the Cost Actually Falls",1),("16","What Others Optimise",1),("17","Base Paper",1),("18","References",1)]
toc(d,ent)

d.add_heading("1  Scope and Selection Criteria", level=1)
para(d,"We included work that bears on the question of where an instruction-guided edit should be applied: editors whose localization is implicit, methods that read or consume regions, insertion and affordance work, text-guided segmentation, benchmarks, and surveys framing the area. We also included patents that claim region generation, since the commercial position is material to an argument for novelty.")
para(d,"Every work is dated 2024 or later, a three-year window as of September 2026. Two foundational works fall outside this window and are cited as method components rather than as related work: FastSAM (2023), whose prototypes form the basis over which the present head predicts coefficients, and LISA (2023), which established reasoning segmentation. Excluding them from the survey preserves the stated window.")
para(d,"We scored each entry from 1 to 5, where 5 means a work that defines the problem, bounds it, or constitutes a direct comparison. Sixteen of the forty-five qualify. All links resolve and 42 PDFs are archived with the matrix.")
table(d,"Table 1. Distribution by publication year.",["Year","Works"],
 [["2024","6"],["2025","20"],["2026","19"]],widths=[1.4,1.0])
para(d,"Thirty-nine of forty-five works date from 2025 or 2026, indicating an actively worked area. This is simultaneously an argument for the relevance of the project and a constraint on any claim of novelty.")

d.add_heading("2  Classification of the Field", level=1)
para(d,"Classifying the works by what each does with the edit region, not by venue or method, gives a clearer account than a chronological reading. Three positions account for almost all of them.")
table(d,"Table 2. The forty-five works classified by treatment of the edit region.",
 ["Position","Works","Representative","Implication"],
 [["Consumes a region supplied to it","17","NEP, MaskFlow, LazyDiffusion","Demand established; supply assumed"],
  ["Reads a region from model internals","11","WhereEdit, RegionE","Unsupervised; blind to empty space"],
  ["Predicts a region from the instruction","2","AdaptEdit, MADiff","The formulation adopted here"],
  ["Scores or frames the problem","15","LocateEdit-Bench, surveys","None scores a predicted region"]],
 widths=[2.3,0.7,1.8,2.0],bold_rows=(2,))
para(d,"Of forty-five recent works on instruction-guided editing and its localization, two train a model to produce the edit region from language, and one of those is restricted to a single domain.")

for i,c in enumerate(CL):
    rs=sorted([r for r in rows if r["Cluster"]==c],key=lambda x:(-int(x["Rel. 1-5"]),-int(x["Year"])))
    d.add_heading(f"{i+3}  {NAME[c]}", level=1)
    para(d,SYN[c])
    t = table(d,f"Table {i+3}. {NAME[c]} — {len(rs)} works. Titles are links.",
     ["Work","Yr","What it improves","What is missing"],
     [["", str(r["Year"]),
       str(r["Parameters/metrics they improve"])[:95],
       str(r["What is MISSING (our opening)"])[:95]] for r in rs],
     widths=[1.9,0.4,2.2,2.3], align=["left","center","left","left"])
    for ri, r in enumerate(rs, start=1):
        link_cell(t.rows[ri].cells[0], str(r["Paper"])[:70], r["PDF link"], size=9)
d.save("Review1_Literature_Survey.docx"); print("survey part 1 done")

d.add_heading("12  Identified Research Gaps", level=1)
para(d,"The clusters converge on four openings. Each is stated with the evidence from the literature and with the corresponding measurement made in this project, since a gap asserted from reading alone is a weaker claim than one that has been reproduced.")
d.add_heading("12.1  Editing is solved given a region; producing one is not", level=2)
para(d,"Works on editing, efficiency and flow matching perform well when supplied with a region. NEP accepts the mask as an input, MaskFlow incorporates it into its objective, and LazyDiffusion requires a user-drawn mask. We tested the obvious substitute and it failed: selecting a FastSAM mask by CLIP similarity scores zero of seven across modification, removal and insertion, failing on granularity, on spatial language, and on insertion in principle.")
d.add_heading("12.2  The edit region is frequently not an object", level=2)
para(d,"The base paper states this against itself, citing geometry-changing edits with no localized source region. WhereEdit states that it does not address addition into empty regions. Referring segmentation cannot assist, since it returns the region a phrase denotes. Measured: CLIPSeg attains 0.1849 overall and 0.054 on insertion.")
d.add_heading("12.3  Supervision rather than architecture is the binding constraint", level=2)
para(d,"AdaptEdit supervises its MaskPredictor with MagicBrush human annotation and rendered text regions, explicitly not with pixel differences. We measured what that costs: across all 528 development turns the human masks cover 62.4 percent of the frame against a true changed area of 10.3 percent, a factor of 9.1, with precision 15.8 percent and a factor of 11.9 for insertions.")
d.add_heading("12.4  No benchmark scores a predicted edit region", level=2)
para(d,"LocateEdit-Bench scores edits that have already occurred. GEdit-Bench and the region-and-semantics metric score the output image. The affordance benchmark measures plausibility. There exists no evaluation on which an absent mask IoU would have been identified.")

d.add_heading("13  Proposed Mitigations and Outcomes", level=1)
para(d,"Each gap has a corresponding response in the implemented system. One of the proposed mitigations did not survive its own control and is reported here rather than omitted.")
table(d,"Table 12. Gap, response and measured outcome.",
 ["Gap","Response","Outcome"],
 [["Producing a region","Predict prototype coefficients from the instruction rather than selecting a detection, so that the mask is synthesised","IoU 0.2065, 12 seeds"],
  ["Region is not an object","A 20 × 20 spatial coefficient field, so that different locations use different prototype mixtures","Insert 0.1186 against 0.054"],
  ["Supervision","Derive the target by CIE76 colour difference in L*a*b* from authentic source and target pairs","Exceeds human masks by 0.0554"],
  ["No evaluation exists","Report mask IoU, precision, recall and F1 on a held-out split, then test downstream","Localization worth a factor of 3.3"],
  ["Empty space, explicitly","Append hand-designed free-space basis functions to the prototype set","Rejected, p = 0.33"]],
 widths=[1.5,3.3,1.8], align=["left","left","left"], bold_rows=(4,))
para(d,"The final row was the fourth planned mitigation. Any added basis is required to exceed a random basis of equal dimensionality; this one exceeded neither the random control (+0.0035, p = 0.33) nor the plain spatial head (−0.0030, p = 0.47), and was removed. The gap it addressed remains open.")

d.add_heading("14  Quantitative Positioning", level=1)
para(d,"Most works in this survey cannot be compared numerically with the present system: they optimise image-quality scores on benchmarks that do not expose a region, or they are training-free methods addressing a different task. Reporting a percentage against them would not be meaningful. Four reference points are directly comparable, having been evaluated on the same 503 held-out turns against the same ground truth.")
table(d,"Table 13. Measured against every comparable reference. The present system is the spatial head at 8,306 training samples, 12 seeds.",
 ["Reference","Reference IoU","Present work","Relative","Insertion"],
 [["CLIPSeg, 150.7 M parameters","0.1849","0.2065","+11.7%","0.054 → 0.1186 (+120%)"],
  ["MagicBrush human masks","0.1511","0.2065","+36.7%","0.120 → 0.1186 (−1%)"],
  ["Whole frame (trivial)","0.0958","0.2065","+115.6%","0.071 → 0.1186 (+67%)"],
  ["Random centred box (trivial)","0.0621","0.2065","+232.5%","0.033 → 0.1186 (+259%)"]],
 widths=[1.9,1.1,1.1,0.9,1.8], bold_rows=(0,))
para(d,"All differences are significant at p < 0.003 or better; against CLIPSeg, p = 5.2 × 10⁻⁷. On insertion the present system is level with human annotation and ahead of all other references. The cost is 4,693,633 trainable parameters against 150,747,746 for CLIPSeg, a factor of 32, at 5.30 milliseconds against 54.3 milliseconds per instruction, a factor of 10.3, resolution-matched at 352 pixels.")
para(d,"No entry in that table is AdaptEdit. It is the base paper and the only work sharing this formulation, yet it reports no mask IoU, and at 20.43 billion frozen parameters with approximately 5.0 billion trainable it cannot be executed to obtain one. This is the gap itself: the present work reimplements the mechanism at a scale that can be run and supplies the measurement that is absent.")

d.add_heading("15  Where the Cost Actually Falls", level=1)
para(d,"Accuracy is not the only axis on which this work can pay. Every method in the survey re-runs a large model for each instruction. Ours splits the work in two, and that split is worth stating in commercial terms because it decides whether a system like this is cheap enough to ship.")
table(d,"Table 15. Cost per instruction on one image. Measured on an M5, resolution-matched at 352 px.",
 ["System","Params run per instruction","Per instruction","On 100 instructions"],
 [["CLIPSeg","150.7 M","54.3 ms","5.43 s"],
  ["Ours (backbone amortised)","4.69 M","5.30 ms","0.56 s + 33.6 ms once"],
  ["Qwen-Image-Edit / FLUX class","12–32 B","seconds per edit","minutes"]],
 widths=[2.0,1.7,1.3,1.7])
para(d,"Editing is iterative. A user issues several instructions against the same photograph, and only the instruction changes. Because the backbone output depends on the image alone, we compute the 32 prototypes once and pay 5.30 ms for each instruction after that. CLIPSeg has no such split: every instruction costs a full 150.7 M forward pass. At a hundred instructions on one image the gap is 0.56 s against 5.43 s.")
para(d,"Two works in the survey confirm this is where the field sees value. RegionE skips denoising in unchanged areas and SpotEdit skips stable regions, both for speed. Neither produces the region from language; both have to infer it after generation starts. A predictor that supplies the region before generation is the missing input to exactly that class of optimisation, which is the practical case for the approach.")

d.add_heading("16  What Others Optimise, and What They Leave Unmeasured", level=1)
para(d,"Each cluster reports progress on a metric. None of those metrics is the mask. This table names the quantity each line of work improves and the quantity it never reports, alongside what we measured for the same thing.")
table(d,"Table 16. The parameter each line of work improves, against the one it leaves unmeasured.",
 ["Line of work","What they improve","Never reported","Ours"],
 [["Instruction editors (Step1X, Qwen, FLUX)","GEdit-Bench category scores, instruction following","Mask IoU of the region they edit","0.2065"],
  ["AdaptEdit (base paper)","GEdit-Bench: text_change 8.53, subject-add 8.28","Mask IoU — its MaskPredictor is never scored","0.2065"],
  ["Attention readers (WhereEdit, Rethinking Where)","Edit quality with no training","IoU against a true changed region; empty space excluded by design","insert 0.1186"],
  ["Region-aware efficiency (RegionE, SpotEdit)","Denoising steps skipped, wall-clock","Where the region comes from","5.30 ms/instruction"],
  ["Referring segmenters (CLIPSeg, LISA, VespaSeg)","RefCOCO-style IoU on denoted objects","IoU on edit regions, which are not the denoted object","+0.0216 over CLIPSeg"],
  ["Benchmarks (LocateEdit-Bench)","Localization of edits already made","Any score for a region predicted beforehand","first such score here"]],
 widths=[1.9,1.9,2.0,0.9], align=["left","left","left","left"])
para(d,"The pattern is consistent. Every line of work optimises a quantity downstream of the region and reports nothing about the region itself. That is the gap this project fills, and it is why our contribution is a measurement as much as a model.")

d.add_heading("17  Base Paper", level=1)
table(d,"Table 17. AdaptEdit — Edit Where You Mean: Region-Aware Adapter Injection for Mask-Free Local Image Editing. Cai et al., arXiv:2604.23763, 26 April 2026.",
 ["Aspect","Detail"],
 [["Problem addressed","Local image editing without requiring the user to draw a mask"],
  ["Method","Retrofits a frozen diffusion transformer with Block Adapters at each block, a Condition Encoder and a SpatialGate"],
  ["Component extended here","A thin MaskPredictor head, trained jointly, grounding the edit region from instruction and source image"],
  ["Its supervision","MagicBrush human-annotated masks and rendered text bounding regions, explicitly not pixel differences"],
  ["Loss","Region-aware loss weighting pixels within the predicted region"],
  ["Scale","20.43 B frozen transformer, 4.67 B adapters, 351 M condition encoder"],
  ["Stated limitation","Geometry-changing edits with no localized source region"],
  ["Why not extended directly","A diffusion transformer at 12 to 20 billion parameters cannot be executed or retrained on a 26 GB machine"],
  ["Relationship to present work","Same problem statement; different supervision, different weight class, and the mask IoU it does not report"]],
 widths=[1.9,4.7], align=["left","left"], bold_rows=(6,))

d.add_heading("18  References", level=1)
para(d,"All forty-five works, most recent first. Archived PDFs accompany the matrix in LITERATURE/pdfs/.")
for n,r in enumerate(sorted(rows,key=lambda x:(-int(x["Year"]),str(x["Paper"]))),1):
    p=d.add_paragraph()
    p.paragraph_format.space_after=Pt(3)
    p.paragraph_format.left_indent=Inches(0.35); p.paragraph_format.first_line_indent=Inches(-0.35)
    a=p.add_run(f"[{n}]  "); a.font.size=Pt(9.5)
    if r["PDF link"] and str(r["PDF link"]).startswith("http"):
        hyperlink(p, str(r["Paper"]), str(r["PDF link"]), size=9.5)
    else:
        a2=p.add_run(str(r["Paper"])); a2.font.size=Pt(9.5)
    vid = f" arXiv:{r['arXiv / ID']}." if r["arXiv / ID"] not in (None,"-") else ""
    b=p.add_run(f". {r['Venue']}, {r['Year']}.{vid}"); b.font.size=Pt(9.5)
d.save("Review1_Literature_Survey.docx"); print("survey complete")
