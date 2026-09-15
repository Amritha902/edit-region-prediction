"""Build DA2_Report.docx — Introduction, Literature Survey, Methodology.

Unlike build_da2_skeleton.py this writes the prose as well as the structure, so
the output is a finished document rather than a form to fill in. Every number
traces to a committed result under ../apple-silicon/, and the reference list is
read straight from the literature matrix rather than from a temporary file.

    python build_da2_report.py
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mkdocx import *
import openpyxl

HERE = Path(__file__).resolve().parent
FIG = HERE / "DA2_figures"
MATRIX = HERE.parent / "literature" / "Literature_Survey_Matrix.xlsx"

# Four entries carry no URL in the matrix but have a deterministic official one.
# RePlan and Follow-Your-Shape have no arXiv identifier recorded and are cited as
# text — a guessed URL would be worse than none.
URL_FILL = {
    "US12561956": "https://patents.google.com/patent/US12561956",
    "US12462449B2": "https://patents.google.com/patent/US12462449B2",
    "Qwen-Image-Edit": "https://huggingface.co/Qwen/Qwen-Image-Edit",
    "FLUX.1 Kontext": "https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev",
}


def load_matrix():
    wb = openpyxl.load_workbook(MATRIX)
    ws = wb["Literature Matrix"]
    hdr = [c.value for c in ws[1]]
    rows = []
    for r in ws.iter_rows(min_row=2):
        d = {hdr[i]: (c.value if c.value is not None else "")
             for i, c in enumerate(r) if i < len(hdr)}
        if not d.get("Paper"):
            continue
        if not str(d["Year"]).strip().isdigit() or int(d["Year"]) < 2024:
            continue
        if not str(d["PDF link"]).startswith("http"):
            for key, u in URL_FILL.items():
                if key in str(d["Paper"]):
                    d["PDF link"] = u
                    break
        rows.append(d)
    return rows


def titlepage_da2(d):
    for _ in range(3):
        d.add_paragraph()
    para(d, "DA2 Submission", size=11, italic=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, space=26, color=MUTE)
    p = d.add_paragraph()
    r = p.add_run("Disentangling Localization from Editing\nin Instruction-Guided Image Manipulation")
    r.font.size = Pt(21); r.bold = True; r.font.name = "Cambria"
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    para(d, "Introduction · Literature Survey · Methodology", size=12.5, italic=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, space=40, color=MUTE)
    for line, sz, it in (("Amritha S    23BEC1368", 12, False),
                         ("Yugeshwaran P    23BEC1404", 12, False),
                         ("", 6, False),
                         ("Guide: Dr. Saranya M (54783)", 11, True),
                         ("School of Electronics Engineering", 11, True),
                         ("Vellore Institute of Technology, Chennai", 11, True),
                         ("", 6, False),
                         ("Foundations of Data Science — DA2", 11, False),
                         ("15 September 2026", 11, False)):
        para(d, line, size=sz, italic=it, align=WD_ALIGN_PARAGRAPH.CENTER, space=3)
    d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def quote(d, text, size=10.5):
    q = d.add_paragraph()
    r = q.add_run(text); r.font.size = Pt(size)
    q.paragraph_format.left_indent = Inches(0.4)
    q.paragraph_format.right_indent = Inches(0.3)
    q.paragraph_format.space_after = Pt(12)
    q.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return q


def numbered(d, items, size=11):
    for i, t in enumerate(items, 1):
        p = d.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.45)
        p.paragraph_format.first_line_indent = Inches(-0.28)
        p.paragraph_format.space_after = Pt(5)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p.add_run(f"{i}.   {t}")
        r.font.size = Pt(size)


rows = load_matrix()
refs = sorted(rows, key=lambda x: (-int(x["Year"]), str(x["Paper"])))

# Short tokens used in the prose as [[token]], resolved to the reference number
# at build time so the two can never disagree. Each value is a substring that
# must match exactly one paper title in the matrix.
CITE_KEYS = {
    "adaptedit": "AdaptEdit", "step1x": "Step1X-Edit:", "qwen": "Qwen-Image-Edit",
    "flux": "FLUX.1 Kontext", "nep": "NEP:", "rethinking": "Rethinking Where to Edit",
    "whereedit": "WhereEdit:", "maskflow": "MaskFlow:", "madiff": "MADiff:",
    "madd": "MADD:", "addit": "Add-it:", "pat956": "US12561956", "pat449": "US12462449B2",
    "regione": "RegionE:", "spotedit": "SpotEdit:", "lazy": "LazyDiffusion:",
    "refsurvey": "Multimodal Referring Segmentation", "vespaseg": "VespaSeg:",
    "locatebench": "LocateEdit-Bench", "gedit": "GEdit-Bench", "prefmetric": "Balancing Preservation",
    "affbench": "Adding Affordance Benchmark", "xplanner": "X-Planner:", "replan": "RePlan:",
    "i2e": "I2E:", "layermem": "Layer-wise Memory", "editsurvey": "Instruction-based Image Editing",
    "editmgt": "EditMGT:", "var": "Visual Autoregressive Modeling", "gide": "GIDE:",
    "iep": "Image Editing As Programs", "cannyedit": "CannyEdit:", "ric": "Region in Context",
    "geom": "Training-free Geometric", "anchorseg": "AnchorSeg:", "setcon": "SetCon:",
    "dgseg": "DGSeg:", "rsagent": "RSAgent:", "affsurvey": "Visual Affordance Prediction",
    "ffse": "Free-Form Scene Editor", "fys": "Follow-Your-Shape", "muse": "MUSE:",
    "editssc": "EditSSC:", "cfseg": "Counterfactual Segmentation", "segllm": "SegLLM:",
}
CITE = {}
for tok, needle in CITE_KEYS.items():
    hits = [i for i, r in enumerate(refs, 1) if needle in str(r["Paper"])]
    if len(hits) != 1:
        raise SystemExit(f"citation token {tok!r} ({needle!r}) matched {len(hits)} papers")
    CITE[tok] = hits[0]


def cite(text):
    """Expand [[token]] and [[a,b]] into reference numbers."""
    import re as _re
    def sub(m):
        nums = [CITE[t.strip()] for t in m.group(1).split(",")]
        return "[" + ", ".join(str(n) for n in sorted(nums)) + "]"
    return _re.sub(r"\[\[([a-z0-9, ]+)\]\]", sub, text)


def cpara(d, text, **kw):
    return para(d, cite(text), **kw)


d = base_doc()
titlepage_da2(d)

toc(d, [
    ("1", "Introduction", 1),
    ("1.1", "Background", 2),
    ("1.2", "Motivation", 2),
    ("1.3", "Base Paper and the Identified Gap", 2),
    ("1.4", "Objectives", 2),
    ("2", "Literature Survey", 1),
    ("2.1", "Scope and Selection", 2),
    ("2.2", "Classification of the Field", 2),
    ("2.3", "Instruction-Following Editors", 2),
    ("2.4", "Edit Localization", 2),
    ("2.5", "Insertion and Affordance", 2),
    ("2.6", "Region-Aware Efficiency", 2),
    ("2.7", "Text-Guided Segmentation", 2),
    ("2.8", "Benchmarks and Metrics", 2),
    ("2.9", "Planning, Scene Representations and Surveys", 2),
    ("2.10", "Identified Research Gaps", 2),
    ("2.11", "Position of This Work", 2),
    ("3", "Methodology", 1),
    ("3.1", "Overview", 2),
    ("3.2", "Frozen Feature Extraction", 2),
    ("3.3", "Coefficient Prediction", 2),
    ("3.4", "Supervision", 2),
    ("3.5", "Dataset and Split Integrity", 2),
    ("3.6", "Training Protocol", 2),
    ("3.7", "Evaluation", 2),
    ("", "References", 1),
])

# ------------------------------------------------------------------ 1
d.add_heading("1  Introduction", level=1)

d.add_heading("1.1  Background", level=2)
cpara(d, "Instruction-guided image editing asks a model to change a photograph according to a "
        "sentence: make the jacket blue, take the poles out of his hands, put a hat on the dog. "
        "The systems that do this well — InstructPix2Pix and, more recently, Qwen-Image-Edit [[qwen]] and "
        "FLUX.1 Kontext [[flux]] — denoise the entire frame while conditioning on the instruction. The "
        "instruction tells such a model what to change. Nothing tells it where.")
para(d, "Two failures follow from that omission, and they pull in opposite directions. The "
        "intended change can come out weak, because the model is spreading its effort over the "
        "whole image rather than concentrating it. At the same time, regions the instruction "
        "never mentioned drift in colour and texture, because nothing holds them fixed. Neither "
        "failure is recoverable after the fact: the spatial decision was made implicitly inside a "
        "denoising network, so there is nothing for a user to inspect and nothing to correct.")
para(d, "The alternative is to make that decision explicit. Predict the region the instruction "
        "requires, then confine the editor to it. This splits a single opaque task into two that "
        "can be built, scored and corrected separately: localization, and then editing.")

d.add_heading("1.2  Motivation", level=2)
para(d, "The case that makes the split necessary rather than merely tidy is insertion. Consider "
        "“put a hat on the dog” applied to a photograph of a bare-headed dog. The instruction "
        "names an object that is not in the picture. There is no source region to point at, "
        "because the region that must change is the empty space above the animal's head.")
para(d, "This breaks every method that works by referring to visible content. Referring-expression "
        "segmentation — CLIPSeg, LAVT, Grounded-SAM — returns the region a phrase denotes, and the "
        "phrase here denotes the dog. Returning the dog is a correct answer to the question those "
        "models were trained on and a wrong answer to the question editing asks. We verified that "
        "this is a structural limitation rather than a tuning problem before building anything: a "
        "pipeline that segments an image with FastSAM and then selects among the candidate masks "
        "by CLIP similarity scored zero correct on seven probes spanning modification, removal and "
        "insertion, and failed in three distinct ways. It could not reach a part of an object when "
        "only whole objects were proposed; it could not resolve “the cat on the left”, where the "
        "two leading candidates tied at a margin of exactly zero because each crop is scored in "
        "isolation; and on the three insertions it could not reach the answer even in principle, "
        "since no candidate mask covers empty snow or the space above a cat's neck.")
if (FIG / "fig1_baseline_failure.png").exists():
    figure(d, str(FIG / "fig1_baseline_failure.png"),
           "Figure 1. The selection baseline on seven probes. Candidate masks come from FastSAM at "
           "conf = 0.05 and are chosen by CLIP similarity to the instruction; the selected region is "
           "tinted. Panels 1 to 3 return the identical whole-person mask for three different "
           "instructions, panels 5 and 6 cannot separate the two cats, and panels 3, 4 and 6 ask for "
           "regions that contain nothing and therefore appear in no candidate set.", width=6.2)

d.add_heading("1.3  Base Paper and the Identified Gap", level=2)
cpara(d, "AdaptEdit [[adaptedit]] (arXiv:2604.23763, April 2026) is the work this project positions itself "
        "against, because it is the only recent work that trains a predictor to ground the edit "
        "region from the instruction — the same formulation adopted here. It injects region-aware "
        "adapters into a frozen diffusion transformer and trains a MaskPredictor head jointly with "
        "them, which removes the requirement that a user supply a mask at deployment.")
para(d, "Two facts about it define the opening. The first is scale: 20.43 billion frozen parameters, "
        "4.67 billion trainable adapters and a 351 million parameter condition encoder, trained on "
        "multiple GPUs. That cannot be reproduced on a 26 GB laptop, so a replication was never the "
        "plan. The second matters more. The paper reports category scores on GEdit-Bench — "
        "text_change 8.53, subject-add 8.28 — but never scores its MaskPredictor directly. There is "
        "no mask IoU anywhere in it. A component is introduced, argued for, and never measured on "
        "the quantity it exists to produce.")
para(d, "Its own limitations section then names the case that the measurement would have exposed:")
quote(d, "“Geometry-changing edits with no localized source region (‘put a hat on the dog’ where "
         "the dog has no hat): the GT mask covers the new content’s location but there is no clean "
         "source signal to ‘protect’ outside it.” — Cai et al., arXiv:2604.23763")
para(d, "We think the two observations are connected, and that the connection is a data-science "
        "problem rather than an architectural one. AdaptEdit takes its masks from MagicBrush's "
        "human annotations and from rendered text boxes, explicitly not from pixel differences. "
        "Measured against the region that actually changes between a source and target pair, those "
        "annotations cover 62.4 per cent of the frame where 10.3 per cent changed — they are "
        "roughly nine times too large, and worst of all on insertions. A predictor trained on them "
        "learns to emit regions an order of magnitude too big, which is precisely the behaviour the "
        "quoted limitation describes. No benchmark scores a predicted edit region, so nothing in "
        "the literature forces this to surface.")

d.add_heading("1.4  Objectives", level=2)
numbered(d, [
    "Establish whether an off-the-shelf segmentation-and-retrieval pipeline is sufficient for "
    "edit-region prediction, and characterise how it fails when it is not.",
    "Construct supervision for the edit region that does not depend on human region annotation, "
    "and quantify how far the annotations it replaces are from the region that actually changes.",
    "Train an instruction-conditioned predictor over frozen features within the compute budget of "
    "a single laptop.",
    "Evaluate the predicted region directly, by intersection over union against the changed "
    "region — the measurement the base paper omits.",
    "Determine whether predicting the region improves the resulting edit, and by how much, rather "
    "than assuming that a better mask metric implies a better edit.",
    "Establish the upper bound that the chosen representation imposes, and report the fraction of "
    "it the trained model reaches.",
])

# ------------------------------------------------------------------ 2
d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
d.add_heading("2  Literature Survey", level=1)

d.add_heading("2.1  Scope and Selection", level=2)
para(d, "We surveyed forty-five works bearing on where an instruction-guided edit should be "
        "applied: editors whose localization is implicit, methods that read or consume a region, "
        "insertion and affordance work, text-guided segmentation, benchmarks, and surveys framing "
        "the area. Patents that claim region generation are included, because the commercial "
        "position is material to any argument about novelty. Every entry is dated 2024 or later. "
        "Two older works are cited as method components rather than as related work — FastSAM "
        "(2023), whose prototypes form the basis this project predicts coefficients over, and LISA "
        "(2023), which established reasoning segmentation — so that the three-year window holds. "
        "CLIPSeg (2022) is treated the same way: it predates the window and appears throughout as a "
        "measured baseline rather than as a surveyed work.")
table(d, "Table 1. Works surveyed, by year of publication.", ["Year", "Works"],
      [["2024", "6"], ["2025", "20"], ["2026", "19"], ["Total", "45"]],
      widths=[1.4, 1.0], bold_rows=(3,))
para(d, "Thirty-nine of the forty-five appeared in 2025 or 2026. That is an argument for the "
        "relevance of the problem and, equally, a constraint on how strongly novelty can be "
        "claimed in an area moving this quickly.")

d.add_heading("2.2  Classification of the Field", level=2)
para(d, "Sorting these works by date produces a list. Sorting them by what each one does with the "
        "edit region produces an argument, because the same asymmetry appears in every cluster: "
        "the region is treated as something a method receives, not something a method has to "
        "produce.")
table(d, "Table 2. The forty-five works by their treatment of the edit region.",
      ["Posture toward the edit region", "Works", "Representative work"],
      [["Consumes a region supplied to it", "17", "NEP, MaskFlow, LazyDiffusion"],
       ["Reads a region out of a running generator", "11", "WhereEdit, RegionE"],
       ["Predicts a region from the instruction", "2", "AdaptEdit, MADiff"],
       ["Scores or frames the problem", "15", "LocateEdit-Bench, surveys"]],
      widths=[2.7, 0.8, 2.5], bold_rows=(2,), align=["left", "right", "left"])
cpara(d, "Of forty-five recent works, two produce the region from language. One of those two, "
        "MADiff [[madiff]], is restricted to fashion imagery and to garments already present in the "
        "photograph, which leaves the general problem to a single paper. Demand for edit regions "
        "is established across seventeen works; supply is assumed by almost all of them.")

d.add_heading("2.3  Instruction-Following Editors", level=2)
cpara(d, "Nine works, with a consistent pattern: strong editing, silence on localization. "
        "Step1X-Edit [[step1x]] parses an instruction into editing tokens and decodes with a diffusion "
        "transformer; Qwen-Image-Edit [[qwen]] and FLUX.1 Kontext [[flux]] supply open-weight editors "
        "between twelve and thirty-two billion parameters. In each, the choice of which pixels to change is "
        "internal and unscored. NEP [[nep]] makes the asymmetry explicit rather than hiding it, by "
        "conditioning on edit-region mask tokens as an input: it consumes a region and does not "
        "produce one. These are the systems our Stage 2 would gate, not systems we compete with. The "
         "rest of the cluster varies the generator rather than the question — a masked generative "
         "transformer [[editmgt]], autoregressive decoding [[var]], a diffusion language model "
         "[[gide]], a program-synthesis formulation [[iep]] and Canny-guided dual prompting "
         "[[cannyedit]] — and none of them reports where its edit went.")

d.add_heading("2.4  Edit Localization", level=2)
cpara(d, "Seven works, and the cluster in which this project sits. It divides cleanly. One group "
        "reads a region out of a generator that is already running: Rethinking Where to Edit "
        "[[rethinking]] partitions tokens using attention cues from the source and target streams, and "
        "WhereEdit [[whereedit]] "
        "matches changed prompt tokens and aggregates cross-attention into a mask. Both are "
        "training-free, and both inherit whatever attention happens to emphasise; WhereEdit states "
        "outright that it does not address addition into empty regions, and keeping only the "
        "largest connected component breaks on repeated instances. The second group consumes a "
        "region produced elsewhere — MaskFlow [[maskflow]] builds the mask into the probability path of "
        "a flow-matching objective — or predicts one in a narrow domain, as MADiff [[madiff]] does for "
        "garments. "
        "AdaptEdit [[adaptedit]] alone trains a general predictor from the instruction. The opening it leaves is "
        "its supervision, not its architecture. Two further works complete the cluster and both "
        "illustrate the same boundary: Region in Context [[ric]] reasons over a region and its "
        "surroundings but stays tied to existing content, and training-free geometric editing "
        "[[geom]] moves and resizes regions the user has already specified.")

d.add_heading("2.5  Insertion and Affordance", level=2)
cpara(d, "Four works, covering the case this project is about, and all four show the field solving "
        "insertion given a location. MADD [[madd]] denoises the image and the insertion mask jointly but "
        "requires a position prompt. Add-it [[addit]] places objects through weighted extended attention "
        "without training, so placement emerges from the generator rather than being predicted. The "
        "patent literature states the same requirement in legal language: US12561956 [[pat956]] claims "
        "affordance-based reposing of “an original image with a marked region”, and US12462449B2 [[pat449]] "
        "generates per-layer blending masks inside a generator. Across research and patents alike, "
        "the location is an input.")

d.add_heading("2.6  Region-Aware Efficiency", level=2)
cpara(d, "Three works exploit a region to save computation, and each assumes it arrives from "
        "somewhere else. RegionE [[regione]] skips denoising in unchanged areas but infers the region "
        "from an initial denoising pass. SpotEdit [[spotedit]] skips stable regions by perceptual "
        "similarity. LazyDiffusion [[lazy]] synthesises only the masked region from a compact global context and needs a "
        "user-drawn mask. The cluster is useful here for a reason unrelated to its claims: it "
        "establishes that splitting a system into a reusable per-image representation and a cheap "
        "per-instruction query is a recognised structure, which is exactly the structure of the "
        "architecture in Section 3.")

d.add_heading("2.7  Text-Guided Segmentation", level=2)
cpara(d, "Five works, and the closest available substitute for what this project predicts — which "
        "is why it matters that the semantics are wrong. Referring segmentation returns the region "
        "a phrase denotes; the referring-segmentation survey [[refsurvey]] makes the distinction "
        "explicit. Rather "
        "than assume the consequence we measured it: CLIPSeg attains 0.1849 IoU overall on this "
        "task, which is a strong score, and 0.054 on insertions, which is worse than predicting the "
        "entire frame. The failure is concentrated exactly where the denoted object and the region "
        "to be changed come apart. VespaSeg [[vespaseg]] is a precedent in the useful direction, showing that "
        "lightweight text-conditioned segmentation is viable without a large backbone. The remaining "
        "works in the cluster advance referring segmentation itself — language-grounded query banks "
        "[[anchorseg]], set-level concept prediction [[setcon]], dynamic gating between semantic and "
        "spatial predictions [[dgseg]] and multi-turn tool invocation [[rsagent]] — which improves "
        "the wrong target more accurately.")

d.add_heading("2.8  Benchmarks and Metrics", level=2)
cpara(d, "Four works, none of which scores a predicted region. LocateEdit-Bench [[locatebench]] localizes edits in "
        "231,000 images that have already been edited, which is forensic analysis of a finished "
        "result rather than an evaluation of a region predicted in advance. GEdit-Bench [[gedit]] and the "
        "preservation-modification metric [[prefmetric]] score the output image. The affordance "
        "benchmark [[affbench]] measures "
        "plausibility of placement, not intersection over union against a region. This absence is "
        "the mechanism by which a 2026 paper built around a MaskPredictor can report no mask IoU "
        "and attract no objection.")

d.add_heading("2.9  Planning, Scene Representations and Surveys", level=2)
cpara(d, "Thirteen works that bound the problem from outside. X-Planner [[xplanner]] decomposes "
        "instructions into steps and RePlan [[replan]] adds region-aligned guidance through a "
        "multimodal planner, but that "
        "planning is semantic — what to do, and in what order — and stops short of pixels, while "
        "placing a multi-billion parameter model in the loop for a decision this project's head "
        "makes in about five milliseconds. The structured-representation works, among them I2E's "
        "instance-level amodal decomposition [[i2e]] and layer-wise latent memory [[layermem]], build persistent "
        "structure over a scene, but they represent things that exist; none makes empty space "
        "addressable. The three surveys confirm the framing: the 2026 instruction-editing survey "
        "[[editsurvey]] "
        "treats localization as a sub-component of editing rather than as a first-class output with "
        "a metric of its own, and the affordance survey [[affsurvey]] shows that affordance "
        "prediction grew out of robotics rather than out of edit localization. The remainder of "
        "the cluster sits adjacent to the problem rather than inside it: multi-round object "
        "manipulation [[ffse]] and shape editing driven by trajectories rather than by language "
        "[[fys]], agentic 3D scene authoring [[muse]] and editable semantic occupancy [[editssc]] "
        "in three dimensions, and, on the reasoning side, a diagnosis of pixel-grounding "
        "hallucination [[cfseg]] and multi-round conversational segmentation [[segllm]].")

d.add_heading("2.10  Identified Research Gaps", level=2)
para(d, "Four gaps follow from the survey. Each is stated with evidence from the literature and "
        "with a measurement of our own, because a gap argued only from what papers do not say is "
        "weaker than one that has been reproduced.")
para(d, "G1 — Producing a region is unsolved while editing given a region is not. Seventeen works "
        "consume a mask and perform well with it. Our measurement: the selection baseline of "
        "Section 1.2 scores 0 of 7, failing on granularity, on spatial language and, for "
        "insertions, in principle.")
cpara(d, "G2 — The edit region is frequently not an object. The base paper [[adaptedit]] says so in its "
        "limitations and WhereEdit [[whereedit]] says so in its scope. Our measurement: CLIPSeg reaches 0.1849 "
        "IoU overall but 0.054 on insertions, so the gap is specific to the case where no source "
        "object exists rather than general weakness.")
para(d, "G3 — Supervision binds harder than architecture. Nobody can annotate where a hat should go "
        "on a dog that has no hat, so the annotations that exist mark a region of interest instead. "
        "Our measurement over all 528 MagicBrush development turns is given in Table 3: the human "
        "masks have 96.7 per cent recall and 15.8 per cent precision against the region that "
        "actually changed. They are correct and far too large, and they are worst on insertions.")
table(d, "Table 3. MagicBrush human annotations measured against the region that actually changed "
         "between source and target, recovered by CIE76 ΔE in L*a*b*. All 528 development turns.",
      ["Edit kind", "n", "Annotated", "Changed", "Ratio", "Recall", "Precision", "IoU"],
      [["Insert", "103", "60.0%", "7.0%", "11.9×", "97.8%", "11.9%", "0.12"],
       ["Modify", "390", "62.9%", "11.2%", "8.5×", "96.5%", "17.1%", "0.17"],
       ["Remove", "35", "63.6%", "8.9%", "7.1×", "96.2%", "13.4%", "0.13"],
       ["All", "528", "62.4%", "10.3%", "9.1×", "96.7%", "15.8%", "0.16"]],
      widths=[0.85, 0.5, 0.85, 0.75, 0.6, 0.7, 0.85, 0.55], bold_rows=(3,))
para(d, "G4 — Nothing scores a predicted edit region. Every benchmark in Section 2.8 scores "
        "something downstream of the region, so a predictor can be published without being "
        "measured on its own output.")

d.add_heading("2.11  Position of This Work", level=2)
para(d, "It is worth separating what this survey licenses us to claim from what it does not. "
        "Available to claim: predicting the edit region from the instruction is a nearly vacant "
        "position, occupied by two works of which one is domain-restricted; deriving the "
        "supervision from pixel differences between a real source and target pair appears "
        "unattempted in this setting; and reporting an intersection over union for the predicted "
        "region is novel by absence rather than by difficulty.")
para(d, "Not available to claim: the mechanism. Feature-wise linear modulation of an image "
        "representation by a text embedding is AdaptEdit's conditioning too, and a loss combining "
        "binary cross-entropy with a Dice term is a standard segmentation choice that the base "
        "paper also makes. Stating this plainly is a stronger position than being caught on it, and "
        "it leaves the contribution where the evidence actually puts it — in the supervision, in "
        "the measurement, and in the representational bound of Section 3.7.")
para(d, "One honest limit on the survey itself: thirty-nine of the forty-five works are from 2025 "
        "or 2026, so the area is moving faster than a survey can settle. The claims that should "
        "carry weight are therefore the measured ones, and in particular the bound, which is a "
        "property of the representation and does not depend on what appears next.")

# ------------------------------------------------------------------ 3
d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
d.add_heading("3  Methodology", level=1)

d.add_heading("3.1  Overview", level=2)
para(d, "The system has two stages. Stage 1 reads an image and an instruction and predicts a binary "
        "edit region. Stage 2 is an editor confined to that region. This document covers Stage 1, "
        "which is where the research question sits; Stage 2 uses an existing editor and is reported "
        "separately.")
para(d, "The design constraint throughout was a single laptop with 26 GB of unified memory. That "
        "rules out training a diffusion transformer and rules in a different division of labour: "
        "two large pretrained encoders are frozen and used only as feature extractors, and a small "
        "head is trained on top of them. Nothing in the backbone learns. The head is 4,693,633 "
        "parameters against 71.75 million frozen in the segmentation backbone, and it is the only "
        "thing that sees a gradient.")
if (FIG / "da2_architecture.png").exists():
    figure(d, str(FIG / "da2_architecture.png"),
           "Figure 2. Stage 1. Both encoders are frozen, so the left branch is computed once per "
           "image and the head is re-evaluated per instruction. The target on the right is derived "
           "from the source and target images of a real human edit, never from the dataset's own "
           "mask annotation.", width=6.3)

d.add_heading("3.2  Frozen Feature Extraction", level=2)
para(d, "The image passes through a frozen YOLOv8x-seg backbone in its FastSAM configuration — "
        "71,751,811 parameters, used purely as a feature source. Two things are taken from it. The "
        "first is the set of 32 mask prototypes its segmentation head emits, at stride 4, giving "
        "160 × 160 maps for a 640 × 640 input. The second is a 640-dimensional image context vector "
        "pooled from the SPPF layer. The instruction passes through a frozen CLIP ViT-B/32 text "
        "encoder to a 512-dimensional embedding.")
para(d, "The prototypes are the load-bearing choice. In ordinary use, a detection supplies the "
        "coefficients that combine them, which is exactly why the output of such a head can only "
        "ever be an object the detector found. Taken by themselves the prototypes are spatial basis "
        "functions: any mask expressible as a linear combination of them, pushed through a sigmoid, "
        "is reachable. Predicting the coefficients from the instruction instead of from a detection "
        "is what removes the constraint that the answer be an existing object — which is the "
        "constraint that made the selection baseline of Section 1.2 fail on insertions in "
        "principle. Section 3.7 describes how far that reasoning actually holds, which is not as "
        "far as it was originally asserted to.")

d.add_heading("3.3  Coefficient Prediction", level=2)
para(d, "Both encoders produce vectors that must be fused before coefficients can be predicted. The "
        "text embedding is projected and used to modulate the image context feature-wise, scaling "
        "it by γ(t) and shifting it by β(t). This is standard conditioning and, as Section 2.11 "
        "notes, the same mechanism the base paper uses.")
para(d, "Two heads were built on top of that fused vector. The first, v1, emits a single "
        "32-dimensional coefficient vector and a bias for the whole image, so the mask is one fixed "
        "linear combination of the prototypes applied everywhere in the frame. The second, v2, "
        "emits a 20 × 20 field of coefficients, bilinearly upsampled to prototype resolution, "
        "alongside a global branch that lets the instruction still set an overall prior. Different "
        "parts of the frame can then draw on different mixtures of prototypes.")
table(d, "Table 4. The two coefficient heads.",
      ["Head", "Trainable parameters", "What the instruction produces"],
      [["v1 global", "1,399,073", "one 32-dimensional vector and a bias, per image-instruction pair"],
       ["v2 spatial", "4,693,633", "a 20 × 20 coefficient field, upsampled to 160 × 160"]],
      widths=[1.1, 1.5, 4.0], align=["left", "right", "left"], bold_rows=(1,))
para(d, "The reason for building the second head was a measurement rather than an intuition. v1 "
        "gains 0.0012 IoU when its width is tripled from 1.4 to 4.4 million parameters, which says "
        "that capacity was never what limited it. A single coefficient vector suits a target that "
        "is one object and suits nothing else, so the limitation is expressiveness. Predicting a "
        "field rather than a vector addresses that directly, and Section 3.7 states how the "
        "comparison between the two is decided.")

d.add_heading("3.4  Supervision", level=2)
para(d, "Training this head needs a ground-truth edit region for each instruction, and this is "
        "where the project departs from the base paper. MagicBrush ships human mask annotations, "
        "and they are not usable as targets for the reason measured in Table 3: they cover 62.4 per "
        "cent of the frame where 10.3 per cent changed. They are region-of-interest scribbles — a "
        "human roughly brushing “work somewhere in here” for an inpainting tool — and a predictor "
        "trained on them learns to emit regions an order of magnitude too large.")
para(d, "The targets are derived instead from the data MagicBrush provides that is not an "
        "annotation: the real human source and target image pair. Where the edited image differs "
        "from the original is, by construction, where the edit happened. The difference is measured "
        "as CIE76 ΔE in L*a*b* rather than in RGB, because a colour change at near-constant "
        "luminance is perceptually large and numerically small in the wrong space, and an earlier "
        "grayscale implementation dropped such edits from the dataset silently.")
table(d, "Table 5. Target derivation.", ["Parameter", "Value", "Why"],
      [["Colour metric", "CIE76 ΔE in L*a*b*", "perceptual difference, not raw RGB distance"],
       ["Threshold", "12.0", "from a noise-robustness sweep; holds at σ = 8"],
       ["Morphology", "5 × 5 ellipse, open then close", "solidifies speckled regions"],
       ["Minimum component", "40 px", "ΔE is noisier per pixel than grayscale differencing"],
       ["Sample filter", "0.4 % – 45 % changed area", "rejects failed edits and whole-frame restyles"]],
      widths=[1.5, 1.9, 3.2], align=["left", "left", "left"])
para(d, "The threshold was not chosen by eye. It was selected against simulated regeneration noise "
        "on a pair whose true edit region is 3.39 per cent of the frame, since the editors "
        "regenerate the entire image and the threshold's real job is rejecting that background "
        "noise. Without morphological cleanup a threshold of 8 recovers 36.9 per cent at σ = 8, "
        "which is unusable; with cleanup every threshold from 8 upward holds at about 3.47 per cent "
        "against the 3.39 per cent ground truth, which is what made a value as low as 12 safe. "
        "Filtering then drops samples outside 0.4 to 45 per cent changed area: a near-empty "
        "difference means the editor ignored the instruction, and a near-full one means it applied "
        "a global restyle. Neither is a localization target.")
para(d, "The idea in one line is that where a hat landed is ground truth for where a hat should go. "
        "It is worth recording that MagicBrush's own masks use bright to mean preserve; assuming "
        "the opposite polarity produced an IoU of 0.001 in the first run of the Table 3 "
        "measurement, which is how the error was caught.")

d.add_heading("3.5  Dataset and Split Integrity", level=2)
para(d, "MagicBrush supplies real human edits: 51 training shards and 4 development shards, about "
        "25 GB of Parquet. After the filter in Table 5, 8,306 of 8,807 training rows are usable and "
        "503 of 528 development turns survive. The distribution is uneven — modify 6,045, insert "
        "1,828, remove 433 — which matters when reading per-kind results.")
para(d, "MagicBrush is multi-turn: one photograph yields several successive edits, so splitting by "
        "turn rather than by image would place the same photograph on both sides of the split and "
        "inflate every number reported. Every source and target image on both sides was therefore "
        "hashed with SHA-1 and the two sets intersected. The audit was run before the results were "
        "trusted, not afterwards to defend them; an earlier phase of this project had a "
        "contaminated cache that produced a plausible-looking headline, and the retraction that "
        "followed is why the check is now a precondition.")
table(d, "Table 6. Dataset after filtering, and the disjointness audit.", ["Quantity", "Value"],
      [["Training shards / rows / usable", "51 / 8,807 / 8,306"],
       ["Training kinds", "modify 6,045 · insert 1,828 · remove 433"],
       ["Held-out development turns", "503 of 528"],
       ["Distinct development images", "794"],
       ["Distinct training images", "13,317"],
       ["Images appearing in both", "0"]],
      widths=[3.0, 3.2], align=["left", "left"], bold_rows=(5,))

d.add_heading("3.6  Training Protocol", level=2)
para(d, "The frozen features are precomputed once into memory-mapped arrays totalling 13.8 GB, so "
        "each epoch touches only the head. That is what brings a twelve-seed experiment inside a "
        "laptop's budget: the backbone runs once for the whole study rather than once per step.")
table(d, "Table 7. Training configuration.", ["Parameter", "Value"],
      [["Optimiser", "AdamW"],
       ["Learning rate", "1 × 10⁻³, cosine decay to zero"],
       ["Weight decay", "0.01"],
       ["Batch size", "32"],
       ["Epochs", "60, best checkpoint selected on a train-validation slice"],
       ["Loss", "BCE + 2 × Dice"],
       ["Gradient clipping", "global norm 1.0"],
       ["Seeds", "12 (1368, and 1 through 11)"],
       ["Decision threshold", "0.5, fixed before evaluation"]],
      widths=[2.2, 3.4], align=["left", "left"])
para(d, "Three choices in that table are protocol rather than hyperparameters, and each exists "
        "because of something that went wrong earlier. The checkpoint epoch is selected on a "
        "held-out 15 per cent slice of the training partition, never on the development set: an "
        "early evaluation that selected on development concluded that the spatial head was worse "
        "than the global one, and selecting on train-validation reverses that ordering at "
        "p = 0.0070. Test-set selection can invert a conclusion, so it is worth the extra slice. The "
        "development set is touched exactly once per seed. The decision threshold is fixed at 0.5 "
        "before evaluation, even though sweeping it on the development set peaks near 0.3; adopting "
        "0.3 on that basis would be the same error in a different place. Every seed seeds the "
        "PyTorch, NumPy, Python and Metal generators, and identical commands reproduce identical "
        "numbers.")

d.add_heading("3.7  Evaluation", level=2)
para(d, "The primary metric is intersection over union between the predicted region and the derived "
        "target, computed at full resolution against the undilated mask, with precision, recall and "
        "Dice reported alongside it because two models with the same IoU can differ sharply in how "
        "they trade the two. Results are reported per edit kind as well as overall, since the "
        "insertion case is the one the project exists to address. Comparisons between heads use "
        "Welch's t-test on the twelve seed means, and comparisons against fixed baselines use "
        "one-sample tests.")
para(d, "Two further evaluations are specified because the IoU alone would mislead. The first is an "
        "ablation in which the text embedding is zeroed: a head that scores well with no "
        "instruction has learned an image prior, not a grounding, and an earlier sweep found "
        "configurations that reached a competitive IoU with a language contribution of "
        "approximately zero by converging on a blob covering a fifth of the frame. Model selection "
        "is therefore on the ablation delta, not on raw IoU.")
para(d, "The second is a representational bound. Fitting the coefficients directly to the "
        "ground-truth mask by least squares gives the best IoU any coefficient predictor could "
        "reach over these 32 prototypes, with no learning involved. Measured on the development "
        "set, that bound is 0.4325 for removal, 0.3867 for modification and 0.2119 for insertion. "
        "The insertion bound is less than half the removal bound, which refutes the justification "
        "originally given for this architecture: FastSAM's prototypes were trained to segment "
        "objects and carry that bias, so they span object-shaped regions roughly twice as well as "
        "they span empty space. Reporting the fraction of the bound attained separates a model that "
        "has not learned something yet from a representation that cannot express it, and it is the "
        "measurement that makes the insertion result interpretable rather than merely "
        "disappointing.")
if (FIG / "fig2_example_prediction.png").exists():
    figure(d, str(FIG / "fig2_example_prediction.png"),
           "Figure 3. The quantity being scored. Source image, the region predicted from the "
           "instruction, and the region overlaid on the image, for a held-out development sample.",
           width=6.2)

# ------------------------------------------------------------------ refs
d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
d.add_heading("References", level=1)
para(d, "Forty-five works, most recent first. Cited in text as [n].",
     size=10, italic=True, space=10, color=MUTE)
for n, r in enumerate(sorted(rows, key=lambda x: (-int(x["Year"]), str(x["Paper"]))), 1):
    p = d.add_paragraph()
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.left_indent = Inches(0.35)
    p.paragraph_format.first_line_indent = Inches(-0.35)
    a = p.add_run(f"[{n}]  "); a.font.size = Pt(9.5)
    title = str(r["Paper"]).replace("*", "").replace("[BASE PAPER]", "").strip()
    if str(r["PDF link"]).startswith("http"):
        hyperlink(p, title, str(r["PDF link"]), size=9.5)
    else:
        t = p.add_run(title); t.font.size = Pt(9.5)
    vid = f" arXiv:{r['arXiv / ID']}." if str(r["arXiv / ID"]) not in ("", "-", "None") else ""
    b = p.add_run(f". {r['Venue']}, {r['Year']}.{vid}"); b.font.size = Pt(9.5)

out = HERE / "DA2_Report.docx"
d.save(str(out))
print("wrote", out)
