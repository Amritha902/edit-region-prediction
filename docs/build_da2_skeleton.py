import sys, json; sys.path.insert(0,".")
from mkdocx import *

rows=[r for r in json.load(open("/tmp/lit.json")) if int(r["Year"])>=2024]
URL_FILL={"US12561956":"https://patents.google.com/patent/US12561956",
 "US12462449B2":"https://patents.google.com/patent/US12462449B2",
 "Qwen-Image-Edit":"https://huggingface.co/Qwen/Qwen-Image-Edit",
 "FLUX.1 Kontext":"https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev"}
for r in rows:
    if not (r["PDF link"] and str(r["PDF link"]).startswith("http")):
        for k,u in URL_FILL.items():
            if k in str(r["Paper"]): r["PDF link"]=u; break

d = base_doc()
titlepage(d,"DA2 Submission",
 "Disentangling Localization from Editing in Instruction-Guided Image Manipulation",
 "Introduction · Literature Survey · Methodology")

def writehere(hint, words):
    p=d.add_paragraph(); r=p.add_run(f"[ WRITE HERE — {hint}   ~{words} words ]")
    r.font.size=Pt(10); r.italic=True; r.font.color.rgb=RGBColor(0xB0,0x40,0x20)
    p.paragraph_format.space_after=Pt(14)

def figslot(n, caption):
    p=d.add_paragraph(); r=p.add_run(f"[ INSERT FIGURE {n} — see figures/ folder ]")
    r.font.size=Pt(10); r.italic=True; r.font.color.rgb=RGBColor(0xB0,0x40,0x20)
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after=Pt(3)
    c=d.add_paragraph(); cr=c.add_run(f"Figure {n}. {caption}")
    cr.font.size=Pt(9.5); cr.italic=True; cr.font.color.rgb=MUTE
    c.paragraph_format.space_after=Pt(14)

para(d,"How to use this file", size=12, bold=True, space=4)
para(d,"Every red bracket is a place you write. The headings, the figures, the tables of "
       "measured numbers and the reference list are already here — those are facts and "
       "structure, not prose, and they are not what Turnitin flags. The sentences must be "
       "yours. Keep DA2_writing_notes.docx open beside this file: it has the facts for "
       "each section in bullet form.", size=10.5, space=10)
para(d,"Delete this page before submitting.", size=10.5, italic=True, space=6, color=MUTE)
d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

# ---------------- 1 ----------------
d.add_heading("1  Introduction", level=1)
d.add_heading("1.1  Background", level=2)
writehere("what instruction-guided editing is; that editors are told WHAT not WHERE; the two failure modes", 150)
d.add_heading("1.2  Motivation", level=2)
writehere("the 'put a hat on the dog' case; why referring segmentation cannot solve it", 150)
figslot(1, "Failure of an off-the-shelf pipeline: FastSAM prototypes selected by CLIP similarity "
           "score 0 of 7 across modification, removal and insertion.")
d.add_heading("1.3  Base paper and the identified gap", level=2)
writehere("AdaptEdit, its scale, that it reports no mask IoU, and the limitation it states about itself", 200)
para(d,"Quote to use verbatim, in quotation marks, with the citation — an attributed quote is "
       "not plagiarism:", size=10.5, italic=True, space=4, color=MUTE)
q=d.add_paragraph(); qr=q.add_run('"Geometry-changing edits with no localized source region '
  '(\'put a hat on the dog\' where the dog has no hat): the GT mask covers the new content\'s '
  'location but there is no clean source signal to \'protect\' outside it." — Cai et al., '
  'arXiv:2604.23763')
qr.font.size=Pt(10.5); q.paragraph_format.left_indent=Inches(0.4)
q.paragraph_format.space_after=Pt(12)
d.add_heading("1.4  Objectives", level=2)
writehere("six numbered objectives — the list is in the writing notes", 150)

# ---------------- 2 ----------------
d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
d.add_heading("2  Literature Survey", level=1)
d.add_heading("2.1  Scope and selection", level=2)
writehere("how many works, the date window, how you scored relevance", 120)
table(d,"Table 1. Works surveyed by publication year.",["Year","Works"],
      [["2024","6"],["2025","20"],["2026","19"],["Total","45"]],
      widths=[1.4,1.0], bold_rows=(3,))
d.add_heading("2.2  Classification of the field", level=2)
writehere("explain the four postures and why sorting this way is more revealing than by date", 200)
table(d,"Table 2. The 45 works by what each does with the edit region.",
 ["Posture","Works","Representative work"],
 [["Consumes a region supplied to it","17","NEP, MaskFlow, LazyDiffusion"],
  ["Reads a region from model internals","11","WhereEdit, RegionE"],
  ["Predicts a region from the instruction","2","AdaptEdit, MADiff"],
  ["Scores or frames the problem","15","LocateEdit-Bench, surveys"]],
 widths=[2.6,0.8,2.6], bold_rows=(2,))
for n,(t,w) in enumerate([("Instruction editors — 9 works",180),("Edit localization — 7 works",200),
        ("Insertion and affordance — 4 works",160),("Region-aware efficiency — 3 works",140),
        ("Text-guided segmentation — 5 works",160),("Benchmarks and metrics — 4 works",140),
        ("Planning, scene representations, surveys — 13 works",160)], start=3):
    d.add_heading(f"2.{n}  {t}", level=2); writehere("summarise the cluster and name what it leaves undone", w)
d.add_heading("2.10  Identified research gaps", level=2)
writehere("the four gaps, each with evidence from the literature AND your own measurement", 300)
d.add_heading("2.11  Position of this work", level=2)
writehere("what is available to claim and what is not — say plainly that FiLM and BCE+Dice are shared", 180)

# ---------------- 3 ----------------
d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
d.add_heading("3  Methodology", level=1)
d.add_heading("3.1  Overview", level=2)
writehere("the two stages, and what trains versus what is frozen", 140)
figslot(2, "System architecture. The frozen backbone runs once per image; the 4.69 M parameter "
           "head runs once per instruction.")
d.add_heading("3.2  Frozen feature extraction", level=2)
writehere("FastSAM prototypes, the context vector, the CLIP text embedding, and why prototypes are a basis", 200)
d.add_heading("3.3  Coefficient prediction", level=2)
writehere("v1 global vector and v2 spatial field; why one vector is structurally limiting", 250)
table(d,"Table 3. The two heads.",["Head","Trainable parameters","Coefficients"],
 [["v1 global","1,399,073","one 32-d vector per image-instruction pair"],
  ["v2 spatial","4,693,633","a 20 x 20 field, upsampled to prototype resolution"]],
 widths=[1.3,1.9,3.4], bold_rows=(1,))
d.add_heading("3.4  Supervision", level=2)
writehere("why the dataset masks are unusable, and how the target is derived instead", 250)
table(d,"Table 4. Supervision parameters.",["Parameter","Value","Reason"],
 [["Colour metric","CIE76 in L*a*b*","perceptual, not raw RGB distance"],
  ["Threshold","12.0","from a noise-robustness sweep; holds at sigma=8"],
  ["Morphological kernel","5 x 5","open then close, to solidify speckled regions"],
  ["Minimum component","40 px","dE is noisier per-pixel than grayscale differencing"],
  ["Sample filter","0.4% - 45%","reject unlearnable edits and whole-frame restyles"]],
 widths=[1.5,1.6,3.5], align=["left","left","left"])
d.add_heading("3.5  Dataset and split integrity", level=2)
writehere("MagicBrush, the counts, and why the image-level audit was necessary", 200)
table(d,"Table 5. Dataset after filtering, and the disjointness audit.",
 ["Quantity","Value"],
 [["Training shards / rows / usable","51 / 8,807 / 8,306"],
  ["Held-out dev turns","503 of 528"],
  ["Distinct dev images","794"],
  ["Distinct training images","13,317"],
  ["Images present in both","0"]],
 widths=[3.2,2.0], bold_rows=(4,))
d.add_heading("3.6  Training protocol", level=2)
writehere("selection on train-val, dev touched once, 12 seeds, and why each matters", 200)
table(d,"Table 6. Training configuration.",["Parameter","Value"],
 [["Optimiser","AdamW"],["Learning rate","1 x 10^-3, cosine to zero"],
  ["Weight decay","0.01"],["Batch size","32"],["Epochs","60, best selected on a train-val slice"],
  ["Loss","BCE + 2 x Dice"],["Gradient clipping","norm 1.0"],
  ["Seeds","12 (1368, 1-11)"],["Decision threshold","0.5"]],
 widths=[2.2,3.0])
d.add_heading("3.7  Evaluation", level=2)
writehere("the metrics, the statistical tests, and the least-squares ceiling method", 200)

# ---------------- refs ----------------
d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
d.add_heading("References", level=1)
para(d,"45 works, most recent first. Cite in text as [n].", size=10, italic=True, space=10, color=MUTE)
for n,r in enumerate(sorted(rows,key=lambda x:(-int(x["Year"]),str(x["Paper"]))),1):
    p=d.add_paragraph()
    p.paragraph_format.space_after=Pt(3); p.paragraph_format.left_indent=Inches(0.35)
    p.paragraph_format.first_line_indent=Inches(-0.35)
    a=p.add_run(f"[{n}]  "); a.font.size=Pt(9.5)
    if r["PDF link"] and str(r["PDF link"]).startswith("http"):
        hyperlink(p, str(r["Paper"]), str(r["PDF link"]), size=9.5)
    else:
        a2=p.add_run(str(r["Paper"])); a2.font.size=Pt(9.5)
    vid=f" arXiv:{r['arXiv / ID']}." if r["arXiv / ID"] not in (None,"-") else ""
    b=p.add_run(f". {r['Venue']}, {r['Year']}.{vid}"); b.font.size=Pt(9.5)
d.save("DA2_document_skeleton.docx"); print("wrote DA2_document_skeleton.docx")
