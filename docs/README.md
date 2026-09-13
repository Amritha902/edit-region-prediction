# Review 1 — submission documents

Four deliverables, plus the scripts that build them.

| File | What it is |
|---|---|
| `Review1_Technical_Report.docx` | Method, protocol, all results, statistics, limitations, reproduction. 16 tables, 4 figures. |
| `Review1_Literature_Survey.docx` | 45 works, 2024–2026, nine clusters. 41 clickable paper links, cost analysis, and what each line of work leaves unmeasured. |
| `Review1_Consolidated_Report.docx` | All 13 experiments, the four corrective actions, repository map, open problems. |
| `ISE_Review1_Final.pptx` | 13 slides with full speaker notes. |

## Rebuilding

The `.docx` files are generated, not hand-edited, so a number can never drift
between documents:

```
python build_report_docx.py
python build_survey_docx.py
python build_consolidated_docx.py
```

`mkdocx.py` holds the shared layout: title page, contents, booktabs tables,
figures and real hyperlinks. Every figure comes from `../apple-silicon/train/`
and every number traces to a committed JSON result.

HTML versions (`report.html`, `survey.html`, `dossier.html`) share `paper.css`.

## Known gap

Two works in the survey carry no URL — RePlan (ECCV 2026) and Follow-Your-Shape
(ICLR 2026) — because the matrix records no arXiv identifier for either. They
are cited as text. A guessed URL would be worse than none.
