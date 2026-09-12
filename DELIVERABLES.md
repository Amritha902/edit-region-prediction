# Review 1 — what has to be produced

Four separate submissions. Recorded 2026-09-12 so nothing is lost between sessions.

## 1. Full technical report / paper
Methodology and every training result. Sections needed:
- problem, the gap AdaptEdit states in its own limitations section
- why AdaptEdit cannot be reproduced (20.43B frozen + 5.0B trainable; it never
  reports a mask IoU) and what we built instead
- architecture: v1 global coefficients vs v2 spatial coefficient field
- data: MagicBrush, the dE target derivation, why not their masks
- protocol: clean split, train-val epoch selection, 12 seeds
- results: n=2,278 vs n=8,306, per-kind, ceilings, latency, Stage 2
- honest limitations: the 0.211 insertion ceiling, frozen backbone, stale
  hyperparameters, InstructPix2Pix still standing in for SD inpainting
- provenance appendix -> apple-silicon/PROVENANCE.md

## 2. Literature survey document
Already built at ~/DATASCIENCE FINAL/LITERATURE/ -- 50 papers, 9 columns,
3 sheets, 42 PDFs. Needs writing up as prose: themes, what each line of work
built, what is missing, how the gap follows from it.

## 3. Consolidated "entire project" document
Everything in one place: survey + methodology + experiments exp01-exp13 +
results + figures + code map + provenance.

## 4. Presentation deck
11 slides exist at ISE_Review1_Final.pptx. Needs the 8,306 numbers, the new
figures, and speaker notes (PH1-PH10, PH8b are still placeholders).

## Assets already produced
| asset | path |
|---|---|
| results, side by side | apple-silicon/implementation/results_full.png |
| qualitative predictions | apple-silicon/implementation/qualitative_full.png |
| run log evidence | apple-silicon/implementation/run_evidence.png |
| per-seed values, n=8,306 | apple-silicon/train/partial_v2.json |
| per-seed values, n=2,278 | apple-silicon/train/clean_eval.json |
| leakage audit | apple-silicon/train/leakage_check.json |
| environment + commands | apple-silicon/PROVENANCE.md |
| per-epoch curves | apple-silicon/train/curves_full.json (pending, ~07:00) |

## Still running as of 2026-09-12 21:35
- v1 x 12 seeds at n=8,306 -> ~03:00, completes the comparison table
- 3 seeds x v1/v2 with --curves -> ~07:00, learning curves + language ablation
