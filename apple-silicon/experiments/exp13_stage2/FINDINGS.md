# Experiment 13 — Stage 2: editing inside the predicted region

Stage 1 predicts *where*. This closes the loop and tests the project's central
claim, which had never been measured: does predicting the region actually
improve editing?

**Design.** One editor (InstructPix2Pix) produces the edit for every arm; the
only thing that varies is which mask gates it back onto the source. Any
difference is therefore attributable to the mask, not to a difference between
two editors. 60 held-out dev samples.

**Metrics.** `recall` = fraction of the true edit region that changed;
`collateral` = fraction of *outside*-region pixels that changed; `net` =
recall − collateral.

## Results

| gating mask | recall ↑ | collateral ↓ | net ↑ |
|---|---:|---:|---:|
| none — whole frame | 33.3% | 24.2% | +9.1% |
| ours, as predicted | 7.5% | 1.5% | +6.0% |
| **ours, tuned** (thr 0.2, dilate 64 px) | 22.9% | 8.8% | **+14.2%** |
| MagicBrush human mask | 31.5% | 13.0% | +18.5% |
| **ground-truth region** | 30.2% | 0.1% | **+30.1%** |

## Three readings

**The premise holds.** Editing inside the correct region is worth **3.3×**:
+30.1% against +9.1% for the whole frame. This is the first direct evidence in
the project that localization pays off downstream, not just on a mask metric.

**Our mask was precision-biased, and that was fixable.** As predicted it gave
1.5% collateral but only 7.5% recall — gating discarded most of the edit, and
net (+6.0%) fell *below* whole-frame editing. Dilating trades precision for
recall: at threshold 0.2 and 64 px the net reaches **+14.2%, 1.56× the status
quo**. Swept to 200 px, net **peaks at 64 px** and declines monotonically beyond it
(96 px +14.1%, 128 px +13.8%, 160 px +13.2%, 200 px +12.3%), so 64 px is the
true optimum rather than the edge of the search.

**Coarse supervision costs at the editing stage too.** The MagicBrush mask —
measured in exp07 as 9.1× too large — carries 13.0% collateral and loses 11.6
points of net against the ground-truth region. The cost of that supervision is
not confined to the mask metric; it propagates to the edited image.

## Honest limits

- Our tuned mask beats whole-frame editing but does **not** yet beat the
  MagicBrush mask (+14.2% vs +18.5%). Their masks are far too large but retain
  more of the edit.
- Gating drives collateral toward zero by construction, so `net` — not
  collateral alone — is the quantity to read.
- 60 samples, one editor, one guidance setting.
