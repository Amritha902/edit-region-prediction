# Experiment 09 — clean evaluation, and a retraction

## The bug

`overnight.sh` line 23 ran `cp train/cache_all.pt train/cache_train.pt`, where
`cache_all` was dev (503) merged with train (2,278). Both caches were
byte-identical at 4,633,999,625. **The model was trained on the dev split, and
evaluation then used dev.** Caught when a baseline run scored 0.3743 against a
training-time 0.1397 — a gap too large to be a metric difference.

### What this invalidates

- The 8-seed headline of **IoU 0.1397 ± 0.0120**. Its held-out split was random
  within a set containing dev, so it was within-distribution, not held out.
- The claim that **our supervision beats human annotation (0.1632 > 0.16)**.
  That number came from the contaminated cache and was measured at 160×160
  against a target dilated by `cv2.resize(..., INTER_AREA) > 0`. **Retracted.**
- The data-scaling curve, which used the same merged cache. The shape is
  probably still right; the absolute values are not trustworthy.

### What it does not invalidate

- exp07 (supervision 9.1× too coarse) — no training involved.
- exp08 (insertion ceiling 0.211) — a least-squares bound, no training.
- The baselines that do not involve our model.

## Clean protocol

Train on the 2,278 samples from `train-*` shards. Evaluate on the 503 from
`dev-*` shards. Different files, disjoint by construction. 8 seeds.

| | IoU |
|---|---|
| with instruction | **0.1471 ± 0.0117** |
| instruction zeroed | 0.0291 ± 0.0092 |
| **language contributes** | **+0.1180 ± 0.0154** |

Paired t-test **p = 1.13 × 10⁻⁷**. The ablation survives cleanly.

## The comparison table — 503 dev samples, full resolution

| method | params | IoU | insert | modify | remove |
|---|---:|---:|---:|---:|---:|
| random centred box | 0 | 0.0621 | 0.0326 | 0.0695 | 0.0705 |
| full frame | 0 | 0.0958 | 0.0708 | 0.1026 | 0.0977 |
| **ours** | **1.4 M** | **0.1232** | 0.0602 | 0.1392 | 0.1394 |
| MagicBrush GT masks | human | 0.1511 | **0.1195** | 0.1603 | 0.1457 |
| **CLIPSeg** | 150 M | **0.1849** | 0.0542 | **0.2112** | **0.2984** |

**We do not win.** An off-the-shelf CLIPSeg beats us by 50%, and the human masks
we criticised as 9.1× too coarse also beat us. We beat only a random box and
predicting the entire frame.

### On the two numbers for our model

`train_clean.py` reports 0.1471 and `baselines.py` reports 0.1232 for the same
checkpoint. The first evaluates at 160×160 against the dilated training target;
the second at full resolution against the true mask. **0.1232 is the honest
figure** and is what should be quoted.

### The one place we are competitive

Insertion: ours 0.0602 against CLIPSeg's 0.0542. A referring segmenter returns
what a phrase *denotes*, which is structurally wrong for "put a hat on the dog",
and the table shows it. But both sit far below the human masks at 0.1195, and
exp08 explains why ours cannot go higher: the prototype basis caps insertion at
0.211.

## Status

A working pipeline, a clean and significant ablation, two solid measurement
findings, and a model that does not beat a 2022 baseline. That is an honest
negative result. The next experiment is whether free-space prototypes lift the
insertion ceiling, because exp08 says the current basis cannot reach it however
well it is trained.
