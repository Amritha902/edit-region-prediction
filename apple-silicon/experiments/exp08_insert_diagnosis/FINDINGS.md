# Experiment 08 — why does insertion not improve with more data?

Insertion is the case the whole project is about, and it was the only edit type
that did not improve when the training set grew 5×: modify 0.11 → 0.164 and
remove 0.10 → 0.148, while insert went 0.067 → 0.062. This tests three
hypotheses. Run: `python diagnose_insert.py`. 2,781 samples
(insert 621, modify 2,001, remove 159).

## H1 — too few insertion samples → **rejected**

Trained per-kind at matched sample count:

| trained on | n_train | IoU | delta |
|---|---:|---:|---:|
| insert only | 497 | **0.0450** | +0.0388 |
| modify @ same n | 497 | 0.1568 | +0.1269 |
| remove @ same n | 128 | 0.1190 | +0.1059 |

Given the same amount of data, insertion is 3.5× worse than modify — and worse
than remove on a third of the samples. Data volume is not the constraint.

## H2 — the ΔE targets are worse for insertion → **minor factor**

| kind | area | blobs | largest component | boundary fraction |
|---|---:|---:|---:|---:|
| insert | 6.81% | 8.58 | 87.3% | 37.9% |
| modify | 10.44% | 7.58 | 88.3% | 32.2% |
| remove | 9.99% | 6.84 | 86.3% | 31.2% |

Insertion targets are smaller, slightly more fragmented and more
boundary-dominated. Harder, but not by a margin that explains a 3.5× gap.

## H3 — the prototype basis cannot represent insertion regions → **confirmed**

Coefficients were fitted **directly to the ground-truth mask by least squares**.
This is an upper bound: the best IoU *any* coefficient predictor could achieve
with these 32 prototypes, with no learning involved.

| kind | ceiling IoU | ceiling Dice | our model | fraction of ceiling reached |
|---|---:|---:|---:|---:|
| remove | **0.4575** | 0.5781 | 0.148 | 32% |
| modify | **0.3866** | 0.4793 | 0.164 | 42% |
| **insert** | **0.2112** | 0.2805 | 0.062 | 29% |

The ceiling for insertion is **less than half** that for removal. No amount of
data or training improves a representational limit.

## The claim this refutes

The architecture was justified as follows:

> "Prototypes are spatial basis functions, not objects, so a text-derived
> combination can describe empty space."

**That is wrong.** FastSAM's prototypes were trained to segment objects and
carry that bias: they span object-shaped regions roughly twice as well as they
span empty space. The mechanism does not do the thing it was designed to do for
the case it was designed for.

## Why this matters

It gives a measured, mechanistic reason why insertion is hard in this family of
models — and it plausibly generalises. AdaptEdit also decodes its mask from a
backbone trained on object-centric objectives, and insertion is precisely the
case its limitations section singles out. A representational ceiling would
explain both.

## What it points to

Insertion needs a basis that is not object-derived: free-space geometry,
support-surface priors, or the existing prototypes augmented with a learned
empty-space component. The experiment to run next is whether adding k learned
free-space prototypes raises the insert ceiling above 0.21.
