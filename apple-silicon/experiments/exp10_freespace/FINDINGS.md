# Experiment 10 — do free-space prototypes lift the insertion ceiling?

exp08 measured that FastSAM's 32 prototypes cap insertion at IoU ≈ 0.20 against
≈ 0.46 for removal, and proposed the fix: augment the basis with functions that
are not object-derived. This tests that proposal, with a control.

Method: the same least-squares ceiling as exp08 — coefficients fitted directly
to the ground-truth mask, so the number is representational capacity, not
optimisation. 2,278 train-shard samples (insert 519, modify 1,632, remove 127).

## Results

| basis | extra fns | insert | modify | remove |
|---|---:|---:|---:|---:|
| 32 prototypes (baseline) | 0 | 0.1969 | 0.3451 | 0.4638 |
| + 12 geometric (x, y, x², sin, cos) | 12 | **0.3244** | 0.4391 | 0.5806 |
| + 12 random smooth — **control** | 12 | **0.3044** | 0.4165 | 0.5427 |
| + 8 free-space | 8 | 0.2533 | 0.3942 | 0.5330 |
| + 8 free-space + 12 geometric | 20 | 0.3495 | 0.4603 | 0.5996 |

## The proposal is not supported

**The control captures 84% of the gain.** Twelve *random* smooth fields lift the
insertion ceiling by +0.1075; twelve geometric ones by +0.1275. Almost all of
the improvement is extra capacity — 32 → 44 basis functions fit anything
better — rather than anything about empty space.

Per basis function, which is the fair comparison:

| family | Δ insert ceiling per function |
|---|---:|
| geometric | **+0.0106** |
| random smooth | +0.0090 |
| **free-space** | **+0.0071** |

The free-space construction is **less useful than noise**, per function. The
hypothesis that empty-space-aware basis functions specifically fix insertion is
rejected.

## What is real

Geometric beats random by **+0.0200** at identical capacity. That gap is the
only part attributable to genuine positional information: a smooth position
encoding helps slightly, beyond raw capacity.

## It is also not insertion-specific

Every edit type rises: modify 1.27×, remove 1.25×, insert 1.65×. Insertion gains
most in relative terms, but the random control alone gives 1.55× of that, so
even the edge is mostly capacity.

## Where this leaves the diagnosis

exp08 stands: the basis has a ceiling, and it is much lower for insertion. This
experiment shows the ceiling *can* be raised — by adding capacity, not by adding
free-space knowledge. The proposed mechanism is not the mechanism doing the work.

Anything that claims to fix insertion must now beat a random-smooth-basis
control of the same size. That is the bar this experiment sets, and it is the
useful output: a cheap, decisive test for the next idea.
