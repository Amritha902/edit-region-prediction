# Experiment 11 — their mask-construction mechanism vs ours, controlled

AdaptEdit's full system cannot run here (20.43 B frozen, 5.0 B trainable,
multi-GPU). Its MaskPredictor (§3.6, eq. 8) is a *mechanism*, and a mechanism
can be reimplemented at this scale and compared fairly.

**Controlled:** both heads receive identical inputs — the same frozen FastSAM
prototypes as the spatial feature map, the same pooled image context, the same
CLIP text embedding — trained on the same 2,278 train-shard samples with the
same loss, schedule and seeds, evaluated on the same 503 held-out dev samples.
The only variable is how the mask is constructed.

| | mechanism |
|---|---|
| **theirs** | `M = σ( Upsample( ConvHead( CA(q = γ⊙Q + β, k,v = φ(f)) ) ) )` — a learned 16×16 query grid, FiLM-modulated by the instruction, cross-attending to patchified features. The mask is **decoded** from learned queries. |
| **ours** | `M = σ( Σ cᵢ · protoᵢ + b )` — the instruction predicts 32 coefficients. The mask is a **combination** of an existing basis. |

## Results — 5 seeds, held-out dev

| head | params | IoU | language delta | insert | modify | remove |
|---|---:|---:|---:|---:|---:|---:|
| query-grid d=128 (theirs) | 402,017 | 0.1032 ± 0.0052 | +0.0521 | 0.075 | 0.115 | 0.098 |
| query-grid d=256 (theirs) | 923,489 | 0.1050 ± 0.0079 | +0.0457 | 0.074 | 0.118 | 0.096 |
| **coefficients h=512 (ours)** | **1,399,073** | **0.1547 ± 0.0045** | **+0.1260** | 0.078 | 0.177 | 0.188 |
| coefficients h=1024 (ours) | 4,368,673 | 0.1559 ± 0.0116 | +0.1287 | 0.081 | 0.180 | 0.178 |

**+47% relative IoU**, and the instruction contributes **2.7× more**.

## It is mechanism, not capacity

Both families are saturated at these sizes: theirs gains +0.0018 going 402k →
923k, ours gains +0.0012 going 1.4M → 4.4M. Doubling or tripling parameters
does not close a 0.05 IoU gap. This is the control discipline exp10 established,
applied here.

The likely reason: their decoder must **learn** spatial structure from scratch
through a 16×16 grid and a conv head. Ours reuses 32 prototypes FastSAM already
trained to delineate regions, so when the edit region *is* an object the basis
is already correct and only 32 coefficients are needed.

## The limit of the win

On insertion the two are **statistically identical**: 0.078 vs 0.075. Our entire
advantage is on modify (0.118 → 0.177) and remove (0.096 → 0.188) — exactly the
cases where the region is an object.

Neither mechanism helps for empty space, which is what exp08's ceiling
predicted. **The head is not the bottleneck for insertion; the features are.**

## What may and may not be claimed

**May:** prototype-basis construction beats query-grid decoding on these
features, under matched inputs, data, loss and seeds, and the gap is not
explained by capacity.

**May not:** "we beat AdaptEdit." Their MaskPredictor was designed to read a
mask-aware representation produced by 5.0 B of Block Adapters, SpatialGate and
Region-Aware Loss. Stripped of that scaffolding and fed FastSAM prototypes, this
is their mechanism, not their system. The scaffolding is most of their paper and
is absent here.
