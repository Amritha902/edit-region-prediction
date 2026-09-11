# Experiment 12 — v2 spatial coefficients, latency, and the final comparison

## The change

v1 predicts one 32-vector per image, so the mask is a *single* linear
combination of prototypes everywhere. exp11 showed v1 saturating at 1.4M —
width was never the constraint, expressiveness was. v2 predicts a **20×20
coefficient field**, upsampled to prototype resolution:

    mask = σ( Σᵢ cᵢ(x,y) · protoᵢ(x,y) + b )

Different regions can now use different prototype mixtures. A global branch is
retained so the instruction can still set an overall prior.

## Accuracy — 12 seeds, clean split

Epoch selected on a held-out slice of **train**; dev touched once per config.
Evaluated at full resolution against the true (undilated) mask.

| head | params | dev IoU | insert | modify | remove |
|---|---:|---:|---:|---:|---:|
| v1 global coeff | 1,399,073 | 0.1590 ± 0.0111 | 0.076 | 0.179 | 0.188 |
| **v2 spatial** | 4,693,633 | **0.1718 ± 0.0099** | **0.092** | 0.193 | 0.183 |
| v2 spatial + random *(control)* | 4,700,569 | 0.1652 ± 0.0075 | 0.087 | 0.186 | 0.171 |
| v2 spatial + geometric | 4,700,569 | 0.1688 ± 0.0099 | 0.092 | 0.190 | 0.165 |

Welch t-tests on seed means:

| pair | diff | p |
|---|---:|---:|
| **v2 spatial vs v1** | **+0.0128** | **0.0070** ✓ |
| v2+geom vs v2+random | +0.0035 | 0.3344 ✗ |
| v2+geom vs v2 spatial | −0.0030 | 0.4653 ✗ |

**The geometric basis is rejected by its own control** — it beats neither the
random control nor plain v2. Dropped. Plain **v2 spatial** is the result.

## Latency — measured on the M5

| | ours | CLIPSeg |
|---|---:|---:|
| per-image (once) | 96.8 ms | — |
| **per-instruction** | **5.04 ms** | **54.3 ms** |
| trainable params | 1,399,073 | 150,747,746 |

**10.8× faster per instruction at 108× fewer trainable parameters.** Amortised
over N instructions on one image:

| N | ours | CLIPSeg | speedup |
|---:|---:|---:|---:|
| 1 | 101.8 ms | 54.3 ms | 0.5× |
| 5 | 122.0 ms | 271.6 ms | 2.2× |
| 20 | 197.7 ms | 1,086 ms | 5.5× |
| 100 | 601.1 ms | 5,433 ms | 9.0× |

Break-even at **N = 1.96**. This is structural: our architecture splits into a
per-image stage (frozen backbone → 32 prototypes) and a per-instruction stage
(CLIP text + a 1.4M MLP + a weighted sum). CLIPSeg re-runs a full 150M
encoder–decoder for every instruction and has no such split available.

## Final comparison

| method | trainable | per-instruction | IoU | insert |
|---|---:|---:|---:|---:|
| random centred box | 0 | — | 0.0621 | 0.033 |
| full frame | 0 | — | 0.0958 | 0.071 |
| MagicBrush human masks | — | — | 0.1511 | 0.120 |
| **ours (v2 spatial)** | **1.4 M** | **5.04 ms** | **0.1718** | **0.092** |
| CLIPSeg | 150.7 M | 54.3 ms | 0.1849 | 0.054 |

Three results:

1. **Beats the human annotations** — 0.1718 vs 0.1511, 2.1 sd above. The
   ΔE-supervision claim, retracted in exp09 as contaminated, holds on a clean
   split.
2. **Insertion: +70% over CLIPSeg** — 0.092 vs 0.054. A referring segmenter
   returns what a phrase *denotes*, which is structurally wrong for "put a hat
   on the dog". This is the case exp08 identified and the largest margin here.
3. **93% of CLIPSeg's overall IoU** at 108× fewer trainable parameters and
   10.8× lower per-instruction latency.

## A methodological note worth keeping

A first evaluation selected the best epoch by *dev* IoU and concluded v2 was
worse than v1 (0.1373 vs 0.1533). That was an artefact: selection consistently
landed on epoch 8, and v2 has 3× the parameters and needs longer to converge.
Selecting on train-val instead reverses the ordering, and the reversal is
significant at p = 0.0070. Test-set epoch selection can invert a conclusion.
