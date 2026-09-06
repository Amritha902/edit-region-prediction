# Experiment 04 — the amortisation curve

3 COCO images, instructions generated from each field's own detected regions so
every one refers to something present. Apple M5. `python run.py`.

| N | field total | recompute | speedup | per-query | resolved |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.367 s | 0.46 s | 0.3× | 135 µs | 1/1 |
| 5 | 1.368 s | 2.28 s | 1.7× | 76 µs | 5/5 |
| 10 | 1.368 s | 4.56 s | 3.3× | 50 µs | 10/10 |
| 20 | 1.368 s | 9.12 s | 6.7× | 47 µs | 20/20 |
| 50 | 1.370 s | 22.79 s | 16.6× | 47 µs | 50/50 |
| 100 | 1.372 s | 45.58 s | 33.2× | 48 µs | 100/100 |
| 200 | 1.377 s | 91.15 s | 66.2× | 48 µs | 200/200 |
| 500 | 1.391 s | 227.88 s | **163.8×** | 48 µs | 500/500 |

Mean analysis 456 ms/image. Per-query 48 µs — **9,535× cheaper** than one
analysis. Break-even at N = 3.0, which is exactly the number of images: the
field costs one analysis each and nothing after.

The curve has the shape the hypothesis predicts — recompute grows linearly in
N, field cost is a constant plus a term too small to see. That is stronger
evidence than any single speedup figure, because the *shape* is what the claim
"scene understanding is instruction-independent" actually asserts.

## What this is not

Not a comparison against RegionE (2.6×) or SpotEdit (1.95×). Those accelerate
the diffusion pass. This measures only the scene-analysis stage and does not
touch generation. The honest framing: analysis is a cost those systems still
pay per instruction, and it need not be paid more than once per image.

## Limits

Query cost excludes the editor entirely — a real edit is still ~19 s of
diffusion. The 164× is on the analysis stage alone, which for a single edit is
a small share of total time. It matters exactly when a user issues many
instructions against one image, which is the normal interaction pattern and the
premise being tested.

Instructions are template-generated from detected classes, so resolution rate
(500/500) measures coverage of the field's own vocabulary, not open-ended
language understanding.
