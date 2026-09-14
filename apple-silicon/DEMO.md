# Demonstration — how to run it

Stop every background job first, or the timings are contention-inflated and the
model looks ~10x slower than it is:

    pkill -f stage2_sd_sweep; pkill -f clean_eval_full

Then:

    ./run_demo.sh

Or one case at a time:

    ./.venv/bin/python demo.py --image demo_images/insert_253975.png \
        --instruction "Add a cruise ship to the ocean."

Add `--edit` to also generate the edited image inside the predicted region
(loads Stable Diffusion inpainting, ~40 s extra on first run).

## What each run prints

- head size (4,693,633 trainable params) and the dev IoU of the checkpoint
- **per-image** cost: the frozen backbone, paid once per photograph
- **per-instruction** cost: CLIP + the head, paid per instruction
- the amortisation table: what N instructions on one image cost us vs CLIPSeg

Benchmark figures on an idle machine, matched at 352 px: **33.6 ms per image,
5.30 ms per instruction, against CLIPSeg's 54.3 ms** — 10.3x cheaper per
instruction at 32x fewer trainable parameters.

## Operating point

`--thr 0.3 --dilate 32` are the defaults, chosen by the joint sweep in
experiment 17. For mask-quality demonstrations rather than editing, `--dilate 0`
shows the raw prediction.

## Coverage varies by image — know this before you run it

The predicted region is a fraction of the frame, and it is not uniform across
images. Measured at threshold 0.3 with no dilation on the five demo images:

| image | instruction | frame covered |
|---|---|---:|
| modify_319096 | paper on the wall -> mirror | 9.5% |
| modify_140513 | mountains in the background | 29.1% |
| insert_253975 | add a cruise ship | 29.7% |
| insert_17320 | add a dog bowl | 41.1% |
| remove_557105 | remove the yellow flowers | 56.3% |

The last two are poor: dense close-ups where the model cannot localize and
over-predicts badly. `run_demo.sh` uses the first three. If asked whether it
always works, say no and point at the per-kind numbers and the ceiling — that is
a stronger answer than a demo that only ever shows easy cases.

## The four cases in run_demo.sh

1. **Modification, tight region** — 0.2313 IoU overall for this kind, the highest
   fraction of its ceiling at 59.8%
2. **Modification, large region** — background edit, shows the mask scaling up
3. **Insertion** — 0.1186 IoU. Expect an imperfect region: this is the case the
   base paper names as unsolved, and the prototype basis caps any predictor at
   0.2119 here. Show it and explain the bound rather than avoiding it.
4. **End to end** — region prediction followed by inpainting inside it, at the
   swept operating point (threshold 0.3, dilation 32 px)
