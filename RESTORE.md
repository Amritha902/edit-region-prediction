# Restoring this project on a fresh machine

Everything irreplaceable is in this repository. Everything else is a download.

## What is here

| | |
|---|---|
| `apple-silicon/checkpoints/` | **24 trained head checkpoints** — 12 seeds x 2 architectures. About 21 hours of compute. These cannot be regenerated without re-running everything. |
| `apple-silicon/train/*.json` | Every measured result: clean_eval, metrics, threshold selection, latency, stage 2, the operating-point sweep |
| `apple-silicon/train/*.py` | All code |
| `apple-silicon/experiments/` | One FINDINGS.md per experiment, failures included |
| `docs/` | The four submission documents and the scripts that build them |
| `apple-silicon/demo_images/` | Demo images with their instructions |

## What is NOT here, and how to get it back

| | Size | How |
|---|---:|---|
| MagicBrush dataset | 26 GB | `huggingface-cli download osunlp/MagicBrush --repo-type dataset --local-dir data/magicbrush` |
| Frozen feature cache | 13 GB | `python train/precompute_full.py --split train --batch 8` — 35 min |
| SD inpainting weights | 2.0 GB | `huggingface-cli download runwayml/stable-diffusion-inpainting --local-dir models/sd-inpaint --include "*.fp16.safetensors" "*.json" "*.txt"` |
| InstructPix2Pix | 2.1 GB | `huggingface-cli download timbrooks/instruct-pix2pix --local-dir models/instruct-pix2pix` |
| FastSAM `best.pt` | 287 MB | FastSAM-x release, or `yolov8x-seg.pt` from Ultralytics |

## Rebuilding the environment

    python3.10 -m venv .venv
    ./.venv/bin/pip install torch numpy opencv-python pandas scipy pillow \
        matplotlib ultralytics diffusers transformers openpyxl python-docx

Versions the results were produced with are in `apple-silicon/PROVENANCE.md`.

## Reproducing a headline number without retraining

The checkpoints are committed, so evaluation runs immediately:

    python train/metrics_full.py        # precision/recall/F1, needs cache_dev.pt
    python train/latency_v2.py          # both heads, resolution-matched
    python demo.py --image demo_images/modify_319096.png \
        --instruction "Make the piece of paper hanging on the wall a mirror"

`cache_dev.pt` (838 MB) is itself regenerable with `precompute.py --split dev`.

## What was running when the machine was reset

The joint threshold x dilation sweep for Stage 2 was partway through. Completed
cells are banked in `apple-silicon/train/stage2_joint*.json` and
`stage2_cell_*.json`. To finish the remaining ones, re-run:

    ./run_cells.sh

It skips any cell whose JSON already exists, so it resumes rather than restarts.
