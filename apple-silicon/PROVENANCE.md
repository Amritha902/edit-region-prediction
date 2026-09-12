# Run provenance — captured 2026-09-12 21:31 IST

## Machine
```
Apple M5
cores: 10 | RAM: 26 GB
ProductName:		macOS ProductVersion:		26.6.2 BuildVersion:		25G83 
```

## Python / libraries
```
Python 3.10.20
torch        2.13.0   (MPS available: True)
numpy        2.2.6
opencv       5.0.0
pandas       2.3.3
scipy        1.15.3
pillow       12.3.0
matplotlib   3.10.9
ultralytics  8.3.209
transformers 5.15.0
diffusers    0.39.0
clip         (no __version__)
```

## Frozen components
```
best.pt 287525809 bytes
yolov8x-seg.pt 144101612 bytes
```

## Dataset
```
MagicBrush train shards: 51  dev shards: 4
on disk:  25G
cached usable train samples: 8306 of 8807 rows
kinds: {'modify': 6045, 'remove': 433, 'insert': 1828}
leakage audit: dev 794 imgs, train 13317 imgs, overlap 0
```

## Exact commands, in order
```
python train/precompute_full.py --split train --batch 8      # 35.0 min -> 8,306 samples, 13.8 GB
python train/leakage_check.py                                # 0 overlapping images
python train/clean_eval_full.py --seeds 12 --configs v2,v1   # ~9 h, 12 seeds each
python train/clean_eval_full.py --seeds 3 --configs v2,v1 --curves   # per-epoch curves
python train/qualitative_full.py                             # prediction panels
python train/plots_full.py                                   # side-by-side results figure
python train/run_evidence.py                                 # log evidence sheet
```

## Protocol held fixed across n=2,278 and n=8,306
- checkpoint epoch selected on a held-out 15% slice of TRAIN, never on dev
- dev touched exactly once per seed; same 503 dev turns in both runs
- 60 epochs, batch 32, AdamW lr 1e-3, weight decay 0.01, cosine schedule
- loss: BCE + 2x Dice; grad-norm clip 1.0
- 12 seeds: 1368, 1..11; every seed sets torch, numpy, random and mps generators
- targets derived by CIE76 dE in L*a*b*, NOT MagicBrush's own masks
- samples rejected outside 0.4%..45% changed area

## Timings measured on this machine
| stage | wall time |
|---|---|
| feature precompute, 8,807 rows | 35.0 min |
| leakage audit, 55 shards | ~9 min |
| v2 training, per seed (on AC power) | 37-40 min |
| v2 training, per seed (on battery, throttled) | 44-53 min |
| v1 training, per seed | ~32 min |
