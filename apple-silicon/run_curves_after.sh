#!/bin/bash
# Wait on the exact PID of the 12-seed run, not a pgrep pattern -- a pattern
# match once made a waiter in this project match itself and spin for 3 hours.
cd "/Users/amritha/DATASCIENCE FINAL/instruct-seg-edit-mac"
while ps -p 2611 >/dev/null 2>&1; do sleep 60; done
echo "=== 12-seed run exited $(date '+%H:%M') — starting curve capture ==="
exec caffeinate -dimsu ./.venv/bin/python train/clean_eval_full.py \
     --seeds 3 --configs v2,v1 --curves --out train/curves_eval.json
