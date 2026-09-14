#!/bin/bash
# Live demonstration. Stop background jobs first or the timings are inflated:
#   pkill -f stage2_sd_sweep
cd "$(dirname "$0")"
V=./.venv/bin/python
echo
echo "####  1. Modification — the model's best case  ####"
$V demo.py --image demo_images/modify_319096.png \
  --instruction "Make the piece of paper hanging on the wall a mirror" --out demo_1.png
echo
echo "####  2. A larger region — background modification  ####"
$V demo.py --image demo_images/modify_140513.png \
  --instruction "edit some mountains in the background" --out demo_2.png
echo
echo "####  3. Insertion — the hard case the project exists for  ####"
$V demo.py --image demo_images/insert_253975.png \
  --instruction "Add a cruise ship to the ocean." --out demo_3.png
echo
echo "####  4. End to end: predict the region, then edit inside it  ####"
$V demo.py --image demo_images/modify_319096.png \
  --instruction "Make the piece of paper hanging on the wall a mirror" \
  --thr 0.3 --dilate 32 --edit --out demo_4_edited.png
