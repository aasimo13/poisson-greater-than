#!/bin/bash
cd "$(dirname "$0")/.."
OUT=${OUT:-results/rerun}; mkdir -p "$OUT"
run(){ python3 src/opt.py "$@" >> $OUT/calib.jsonl 2>> $OUT/calib.err; }
BG=0 EXTRA='[[[104.788,4.691,0.001,29.867],[29.865,0.001,4.691,104.755]]]' run free common 0.005 max 71 1 400
BG=0 run monotone common 0.005 max 72 1 400
BG=0 EXTRA='[[[104.788,4.691,0.001,29.867],[29.865,0.001,4.691,104.755]]]' run free common 0.01 max 71 1 400
BG=0 run monotone common 0.01 max 72 1 400
BG=1.0 EXTRA='[[[156.281,10.351,0.003,49.414],[49.39,0.004,10.35,156.289]]]' run free common 0.005 max 71 1 400
BG=1.0 run monotone common 0.005 max 72 1 400
BG=1.0 EXTRA='[[[156.281,10.351,0.003,49.414],[49.39,0.004,10.35,156.289]]]' run free common 0.01 max 71 1 400
BG=1.0 run monotone common 0.01 max 72 1 400
echo DONE_CAL >> $OUT/calib.jsonl
