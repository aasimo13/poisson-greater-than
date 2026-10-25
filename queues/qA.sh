#!/bin/bash
cd "$(dirname "$0")/.."
OUT=${OUT:-results/rerun}; mkdir -p "$OUT"
run(){ python3 src/opt.py "$@" >> $OUT/bgdrift.jsonl 2>> $OUT/bgdrift.err; }
BG=1.0 run free nominal 0 max 21 2 400
BG=0.1 EXTRA='[[[104.788,4.691,0.001,29.867],[29.865,0.001,4.691,104.755]]]' run free common 0.02 max 22 3 400
BG=1.0 EXTRA='[[[104.788,4.691,0.001,29.867],[29.865,0.001,4.691,104.755]]]' run free common 0.02 max 23 3 400
BG=0.1 run monotone common 0.02 max 24 1 400
BG=1.0 run monotone common 0.02 max 25 1 400
echo DONE_A >> $OUT/bgdrift.jsonl
