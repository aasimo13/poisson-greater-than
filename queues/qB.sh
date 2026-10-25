#!/bin/bash
cd "$(dirname "$0")/.."
OUT=${OUT:-results/rerun}; mkdir -p "$OUT"
run(){ python3 src/opt.py "$@" >> $OUT/bgdrift.jsonl 2>> $OUT/bgdrift.err; }
BG=0.1 EXTRA='[[[116.664,4.84,0.001,31.985],[31.997,0.001,4.843,117.043]]]' run free common 0.05 max 26 3 400
BG=1.0 EXTRA='[[[116.664,4.84,0.001,31.985],[31.997,0.001,4.843,117.043]]]' run free common 0.05 max 27 3 400
BG=0.1 run monotone common 0.05 max 28 1 400
BG=1.0 run monotone common 0.05 max 29 1 400
echo DONE_B >> $OUT/bgdrift.jsonl
