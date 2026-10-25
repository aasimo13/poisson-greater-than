#!/bin/bash
cd "$(dirname "$0")/.."
OUT=${OUT:-results/rerun}; mkdir -p "$OUT"
run(){ python3 src/opt.py "$@" >> $OUT/bgdrift.jsonl 2>> $OUT/bgdrift.err; }
BG=3.0 EXTRA='[[[174.646,10.678,0.004,52.947],[52.954,0.003,10.678,174.61]]]' run free common 0.05 max 31 2 400
BG=3.0 run monotone common 0.05 max 32 1 400
echo DONE_C >> $OUT/bgdrift.jsonl
