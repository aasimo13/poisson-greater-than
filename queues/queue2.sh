#!/bin/bash
cd "$(dirname "$0")/.."
OUT=${OUT:-results/rerun}; mkdir -p "$OUT"
run(){ python3 src/opt.py "$@" >> $OUT/opt.jsonl 2>> $OUT/opt.err; }
run free common 0.02 max 13 1 400
run monotone common 0.02 max 14 1 400
run free common 0.10 max 15 1 400
run monotone common 0.10 max 16 1 400
run free indep 0.05 max 17 1 400
run monotone indep 0.05 max 18 1 400
echo DONE >> $OUT/opt.jsonl
