#!/bin/bash
cd "$(dirname "$0")/.."
OUT=${OUT:-results/rerun}; mkdir -p "$OUT"
run(){ python3 src/opt.py "$@" >> $OUT/opt.jsonl 2>> $OUT/opt.err; }
run monotone nominal 0 avg 8 2 400
run free common 0.05 max 11 2 400
run monotone common 0.05 max 12 2 400
run free common 0.02 max 13 2 400
run monotone common 0.02 max 14 2 400
run free common 0.10 max 15 2 400
run monotone common 0.10 max 16 2 400
run free indep 0.05 max 17 2 400
run monotone indep 0.05 max 18 2 400
echo DONE >> $OUT/opt.jsonl
