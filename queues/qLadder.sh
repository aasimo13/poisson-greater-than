#!/bin/bash
# Threshold (monotone) and banded (free) designs at background 0.1, 1, 3, nominal gain, 1% error. -> $OUT/ladder_opt.jsonl
cd "$(dirname "$0")/.."
OUT=${OUT:-results}; mkdir -p "$OUT"
for bg in 0.1 1.0 3.0; do
  BG=$bg python3 src/opt.py monotone nominal 0 max 1 1 600 | grep -v '^BEST' >> $OUT/ladder_opt.jsonl
  BG=$bg python3 src/opt.py free nominal 0 max 1 1 600 | grep -v '^BEST' >> $OUT/ladder_opt.jsonl
done
