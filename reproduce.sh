#!/usr/bin/env bash
# Reproduce the headline numbers.   ./reproduce.sh quick   (about 3 minutes)
#                                   ./reproduce.sh full    (lists the long jobs; add --run to run them)
# Run from anywhere; everything is run from the repo root. Set NICE="" to drop the nice wrapper.
set -u
cd "$(dirname "$0")"
NICE=${NICE-"nice -n 15"}
PY="$NICE python3"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
PASS=0; FAIL=0
ROWS=()

record() {   # label got expected status
  ROWS+=("$(printf '%-60s %-14s %-14s %s' "$1" "$2" "$3" "$4")")
  if [ "$4" = PASS ]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
}
check() {    # label got expected tolerance   (|got - expected| <= tol)
  if [ -n "$2" ] && awk -v g="$2" -v e="$3" -v t="$4" 'BEGIN{d=g-e; if(d<0)d=-d; exit !(d<=t)}'; then
    record "$1" "$2" "$3" PASS; else record "$1" "${2:-none}" "$3" FAIL; fi
}
check_ge() { # label got threshold   (got >= threshold)
  if [ -n "$2" ] && awk -v g="$2" -v e="$3" 'BEGIN{exit !(g>=e)}'; then
    record "$1" "$2" ">= $3" PASS; else record "$1" "${2:-none}" ">= $3" FAIL; fi
}
show() { sed 's/^/    | /' | cut -c1-170; }

quick() {
  local t0=$SECONDS
  echo "== 1. verify_candidate.py (201 photon candidate, scipy and mpmath)"
  $PY src/verify_candidate.py > "$TMP/cand.txt" 2>&1; show < "$TMP/cand.txt"
  check "candidate worst error % (scipy), pair (2,0)" \
    "$(sed -n 's/^WORST.*scipy \([0-9.]*\)%.*/\1/p' "$TMP/cand.txt")" 0.959542 0.0000005
  check "candidate worst error % (mpmath 50 digits)" \
    "$(sed -n 's/^WORST.*mpmath \([0-9.]*\)%.*/\1/p' "$TMP/cand.txt")" 0.959542 0.0000005
  check "candidate max mean" "$(sed -n 's/^max mean = \([0-9.]*\)/\1/p' "$TMP/cand.txt")" 201 0.05
  check "LP decoder on same encoding, worst error %" \
    "$(sed -n 's/^LP (randomized).*error \([0-9.]*\)%.*/\1/p' "$TMP/cand.txt")" 0.9178 0.00005
  check "budget after LP rescaling (exact value 196.70)" \
    "$(sed -n 's/^Budget after rescaling.*: \([0-9.]*\) (scale.*/\1/p' "$TMP/cand.txt")" 196.70 0.015

  echo "== 2. threshold_optimum.py (K=4)"
  $PY src/threshold_optimum.py > "$TMP/thr.txt" 2>&1; show < "$TMP/thr.txt"
  check "threshold optimum, low counts true" \
    "$(sed -n 's/^K=4 low counts mean true: min P = \([0-9.]*\) .*/\1/p' "$TMP/thr.txt")" 321.540 0.0005
  check "threshold optimum, high counts true" \
    "$(sed -n 's/^K=4 high counts mean true: min P = \([0-9.]*\) .*/\1/p' "$TMP/thr.txt")" 452.319 0.0005
  check_ge "certified lower bound, any threshold design" \
    "$(sed -n 's/^Certified lower bound.*P >= \([0-9.]*\)$/\1/p' "$TMP/thr.txt")" 321.5366
  check "LP cross check on the corner encoder" \
    "$(sed -n 's/^Cross check:.* gives \([0-9.]*\) (.*/\1/p' "$TMP/thr.txt")" 321.540 0.0005

  echo "== 3. banded.py (K = 2..8)"
  $PY src/banded.py 8 > "$TMP/band.txt" 2>&1; show < "$TMP/band.txt"
  check "band rule violations, K=2..8 (sum)" \
    "$(awk '$1 ~ /^[0-9]+$/ && NF>=5 {s+=$2} END{print s+0}' "$TMP/band.txt")" 0 0
  check "band pairwise budget K=4" "$(awk '$1==4 && NF>=5 {print $3}' "$TMP/band.txt")" 196.33 0.005
  for kv in 3:57.56 4:196.62 5:534.62 6:1295.39 7:2936.78 8:6389.06; do
    k=${kv%%:*}; e=${kv##*:}
    check "band exact LP budget K=$k" "$(awk -v k=$k '$1==k && NF>=5 {print $4}' "$TMP/band.txt")" "$e" 0.005
  done

  echo "== 4. lb_pairwise2.py 3 60 120 4 (certified lower bound, K=3; about 1 to 2 minutes)"
  $PY src/lb_pairwise2.py 3 60 120 4 > "$TMP/lb3.txt" 2>&1; cut -c1-150 "$TMP/lb3.txt" | show
  check "K=3 certified lower bound (rounds to 57.43)" \
    "$(tail -n 1 "$TMP/lb3.txt" | python3 -c 'import sys,json; print(round(json.loads(sys.stdin.read())["best_lb"],4))' 2>/dev/null)" 57.43 0.005

  echo "== 5. gaussian_milp.py (K = 3, 4)"
  $PY src/gaussian_milp.py 4 > "$TMP/gau.txt" 2>&1; show < "$TMP/gau.txt"
  check "Gaussian MILP optimum K=3, in units of d" "$(awk '$1==3 && NF>5 {print $3}' "$TMP/gau.txt")" 4 0.0005
  check "Gaussian MILP optimum K=4, in units of d" "$(awk '$1==4 && NF>5 {print $3}' "$TMP/gau.txt")" 6 0.0005
  check "Gaussian band chain K=4, in units of d" "$(awk '$1==4 && NF>5 {print $7}' "$TMP/gau.txt")" 14 0.0005

  echo "== 6. make_hgrid.py (regenerate both h tables, compare with data/)"
  $PY src/make_hgrid.py > "$TMP/hg.txt" 2>&1; show < "$TMP/hg.txt"
  check "h tables identical to shipped files (1 = yes)" "$(grep -c '^ALL MATCH' "$TMP/hg.txt")" 1 0

  echo
  echo "============ SUMMARY ============"
  printf '%-60s %-14s %-14s %s\n' check got expected status
  printf '%s\n' "${ROWS[@]}"
  echo "$PASS passed, $FAIL failed, $((SECONDS - t0)) s"
  [ "$FAIL" -eq 0 ]
}

full() {
  local OUT=${OUT:-results/rerun}
  local run=0; [ "${1:-}" = "--run" ] && run=1
  local -a CMDS=(
    "python3 src/lb_pairwise2.py 4 200 1500 8                                       # K=4 certified bound 195.53 (about 25 min per solve stage, hours in total)"
    "USE_HIGHS=1 MONO=1 HGRID=data/hgrid_ext.npy python3 src/lb_pairwise2.py 4 330 1500 3   # monotone encoders: 320.32"
    "USE_HIGHS=1 OBJ=avg HGRID=data/hgrid_ext.npy python3 src/lb_pairwise2.py 4 400 2400 2  # average photon cost: 65.50"
    "python3 src/opt.py free common 0.05 max 11 1 400                                # banded under +-5% gain drift: 245.4"
    "python3 src/opt.py monotone common 0.05 max 12 1 400                           # monotone under drift: 598.2"
    "python3 src/opt.py free nominal 0 avg 7 4 500                                   # average photon objective: 65.8"
    "python3 src/gaussian_milp.py 5                                                  # K=5 Gaussian MILP: 8d (about 1 to 2 min)"
    "python3 src/banded.py 8"
    "python3 src/threshold_optimum.py table                                          # K=3..10 table (about 1 min)"
    "python3 src/check_bgdrift.py                                                    # refit and test decoders for results/bgdrift.jsonl"
    "bash queues/qA.sh; bash queues/qB.sh; bash queues/qC.sh                         # drift plus background runs -> \$OUT/bgdrift.jsonl"
    "bash queues/qCal.sh                                                             # drift calibration runs -> \$OUT/calib.jsonl"
    "bash queues/qEps.sh                                                             # other error targets -> \$OUT/eps.jsonl"
    "bash queues/queue.sh; bash queues/queue2.sh                                     # first round opt.py queues -> \$OUT/opt.jsonl"
  )
  if [ $run -eq 0 ]; then
    echo "Long jobs (run './reproduce.sh full --run' to execute them one after another, hours in total)."
    echo "Raw outputs of the original runs are in results/. Queue scripts write to \$OUT (default results/rerun)."
    echo "Run with:  nice -n 15 <command>   from the repo root."
    echo
    printf '  %s\n' "${CMDS[@]}"
    return 0
  fi
  mkdir -p "$OUT"
  for c in "${CMDS[@]}"; do
    cmd=${c%%#*}
    echo ">>> $cmd"
    case "$cmd" in
      *queues/*) OUT=$OUT bash -c "$NICE bash -c '$cmd'" ;;
      *lb_pairwise2*) bash -c "$NICE bash -c '$cmd'" | tee -a "$OUT/lb_runs.jsonl" ;;
      *) bash -c "$NICE bash -c '$cmd'" | tee -a "$OUT/full_runs.txt" ;;
    esac
  done
}

case "${1:-}" in
  quick) quick ;;
  full) shift; full "$@" ;;
  *) echo "usage: $0 quick | full [--run]"; exit 2 ;;
esac
