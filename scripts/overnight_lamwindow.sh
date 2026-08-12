#!/bin/sh
# overnight_lamwindow.sh -- job 3: was LAMBDA=6 cherry-picked?
#
# The headline uses LAMBDA=6, chosen because it had the best exact-match rate of {1,3,6,12}. That is
# the choice a referee challenges first, and PROTOCOL item 3 says sweep the load-bearing constants.
# Two questions, in order:
#
#   (a) OVER WHICH LAMBDA IS THE TARGET ACTUALLY THE OPTIMUM? Resolved gates (DRAWS=12, 60 seeds, so
#       per-mask error 0.0048) across the window. We know rank 4 at LAMBDA=1 and rank 1 at 6; the
#       lower edge is somewhere between, and the upper edge is below 12 because at 12 the search
#       outscores the hand-built target. Reporting the window is the honest form of the claim.
#   (b) DOES THE EMERGENCE RATE TRACK THAT WINDOW? Exact-match rates at the neighbouring tariffs. If
#       the rate is a plateau over the window rather than a spike at 6, the result is not a tuned cell.
set -e
cd "$(dirname "$0")/.."
echo "=== job 3a: resolved gates across the lambda window, 60 seeds x 12 draws"
for L in 2 3 4 8 10; do
  SUBSCAN=1 SUBW=7 SEEDS=60 DRAWS=12 LAMBDA=$L \
    ./emerge_transfer > scratch_gate_lam${L}.out 2>&1 &
done
wait
for L in 2 3 4 8 10; do
  printf "    lambda %-3s " $L
  grep -h 'RANK\|margin over' scratch_gate_lam${L}.out | tr '\n' ' ' | sed 's/SUBSCAN//g'
  echo ""
done
echo "=== job 3b: exact-match rate at the neighbouring tariffs, 60 seeds each, fixes on"
for L in 3 4 8; do
  for s in 0 30; do
    RAW=1 ROTPOS=1 FEVAL=12 LAMBDA=$L SEEDS=30 SEED0=$s \
      ./emerge_transfer > scratch_rate_lam${L}_s${s}.out 2>&1 &
  done
done
wait
for L in 3 4 8; do
  cat scratch_rate_lam${L}_s*.out | awk -v L=$L '/^RAW/ && $2==1 {n++; t+=$4; if($4==3 && $6==1) e++}
    END{printf "    lambda %-3s exact %2d/%d (%.0f%%)  mean taps %.2f\n", L, e, n, 100*e/n, t/n}'
done
echo "=== job 3 done"
