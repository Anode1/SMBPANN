#!/bin/sh
# overnight_nb101b.sh -- the definitive E2 sweep: standard errors on every metric, the GENS axis
# extended past the crossover, and the POP=96 cell that the first sweep lost to an unchecked bound.
#
# The first sweep established the shape (PREREG_nb101.md P6): rotation's advantage over a fixed
# selection replicate GROWS with generation count (-0.006, +0.004, +0.048, +0.085, +0.133 at
# 10/20/50/100/200) while averaging's advantage over rotation SHRINKS and reverses by 200 generations.
# That is the accumulation mechanism the toy probe predicted, so this run (a) attaches standard errors
# to quality and regret, which carry the claim and previously had none, (b) adds GENS=400 to check the
# crossover is not the end of the trend, and (c) reproduces the first sweep exactly, since the RNG
# stream is unchanged -- making it a free reproducibility check on all five original cells.
set -e
cd "$(dirname "$0")/.."
T=validation/nasbench101_trials.txt
echo "=== generation sweep, POP=24"
for G in 10 20 50 100 200; do
  POP=24 GENS=$G RUNS=20000 ./nb101_search $T > scratch_nb2_g${G}.out 2>&1 &
done
POP=24 GENS=400 RUNS=10000 ./nb101_search $T > scratch_nb2_g400.out 2>&1 &
echo "=== population sweep, GENS=50"
for P in 8 96; do
  POP=$P GENS=50 RUNS=20000 ./nb101_search $T > scratch_nb2_p${P}.out 2>&1 &
done
wait
echo "=== results ==="
for f in scratch_nb2_g10 scratch_nb2_g20 scratch_nb2_g50 scratch_nb2_g100 scratch_nb2_g200 \
         scratch_nb2_g400 scratch_nb2_p8 scratch_nb2_p96; do
  printf -- "--- %s\n" "${f#scratch_nb2_}"
  grep -E 'FIXED  \(|ROTATE \(|AVG2   \(' $f.out
done
echo "=== reproducibility check against the first sweep (means must match exactly) ==="
for G in 10 20 50 100 200; do
  a=$(awk '/FIXED  \(/{print $(NF-2)}' scratch_nb101_g${G}.out)
  b=$(awk '/FIXED  \(/{print $(NF-2)}' scratch_nb2_g${G}.out | cut -d+ -f1)
  printf "  GENS=%-4s first %s  rerun %s  %s\n" "$G" "$a" "$b" \
         "$([ "$a" = "$b" ] && echo MATCH || echo DIFFER)"
done
echo "=== done"
