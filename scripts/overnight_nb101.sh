#!/bin/sh
# overnight_nb101.sh -- E2: does ROTATION help on real NAS-Bench-101, and does its effect grow with
# the number of generations? See ../articles/smbpann2/PREREG_nb101.md, predictions P6 and P7.
#
# The generation sweep is the decisive axis. Rotation removes a fixed selection sample that could be
# memorised ACROSS generations, so if that mechanism operates here its benefit must grow with
# generation count; if the replicate structure is instead pure iid noise, rotation is inert at every
# count and only averaging helps. A 50-run pilot pointed at the second, which is why the sweep spans
# a 20-fold range rather than testing one setting.
#
# Eight processes, one per cell, ~3.5 h wall (the GENS=200 cell dominates).
set -e
cd "$(dirname "$0")/.."
T=validation/nasbench101_trials.txt
R=${RUNS:-20000}
echo "=== generation sweep, POP=24, $R runs per arm"
for G in 10 20 50 100 200; do
  POP=24 GENS=$G RUNS=$R ./nb101_search $T > scratch_nb101_g${G}.out 2>&1 &
done
echo "=== population sweep, GENS=50, $R runs per arm"
for P in 8 48 96; do
  POP=$P GENS=50 RUNS=$R ./nb101_search $T > scratch_nb101_p${P}.out 2>&1 &
done
wait
echo "=== generation sweep results (arm: infl quality regret) ==="
for G in 10 20 50 100 200; do
  printf -- "--- GENS=%s\n" $G; grep -E 'FIXED|ROTATE|AVG2' scratch_nb101_g${G}.out
done
echo "=== population sweep results ==="
for P in 8 48 96; do
  printf -- "--- POP=%s\n" $P; grep -E 'FIXED|ROTATE|AVG2' scratch_nb101_p${P}.out
done
echo "=== done"
