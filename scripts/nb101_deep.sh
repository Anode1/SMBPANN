#!/bin/sh
# nb101_deep.sh -- push the generation axis far past the crossover.
# The finding worth resolving: rotation's advantage over a fixed selection replicate GROWS with search
# depth while averaging's is flat, so the two cross. The first sweep saw the crossover at ~200
# generations; this spans 10 to 3200 to show whether it holds or turns over. Cheap: ~17us per
# generation-run, so the whole sweep is under an hour wall-clock at 20k runs per arm.
set -e
cd "$(dirname "$0")/.."
T=validation/nasbench101_trials.txt
for G in 10 20 50 100 200 400 800 1600 3200; do
  POP=24 GENS=$G RUNS=20000 ./nb101_search $T > scratch_nbdeep_g${G}.out 2>&1 &
done
wait
echo "GENS  FIXED_q  ROTATE_q  AVG2_q   ROT-FIX   ROT-AVG2   (quality, +-SE in files)"
for G in 10 20 50 100 200 400 800 1600 3200; do
  awk -v G=$G '/FIXED  \(/{split($5,a,"+");f=a[1]} /ROTATE \(/{split($6,b,"+");r=b[1]}
       /AVG2   \(/{split($6,c,"+");v=c[1]}
       END{printf "%-5s %8.4f %9.4f %8.4f %+9.4f %+10.4f\n", G, f, r, v, r-f, r-v}' scratch_nbdeep_g${G}.out
done
