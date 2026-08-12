#!/bin/sh
# space_sweep.sh -- does directed search earn its keep once random search cannot cover the space?
#
# On NAS-Bench-101 matched-budget random selection beats our GA at every budget tried, but that
# benchmark holds 423,624 architectures while our largest budget draws 76,800, so random search samples
# about a fifth of the whole space. The result is a property of the benchmark, not a fact about search.
#
# Here the support space is 2^NOFF times the sharing gene, with NOFF = 2N-3, so it grows exponentially
# while an evaluation stays cheap:
#     N=12  NOFF=21  ~4.2e6 supports      budget/space  ~3e-4
#     N=16  NOFF=29  ~1.1e9               ~1e-6
#     N=20  NOFF=37  ~2.7e11              ~4e-9
#     N=24  NOFF=45  ~7.0e13              ~2e-11
# The prediction under test: the GA's advantage over matched-budget random search grows with N, and is
# absent or negative at N=12 where random search still covers a meaningful fraction.
#
# Both arms get POP*GENS fitness evaluations, drawn from the same distribution and scored by the same
# fitness, so the comparison is matched on compute and on candidate source.
cd "$(dirname "$0")/.." || exit 1
S=${SEEDS:-12}
for NN in 12 16 20 24; do
  SEEDS=$S GENS=50 POP=24 LAMBDA=6 ROTPOS=1 FEVAL=1 \
    ./et_n${NN} > scratch_space_n${NN}.out 2>&1 &
done
wait
echo "space sweep: GA flip-only vs matched-budget RANDOM SEARCH, objective (higher better)"
printf "%-4s %-10s %-12s %-12s %-10s\n" N supports GA_flip RANDOM GA-RANDOM
for NN in 12 16 20 24; do
  awk -v N=$NN '/GA flip-only/{g=$4} /RANDOM SEARCH/{r=$5}
    END{ noff=2*N-3; printf "%-4s 2^%-8s %-12s %-12s %+.4f\n", N, noff, g, r, g-r }' scratch_space_n${NN}.out
done
