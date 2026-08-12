#!/bin/sh
# space_sweep2.sh -- the control the first space sweep needs: budget growing with the space.
#
# The first sweep held the budget at 1200 evaluations while the space grew from 2^21 to 2^45, so the
# GA's narrowing advantage over random search cannot be separated from "the same compute now buys less
# of a bigger problem". Paper 1 met the same objection in its scaling section and answered it by growing
# the budget quadratically in N; this repo's own notes call that the blocking experiment. GENS is
# therefore scaled as N^2, normalised so N=12 keeps GENS=50.
#     N=12 -> 50    N=16 -> 89    N=20 -> 139    N=24 -> 200
# Both arms still receive identical budget at each N, so the GA-vs-random comparison stays matched.
cd "$(dirname "$0")/.." || exit 1
S=${SEEDS:-12}
set -- "12 50" "16 89" "20 139" "24 200"
for pair in "$@"; do
  NN=`echo $pair | cut -d' ' -f1`; GG=`echo $pair | cut -d' ' -f2`
  SEEDS=$S GENS=$GG POP=24 LAMBDA=6 ROTPOS=1 FEVAL=1 \
    ./et_n${NN} > scratch_space2_n${NN}.out 2>&1 &
done
wait
echo "budget ~ N^2: objective, higher better"
printf "%-5s %-8s %-7s %-9s %-9s %-9s %s\n" N space GENS ideal GA RANDOM "GA-RND"
for pair in "$@"; do
  NN=`echo $pair | cut -d' ' -f1`; GG=`echo $pair | cut -d' ' -f2`
  awk -v N=$NN -v G=$GG '/^  ideal conv/{i=$NF} /^  GA flip-only/{g=$NF} /^  RANDOM SEARCH/{r=$NF}
    END{printf "%-5s 2^%-6s %-7s %-9s %-9s %-9s %+.4f\n", N, 2*N-3, G, i, g, r, g-r}' scratch_space2_n${NN}.out
done
