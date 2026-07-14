#!/bin/sh
#
# inherit_sweep.sh -- map where weight inheritance helps: an epochs x generations
# sweep of the inherit_ab.sh A/B, sharded one cell per core.
#
# Inheritance's benefit is budget-dependent: warm-starting saves training that a
# from-scratch search discards, so the gain should be largest at a LOW per-candidate
# epoch budget and fade as candidates converge on their own. This runs a grid of
# (epochs, generations) cells in parallel, one process per core (JOBS=1 each so the
# cells sum to the core count), each an independent paired A/B over SEEDS seeds.
#
# Env: SEEDS (seeds per cell), plus the task knobs (DIM N FREQ NOISE POP LMAX WMAX MUT).
# Usage: scripts/inherit_sweep.sh    (writes per-cell logs under $DIR, prints a grid)
set -eu

SEEDS="${SEEDS:-200}"
DIR="${DIR:-/tmp/claude-1000/sweep}"
mkdir -p "$DIR"
: > "$DIR/cells.list"

printf 'inheritance sweep: epochs {20,40,80,160} x gens {8,20}, %s seeds/cell, one core per cell\n' "$SEEDS"

for ep in 20 40 80 160; do
    for gn in 8 20; do
        tag="e${ep}_g${gn}"
        echo "$tag" >> "$DIR/cells.list"
        JOBS=1 LOG="$DIR/$tag.log" RUNS="$SEEDS" EPOCHS="$ep" GENS="$gn" \
            DIM="${DIM:-6}" N="${N:-300}" FREQ="${FREQ:-2}" NOISE="${NOISE:-3}" \
            POP="${POP:-10}" ELITE="${ELITE:-2}" LMAX="${LMAX:-3}" WMAX="${WMAX:-12}" MUT="${MUT:-1}" \
            scripts/inherit_ab.sh > "$DIR/$tag.out" 2>&1 &
    done
done
wait

# aggregate each cell's paired result into a grid
printf '\n%-10s %8s %8s %8s %8s\n' "cell" "scratch" "inherit" "win/N" "p"
for tag in $(cat "$DIR/cells.list"); do
    awk -v tag="$tag" '
      {s+=$2; h+=$3; n++; if($3<$2)w++; else if($2<$3)l++}
      END{
        if(n==0){ printf "%-10s   (no data)\n", tag }
        else {
          dec=w+l; k=(w>l)?w:l; p=0;   # exact two-sided sign test on decisive pairs
          for(i=k;i<=dec;i++){
            lc=0; for(j=1;j<=dec;j++) lc+=log(j);
            for(j=1;j<=i;j++) lc-=log(j); for(j=1;j<=dec-i;j++) lc-=log(j);
            p+=exp(lc)
          }
          p=(dec>0)?2*p/(2^dec):1;
          printf "%-10s %8.4f %8.4f %6d/%d %8.3f\n", tag, s/n, h/n, w+0, n, p
        }
      }' "$DIR/$tag.log"
done
