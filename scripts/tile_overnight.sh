#!/bin/sh
# tile_overnight.sh -- the long emerge_tile runs, for statistics the pilot cannot give.
# Run from the repo root:   nohup sh scripts/tile_overnight.sh > /dev/null 2>&1 &
# Everything lands in scratch_tile_*.out; nothing here modifies an existing experiment.

set -e
cd "$(dirname "$0")/.."
make emerge_tile

# 1. The headline at 200 seeds. The 40-seed run gave 0/40 vs 20/40 (Fisher p=7.8e-08);
#    this pins the effect size and gives every arm a usable error bar.
SEEDS=200 GENS=80 TARGET=0.90 DUMP=1 ./emerge_tile > scratch_tile_main200.out 2>&1

# 2. Is 50% a peak or a plateau? The tiling rate was picked, never swept.
#    PTILE is read by the (2) sweep; one file per rate keeps them comparable.
for r in 0.01 0.02 0.05 0.10 0.20; do
    SEEDS=100 GENS=80 TARGET=0.90 PTILE="$r" ./emerge_tile > "scratch_tile_rate_$r.out" 2>&1
done

# 3. Does the per-neuron energy normalisation hold as the network grows?
#    NMAX is 24, so 8..24. If the result is scale-stable, %conv should not collapse with N.
for n in 8 12 16 20 24; do
    SEEDS=100 GENS=80 TARGET=0.90 NIN="$n" ./emerge_tile > "scratch_tile_N$n.out" 2>&1
done

# 4. Longer evolution: does local mutation EVER reach a tiling given far more generations?
#    This is the honest control on the central negative. If 400 generations still gives 0%,
#    the claim is about reachability, not about budget.
SEEDS=60 GENS=400 TARGET=0.90 ./emerge_tile > scratch_tile_long400.out 2>&1

echo "done: $(date)" > scratch_tile_OVERNIGHT_DONE
