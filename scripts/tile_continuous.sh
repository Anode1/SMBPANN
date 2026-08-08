#!/bin/sh
# Continuous objective (acc - lambda*energy) runs. Waits for any in-flight emerge_tile jobs first.
cd "$(dirname "$0")/.."
# -x matches the process NAME exactly. Do NOT use `pgrep -f`: it matches whole command lines, so any
# shell whose command line merely mentions the probe -- an echo, a comment, this script's own
# launcher -- counts as a running job and the loop waits on itself forever.
# Use pgrep's EXIT STATUS (0 = something matched), not its printed count: `pgrep -c` prints 0 and
# also exits 1 when nothing matches, so a `|| echo 0` fallback emits two lines and the integer test
# becomes invalid.
while pgrep -x emerge_tile >/dev/null 2>&1; do sleep 30; done

# (a) lambda sweep at N=12, E_param arm, to pick an operating point
for L in 0.02 0.05 0.10 0.20 0.40; do
    SEEDS=200 NIN=12 GENS=80 FAST=1 OBJ=1 LAMBDA="$L" ./emerge_tile > "scratch_cont_lam_$L.out" 2>&1 &
done
wait

# (b) N sweep at lambda=0.10, no target anywhere in the loop
for n in 8 12 16 20 24; do
    SEEDS=200 NIN=$n GENS=80 FAST=1 OBJ=1 LAMBDA=0.10 ./emerge_tile > "scratch_cont_N$n.out" 2>&1 &
done
wait

# (c) full table at N=12: all three mutation arms x all three energy terms, with genome dump
SEEDS=200 NIN=12 GENS=80 OBJ=1 LAMBDA=0.10 DUMP=1 ./emerge_tile > scratch_cont_main.out 2>&1

echo "done: $(date)" > scratch_cont_DONE
