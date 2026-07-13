#!/bin/sh
#
# errortest_multi.sh -- fixed-task time-to-target over several tasks (heterogeneity).
#
# The fresh-task benchmark (errortest.sh, default mode) already pairs the two methods
# on the same task within each run, so its head-to-head is a fair search-efficiency
# comparison averaged over tasks. This script asks the finer question: on a SINGLE
# task, held fixed while only the search seed varies, which method reaches the target
# sooner -- and does the answer depend on the task? It fixes each of NTASKS tasks in
# turn and races the self-modifying search against random over SEEDS search seeds,
# then reports the per-task head-to-head (the heterogeneity) and the pooled result.
#
# Env (all optional): NTASKS SEEDS DIM N FREQ NOISE TARGET POP GENS EPOCHS
#                     LMAX WMAX MUT ADAPT ELITE
#
# Usage: scripts/errortest_multi.sh
#        NTASKS=8 SEEDS=50 TARGET=0.17 scripts/errortest_multi.sh
#
set -eu

NTASKS="${NTASKS:-8}"; SEEDS="${SEEDS:-50}"
DIM="${DIM:-8}"; N="${N:-300}"; FREQ="${FREQ:-3}"; NOISE="${NOISE:-2}"
TARGET="${TARGET:-0.17}"
POP="${POP:-10}"; GENS="${GENS:-30}"; EPOCHS="${EPOCHS:-1000}"
ELITE="${ELITE:-2}"; LMAX="${LMAX:-3}"; WMAX="${WMAX:-16}"
MUT="${MUT:-2}"; ADAPT="${ADAPT:-1}"

task=$(mktemp)
res="${LOG:-$(mktemp)}"                  # durable per-run log: task seed ga rand
: > "$res"
trap 'rm -f "$task"' EXIT

printf 'multi-task fixed-task time-to-target: %s tasks x %s search seeds = %s runs | target=%s dim=%s N=%s freq=%s noise=%s%% space<=%sx%s pop=%s cap=%s epochs=%s mut=%s adapt=%s\n\n' \
    "$NTASKS" "$SEEDS" "$((NTASKS*SEEDS))" "$TARGET" "$DIM" "$N" "$FREQ" "$NOISE" "$LMAX" "$WMAX" "$POP" "$GENS" "$EPOCHS" "$MUT" "$ADAPT"

t=1
while [ "$t" -le "$NTASKS" ]; do
    ./gentask -d "$DIM" -N "$N" -f "$FREQ" -e "$NOISE" -s "$t" > "$task"
    s=1
    while [ "$s" -le "$SEEDS" ]; do
        line=$(COMMON="-f $task -i $DIM -o 1 -e $EPOCHS" \
               ./evolve -i "$DIM" -o 1 -P "$POP" -G "$GENS" -k "$ELITE" -L "$LMAX" -W "$WMAX" -M "$MUT" -A "$ADAPT" -E "$TARGET" -s "$s" 2>/dev/null \
               | grep '^TARGET')
        gg=$(printf '%s\n' "$line" | sed -n 's/.*ga_gens=\([0-9]*\).*/\1/p')
        rg=$(printf '%s\n' "$line" | sed -n 's/.*rand_gens=\([0-9]*\).*/\1/p')
        printf '%d %d %s %s\n' "$t" "$s" "$gg" "$rg" >> "$res"   # durable, flushed
        s=$((s + 1))
    done
    # per-task head-to-head, printed as each task completes
    awk -v tk="$t" '$1==tk{g=$3;r=$4;
        if(g>0&&r>0){if(g<r)gw++;else if(r<g)rw++;else ti++} else if(g>0)gw++; else if(r>0)rw++; else nn++}
        END{printf "task %2d:  GA %2d  RAND %2d  tie %2d  neither %2d\n", tk, gw+0,rw+0,ti+0,nn+0}' "$res"
    t=$((t + 1))
done

# pooled summary across all tasks
awk '{g=$3;r=$4;
  if(g>0)ga++; if(r>0)ra++;
  if(g>0&&r>0){if(g<r)gw++;else if(r<g)rw++;else ti++} else if(g>0)gw++; else if(r>0)rw++; else nn++}
  END{
    printf "\n--- pooled over %d runs ---\n", NR;
    printf "reached target: GA %d, RAND %d\n", ga+0, ra+0;
    printf "head-to-head (fewer gens): GA %d, RAND %d, tie %d, neither %d  (decisive %d)\n",
        gw+0, rw+0, ti+0, nn+0, gw+rw;
}' "$res"
