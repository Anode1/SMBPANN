#!/bin/sh
#
# xover_ab.sh -- a paired A/B of crossover against mutation-only, time-to-target.
#
# For each seed a fresh task is generated, and the self-modifying search is run
# twice on it: once asexual (mutation only, evolve -X 0) and once sexual (a
# fraction XPCT of offspring made by crossing two elites, evolve -X XPCT). Both
# start from the identical initial population and see the identical task, split,
# and matched random control, so they differ only in whether offspring recombine.
# We record, per seed, the generations each variant needed to reach the target
# error, and report the paired head-to-head: does adding crossover reach the
# target sooner?
#
# Env (all optional): XPCT RUNS DIM N FREQ NOISE TARGET POP GENS EPOCHS
#                     LMAX WMAX MUT ADAPT ELITE
#
# Usage: scripts/xover_ab.sh
#        XPCT=100 RUNS=150 scripts/xover_ab.sh
#
set -eu

XPCT="${XPCT:-100}"                       # percent of offspring made by crossover
RUNS="${RUNS:-150}"
DIM="${DIM:-8}"; N="${N:-300}"; FREQ="${FREQ:-3}"; NOISE="${NOISE:-2}"
TARGET="${TARGET:-0.17}"
POP="${POP:-12}"; GENS="${GENS:-30}"; EPOCHS="${EPOCHS:-1000}"
ELITE="${ELITE:-4}"; LMAX="${LMAX:-3}"; WMAX="${WMAX:-16}"
MUT="${MUT:-2}"; ADAPT="${ADAPT:-1}"

task=$(mktemp)
res="${LOG:-$(mktemp)}"                    # per-run log: seed ga_mut ga_xover rand
: > "$res"
trap 'rm -f "$task"' EXIT

# evolve args shared by both variants (only -X differs)
common_args="-i $DIM -o 1 -P $POP -G $GENS -k $ELITE -L $LMAX -W $WMAX -M $MUT -A $ADAPT -E $TARGET"

# extract ga_gens / rand_gens from a run's TARGET line ($1=field name)
gens_of() { printf '%s\n' "$2" | sed -n "s/.*$1=\([0-9]*\).*/\1/p"; }

printf 'crossover A/B: mutation-only (X=0) vs crossover (X=%s%%) | %s runs, fresh task each | target=%s dim=%s N=%s freq=%s noise=%s%% pop=%s elite=%s cap=%s epochs=%s\n\n' \
    "$XPCT" "$RUNS" "$TARGET" "$DIM" "$N" "$FREQ" "$NOISE" "$POP" "$ELITE" "$GENS" "$EPOCHS"

run=1
while [ "$run" -le "$RUNS" ]; do
    ./gentask -d "$DIM" -N "$N" -f "$FREQ" -e "$NOISE" -s "$run" > "$task"
    lm=$(COMMON="-f $task -i $DIM -o 1 -e $EPOCHS" ./evolve $common_args -X 0     -s "$run" 2>/dev/null | grep '^TARGET')
    lx=$(COMMON="-f $task -i $DIM -o 1 -e $EPOCHS" ./evolve $common_args -X "$XPCT" -s "$run" 2>/dev/null | grep '^TARGET')
    gm=$(gens_of ga_gens "$lm"); gx=$(gens_of ga_gens "$lx"); rr=$(gens_of rand_gens "$lm")
    printf '%d %s %s %s\n' "$run" "$gm" "$gx" "$rr" >> "$res"
    ms=$([ "$gm" -gt 0 ] 2>/dev/null && echo "gen $gm" || echo "----")
    xs=$([ "$gx" -gt 0 ] 2>/dev/null && echo "gen $gx" || echo "----")
    printf 'run %3d:  mutation %-8s  crossover %-8s\n' "$run" "$ms" "$xs"
    run=$((run + 1))
done

awk '{
  m=$2; x=$3; r=$4;
  if(m>0)ma++; if(x>0)xa++; if(r>0)ra++;
  # paired crossover vs mutation (0 = never reached = worst)
  if(x>0&&m>0){ if(x<m)xw++; else if(m<x)mw++; else tie++ }
  else if(x>0)xw++; else if(m>0)mw++; else nn++;
}
END{
  printf "\n--- crossover A/B over %d paired runs ---\n", NR;
  printf "reached target:  mutation %d/%d,  crossover %d/%d,  random %d/%d\n", ma,NR, xa,NR, ra,NR;
  printf "crossover vs mutation (fewer gens to target): crossover %d, mutation %d, tie %d, neither %d  (decisive %d)\n",
      xw+0, mw+0, tie+0, nn+0, xw+mw;
}' "$res"
