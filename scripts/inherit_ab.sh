#!/bin/sh
#
# inherit_ab.sh -- a paired A/B of weight inheritance against training from scratch.
#
# At a low per-candidate epoch budget, warm-starting a child from its parent's
# trained weights lets a surviving lineage keep improving instead of discarding its
# training each generation. This does not spend more compute (the same number of
# evaluations, the same epochs each); it just stops throwing training away. For each
# seed a fresh task is generated and the self-modifying search is run twice on it,
# identically except for weight inheritance (evolve -w 0 vs -w 1), for a fixed
# generation budget. We record each run's final GA validation error and report the
# paired comparison: does inheritance reach a lower error at equal compute?
#
# The random control draws fresh architectures every generation and cannot inherit,
# so it is identical in both arms; we print it too, as a baseline and a plumbing
# check (rand_off must equal rand_on).
#
# Env (all optional): RUNS DIM N FREQ NOISE POP GENS EPOCHS LMAX WMAX MUT ELITE
#
# Usage: scripts/inherit_ab.sh
#        RUNS=40 EPOCHS=120 GENS=10 scripts/inherit_ab.sh
#
set -eu

RUNS="${RUNS:-40}"
DIM="${DIM:-6}"; N="${N:-300}"; FREQ="${FREQ:-2}"; NOISE="${NOISE:-3}"
POP="${POP:-10}"; GENS="${GENS:-10}"; EPOCHS="${EPOCHS:-120}"
ELITE="${ELITE:-2}"; LMAX="${LMAX:-3}"; WMAX="${WMAX:-12}"; MUT="${MUT:-1}"

task=$(mktemp)
res="${LOG:-$(mktemp)}"
: > "$res"
trap 'rm -f "$task"' EXIT

args="-i $DIM -o 1 -P $POP -G $GENS -k $ELITE -L $LMAX -W $WMAX -M $MUT -A 1"
ga_of() { printf '%s\n' "$1" | awk '{for(i=1;i<=NF;i++) if($i=="GA") print $(i+1)}'; }
rd_of() { printf '%s\n' "$1" | awk '{for(i=1;i<=NF;i++) if($i=="RAND") print $(i+1)}'; }

printf 'weight-inheritance A/B: from-scratch (-w 0) vs inherit (-w 1) | %s runs, fresh task each | dim=%s N=%s freq=%s noise=%s%% pop=%s gens=%s epochs=%s (budget %s evals)\n\n' \
    "$RUNS" "$DIM" "$N" "$FREQ" "$NOISE" "$POP" "$GENS" "$EPOCHS" "$((POP*GENS))"

run=1
while [ "$run" -le "$RUNS" ]; do
    ./gentask -d "$DIM" -N "$N" -f "$FREQ" -e "$NOISE" -s "$run" > "$task"
    f0=$(COMMON="-f $task -i $DIM -o 1 -e $EPOCHS" ./evolve $args -w 0 -s "$run" 2>/dev/null | grep '^final')
    f1=$(COMMON="-f $task -i $DIM -o 1 -e $EPOCHS" ./evolve $args -w 1 -s "$run" 2>/dev/null | grep '^final')
    g0=$(ga_of "$f0"); g1=$(ga_of "$f1"); r0=$(rd_of "$f0"); r1=$(rd_of "$f1")
    printf '%d %s %s %s %s\n' "$run" "$g0" "$g1" "$r0" "$r1" >> "$res"
    printf 'run %3d:  scratch GA=%-11s  inherit GA=%-11s  rand=%s\n' "$run" "$g0" "$g1" "$r0"
    run=$((run + 1))
done

awk '{
  s=$2; h=$3; r0=$4; r1=$5;
  n++; ss+=s; hs+=h; rr+=r0;
  if (h < s) win++; else if (s < h) lose++; else tie++;
  if (r0 != r1) badrand++;
}
END{
  printf "\n--- weight inheritance over %d paired runs ---\n", n;
  printf "mean final GA error:  from-scratch %.4f,  inherit %.4f,  random %.4f\n", ss/n, hs/n, rr/n;
  printf "inheritance reached lower error in %d/%d runs (worse %d, tie %d)\n", win+0, n, lose+0, tie+0;
  printf "plumbing check (random identical in both arms): %s\n", (badrand?"FAILED "badrand" runs":"ok");
}' "$res"
