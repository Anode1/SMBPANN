#!/bin/sh
#
# benchmark.sh -- a reproducible GA-versus-random benchmark (roadmap step 6).
#
# Generates one fixed synthetic classification task (where topology matters,
# unlike XOR), then runs the evolutionary search against its matched-compute
# random-search control over several seeds, and reports how often the GA wins
# and the mean test-MSE gap. Fully self-contained and reproducible: no downloads,
# no external benchmark files. (Real NAS-Bench-101/201 use a different, cell-based
# search space and ship as multi-gigabyte PyTorch files; bridging to them is
# future work -- see README.)
#
# Env (all optional): DIM N FREQ NOISE RUNS POP GENS EPOCHS
#
# Usage: scripts/benchmark.sh
#
set -eu

DIM="${DIM:-4}"; N="${N:-600}"; FREQ="${FREQ:-3}"; NOISE="${NOISE:-5}"
RUNS="${RUNS:-5}"; POP="${POP:-10}"; GENS="${GENS:-10}"; EPOCHS="${EPOCHS:-1500}"
ELITE="${ELITE:-2}"                     # survivors kept per generation (evolve -k)
LMAX="${LMAX:-3}"; WMAX="${WMAX:-16}"   # search-space bounds (evolve -L / -W)
MUT="${MUT:-1}"     # mutation rate (initial, if self-adaptive)
ADAPT="${ADAPT:-1}" # 1 = self-adaptive rate, 0 = fixed rate

# A fresh task is generated for each run (task seed = run index), so the win-rate
# is measured across many task instances, not one -- a more representative sample.
task=$(mktemp)
trap 'rm -f "$task"' EXIT

printf 'task: dim=%s samples=%s freq=%s noise=%s%% space=<=%sx%s | %s runs (fresh task each), pop=%s gens=%s epochs=%s mut=%s adapt=%s\n' \
    "$DIM" "$N" "$FREQ" "$NOISE" "$LMAX" "$WMAX" "$RUNS" "$POP" "$GENS" "$EPOCHS" "$MUT" "$ADAPT"

wins=0
sumgap=0
run=1
while [ "$run" -le "$RUNS" ]; do
    ./gentask -d "$DIM" -N "$N" -f "$FREQ" -e "$NOISE" -s "$run" > "$task"
    final=$(COMMON="-f $task -i $DIM -o 1 -e $EPOCHS" \
            ./evolve -i "$DIM" -o 1 -P "$POP" -G "$GENS" -k "$ELITE" -L "$LMAX" -W "$WMAX" -M "$MUT" -A "$ADAPT" -s "$run" 2>/dev/null \
            | grep '^final')
    # tokens after "GA" and after "RAND" are the two best test-MSE values.
    ga=$(printf '%s\n' "$final" | awk '{for(i=1;i<=NF;i++) if($i=="GA") print $(i+1)}')
    rd=$(printf '%s\n' "$final" | awk '{for(i=1;i<=NF;i++) if($i=="RAND") print $(i+1)}')
    win=$(awk -v a="$ga" -v b="$rd" 'BEGIN{print (a<b)?1:0}')
    wins=$((wins + win))
    sumgap=$(awk -v s="$sumgap" -v a="$ga" -v b="$rd" 'BEGIN{printf "%.6g", s+(b-a)}')
    if [ "$win" -eq 1 ]; then verdict="GA wins"; else verdict="RAND wins/ties"; fi
    printf 'run %d (seed %d):  GA=%-11s RAND=%-11s  %s\n' "$run" "$run" "$ga" "$rd" "$verdict"
    run=$((run + 1))
done

meangap=$(awk -v s="$sumgap" -v r="$RUNS" 'BEGIN{printf "%.4g", s/r}')
printf '\nGA beat random search in %s of %s runs; mean (RAND - GA) test-MSE gap = %s\n' \
    "$wins" "$RUNS" "$meangap"
