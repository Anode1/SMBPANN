#!/bin/sh
#
# errortest.sh -- a reproducible time-to-target benchmark (error control).
#
# This is the self_modifying_predict(Train, Test, Error) measure: instead of a
# fixed generation budget and comparing final fitness, each run searches UNTIL the
# best validation error reaches a target (evolve -E), with -G a safety cap. What
# matters is then the WORK to get there. For each of several seeds this races the
# GA against its matched random-search control and reports, for each method, how
# often it reached the target and how fast (generations and evaluations), plus the
# head-to-head: in how many runs the GA reached the target sooner than random.
#
# A fresh synthetic task is generated per run (task seed = run index), so the
# result is a sample over task instances, not one. Fully self-contained.
#
# Env (all optional): DIM N FREQ NOISE TARGET RUNS POP GENS EPOCHS LMAX WMAX
#                     MUT ADAPT ELITE
#
# Usage: scripts/errortest.sh
#        TARGET=0.12 RUNS=30 GENS=60 scripts/errortest.sh
#
set -eu

DIM="${DIM:-8}"; N="${N:-400}"; FREQ="${FREQ:-3}"; NOISE="${NOISE:-2}"
TARGET="${TARGET:-0.12}"                 # the Error to reach (validation MSE)
RUNS="${RUNS:-20}"; POP="${POP:-12}"; GENS="${GENS:-50}"; EPOCHS="${EPOCHS:-1500}"
ELITE="${ELITE:-2}"                      # survivors kept per generation (evolve -k)
LMAX="${LMAX:-3}"; WMAX="${WMAX:-16}"    # search-space bounds (evolve -L / -W)
MUT="${MUT:-2}"     # mutation rate (initial, if self-adaptive)
ADAPT="${ADAPT:-1}" # 1 = self-adaptive rate, 0 = fixed rate

task=$(mktemp)
res=$(mktemp)
trap 'rm -f "$task" "$res"' EXIT

# FIXTASK (optional): a seed to hold the task constant across runs, so only the
# search seed varies. This isolates search efficiency from task-difficulty luck
# (the cleaner time-to-target measure). Unset = a fresh task per run (a sample
# over task instances, as in benchmark.sh).
FIXTASK="${FIXTASK:-}"
if [ -n "$FIXTASK" ]; then
    ./gentask -d "$DIM" -N "$N" -f "$FREQ" -e "$NOISE" -s "$FIXTASK" > "$task"
    taskdesc="fixed task (seed $FIXTASK), search seed varies"
else
    taskdesc="fresh task per run"
fi

printf 'error control: target=%s | task dim=%s samples=%s freq=%s noise=%s%% space=<=%sx%s | %s runs (%s), pop=%s cap=%s gens epochs=%s mut=%s adapt=%s\n\n' \
    "$TARGET" "$DIM" "$N" "$FREQ" "$NOISE" "$LMAX" "$WMAX" "$RUNS" "$taskdesc" "$POP" "$GENS" "$EPOCHS" "$MUT" "$ADAPT"

run=1
while [ "$run" -le "$RUNS" ]; do
    [ -n "$FIXTASK" ] || ./gentask -d "$DIM" -N "$N" -f "$FREQ" -e "$NOISE" -s "$run" > "$task"
    # the TARGET line is machine-readable: ga_gens/ga_evals rand_gens/rand_evals
    # (a gens field of 0 means that method never reached the target by the cap).
    line=$(COMMON="-f $task -i $DIM -o 1 -e $EPOCHS" \
           ./evolve -i "$DIM" -o 1 -P "$POP" -G "$GENS" -k "$ELITE" -L "$LMAX" -W "$WMAX" -M "$MUT" -A "$ADAPT" -E "$TARGET" -s "$run" 2>/dev/null \
           | grep '^TARGET')
    gg=$(printf '%s\n' "$line" | sed -n 's/.*ga_gens=\([0-9]*\).*/\1/p')
    ge=$(printf '%s\n' "$line" | sed -n 's/.*ga_evals=\([0-9]*\).*/\1/p')
    rg=$(printf '%s\n' "$line" | sed -n 's/.*rand_gens=\([0-9]*\).*/\1/p')
    re=$(printf '%s\n' "$line" | sed -n 's/.*rand_evals=\([0-9]*\).*/\1/p')
    printf '%s %s %s %s\n' "$gg" "$ge" "$rg" "$re" >> "$res"
    ga_s=$([ "$gg" -gt 0 ] 2>/dev/null && echo "gen $gg" || echo "----")
    rd_s=$([ "$rg" -gt 0 ] 2>/dev/null && echo "gen $rg" || echo "----")
    printf 'run %2d (seed %2d):  GA reached %-8s  RAND reached %-8s\n' "$run" "$run" "$ga_s" "$rd_s"
    # LOG (optional): a durable, per-run progress line for watching a long run
    # (each append reopens the file, so it lands on disk immediately).
    [ -z "${LOG:-}" ] || printf 'run %d seed %d ga_gens %s rand_gens %s\n' "$run" "$run" "$gg" "$rg" >> "$LOG"
    run=$((run + 1))
done

# Summary: reach counts, mean/median generations-to-target among the runs that
# reached, and the head-to-head (fewer generations wins; both must reach to race).
awk -v runs="$RUNS" '
function median(a, m,   n, i, t, j) {
    n = 0; for (i = 1; i <= m; i++) n = i;
    for (i = 1; i < n; i++) for (j = i+1; j <= n; j++) if (a[j] < a[i]) { t=a[i]; a[i]=a[j]; a[j]=t }
    if (n == 0) return 0;
    return (n % 2) ? a[(n+1)/2] : (a[n/2] + a[n/2+1]) / 2.0;
}
{
    gg=$1; ge=$2; rg=$3; re=$4;
    if (gg > 0) { gah++; gag[gah]=gg; gae_sum+=ge; gag_sum+=gg }
    if (rg > 0) { rah++; rag[rah]=rg; rae_sum+=re; rag_sum+=rg }
    if (gg > 0 && rg > 0) { if (gg < rg) gawin++; else if (rg < gg) rawin++; else tie++ }
    else if (gg > 0) gawin++;      # only GA reached -> GA wins the race
    else if (rg > 0) rawin++;      # only RAND reached
    else neither++;
}
END {
    printf "\n--- summary over %d runs (target reached / how fast) ---\n", runs;
    printf "GA   reached target in %d/%d runs", gah, runs;
    if (gah) printf "   mean %.1f gens (%.0f evals), median %.1f gens", gag_sum/gah, gae_sum/gah, median(gag, gah);
    printf "\n";
    printf "RAND reached target in %d/%d runs", rah, runs;
    if (rah) printf "   mean %.1f gens (%.0f evals), median %.1f gens", rag_sum/rah, rae_sum/rah, median(rag, rah);
    printf "\n";
    printf "head-to-head (fewer generations to target): GA %d, RAND %d, tie %d, neither reached %d\n",
        gawin+0, rawin+0, tie+0, neither+0;
}
' "$res"
