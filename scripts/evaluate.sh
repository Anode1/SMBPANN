#!/bin/sh
#
# evaluate.sh -- the parallel evaluation coordinator (roadmap step 4).
#
# Reads candidate topologies, one per line (a file argument or stdin; blank
# lines and #comments ignored), evaluates each in its OWN worker process, JOBS
# (default: nproc) at a time, and prints a leaderboard sorted by fitness, lowest
# first. No shared memory and no synchronization: every worker is an independent
# `smbpann` process that exchanges a single plain-text RESULT line. A worker that
# crashes scores `inf` and does not take down the run.
#
# Env:
#   SMB     path to the worker binary    (default ./smbpann)
#   JOBS    parallel workers             (default nproc)
#   COMMON  extra args for every worker  (e.g. "-f data.txt -i 2 -o 1 -e 5000")
#
# Usage:
#   scripts/evaluate.sh candidates.txt
#   printf '2,2,1\n2,4,1\n2,8,1\n' | scripts/evaluate.sh
#
set -eu

SMB="${SMB:-./smbpann}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
COMMON="${COMMON:-}"

# --worker: evaluate one candidate topology; emit "<fitness>\t<RESULT line>".
# The fitness is prefixed as a sort key and stripped again by the coordinator.
if [ "${1:-}" = "--worker" ]; then
    topo="$2"
    line=$($SMB -t "$topo" $COMMON -q 2>/dev/null | grep '^RESULT' || true)
    if [ -z "$line" ]; then
        printf 'inf\tRESULT topology=%s fitness=inf (worker failed)\n' "$topo"
    else
        fit=$(printf '%s\n' "$line" | sed -n 's/.*fitness=\([^ ]*\).*/\1/p')
        printf '%s\t%s\n' "$fit" "$line"
    fi
    exit 0
fi

# Coordinator: fan candidates out to workers, sort the results by fitness.
export SMB COMMON
input="${1:--}"

printf 'coordinator: %s workers, one process per candidate\n' "$JOBS" >&2
grep -v -e '^[[:space:]]*$' -e '^[[:space:]]*#' "$input" \
    | xargs -P "$JOBS" -I{} "$0" --worker '{}' \
    | sort -g \
    | cut -f2-
