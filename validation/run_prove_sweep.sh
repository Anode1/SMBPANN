#!/usr/bin/env bash
# Reproduce the emerge_prove scaling study (directed GA vs random under an energy budget).
# Runs three grids in 8-way parallel and prints each aggregated table:
#   1. fixed budget   (200 generations at every N)
#   2. scaled budget  (generations grow linearly with N: gens = round(200*N/12))
#   3. denoise control (fixed budget; random arm gets r=1 vs r=8 evaluations per mask,
#                       to isolate directed search from repeated-evaluation noise-averaging)
# Everything is deterministic in the seed, so the seed range is chunked across cores and
# re-aggregated with the in-C AGG mode; results are identical to a single serial run.
#
# Usage:  bash validation/run_prove_sweep.sh [SEEDS] [OUTDIR]
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS="${1:-50}"
OUT="${2:-/tmp/prove_sweep}"
NLIST="12 16 20 24 28"
CHUNK=10
PAR=8
cc -std=c99 -pedantic -Wall -Wextra -O2 -o emerge_prove validation/emerge_prove.c -lm
mkdir -p "$OUT"

# emit jobs for one grid: args = tag, extra-env (GENS/REVALS overrides applied per-N via the caller)
run_grid () {   # $1=tag  $2=gens_expr(function of N, bash arith using $N)  $3=revals
  local tag="$1" gexpr="$2" rev="$3"
  local d="$OUT/$tag"
  rm -rf "$d"; mkdir -p "$d"
  local jobs="$d/jobs.sh"; : > "$jobs"
  for N in $NLIST; do
    local G; G=$(python3 -c "N=$N; print($gexpr)")
    local s=1
    while [ "$s" -le "$SEEDS" ]; do
      echo "SEEDS=$CHUNK SEED0=$s GENS=$G NLIST=$N REVALS=$rev RAW=1 $(pwd)/emerge_prove > $d/N${N}_s${s}.txt" >> "$jobs"
      s=$(( s + CHUNK ))
    done
  done
  xargs -P "$PAR" -I{} bash -c '{}' < "$jobs"
  echo "======== $tag ========"
  cat "$d"/N*.txt | AGG=1 "$(pwd)/emerge_prove"
  echo
}

run_grid fixed        "200"                 1
run_grid scaled       "round(200*N/12)"     1
run_grid denoise_r8   "200"                 8   # confound control: random gets 8 evals/mask (vs fixed's r=1)
