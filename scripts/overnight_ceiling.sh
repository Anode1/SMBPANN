#!/bin/sh
# overnight_ceiling.sh -- the two experiments worth a night, split across cores by seed range.
#
# JOB 1, the ceiling (~2 h at 8 ways). On what fraction of task draws is the planted support actually
# the argmax of its 127-support neighbourhood? Without this, "the search returns it in 22% of runs" has
# nothing to be compared against; with it, 22% is either near-optimal or a large shortfall. DRAWS=1000
# puts the per-mask error at 0.13/sqrt(1000) = 0.0041, well under the ~0.01 spacing of the leading
# candidates, which is what defeats the winner's curse that made the DRAWS=1 attempt an artifact.
#
# JOB 2, more seeds on the headline (~4 h at 6 ways). The claim that repairing the selection signal
# changes STRUCTURE rests on 6/60 vs 13/60, exact McNemar p=0.039 -- the weakest link in the result.
# 240 seeds at LAMBDA=6 either firms it up or withdraws it.
#
# Seed ranges partition exactly: verified SPLIT-EXACTLY-EQUIVALENT against a contiguous run.
set -e
cd "$(dirname "$0")/.."
echo "=== job 1: the per-seed ceiling, 60 seeds x 1000 draws, LAMBDA 6 and 1"
for L in 6 1; do
  for s in 0 8 16 24 32 40 48 56; do
    n=8; [ "$s" = 56 ] && n=4
    RAW=1 PERSEED=1 SUBW=7 DRAWS=1000 SEEDS=$n SEED0=$s LAMBDA=$L \
      ./emerge_transfer > scratch_ceil_lam${L}_s${s}.out 2>&1 &
  done
  wait
  echo "--- LAMBDA=$L ceiling:"
  cat scratch_ceil_lam${L}_s*.out | awk '/^PERSEED RAW/{n++; w+=$4} END{
    printf "    planted support is the per-seed argmax on %d of %d seeds (%.0f%%)\n", w, n, 100*w/n}'
done
echo "=== job 2: 240 seeds at LAMBDA=6, both cells"
for s in 0 40 80 120 160 200; do
  RAW=1 ROTPOS=1 FEVAL=12 LAMBDA=6 SEEDS=40 SEED0=$s \
    ./emerge_transfer > scratch_big_fixed_s${s}.out 2>&1 &
done
for s in 0 120; do
  RAW=1 ROTPOS=0 FEVAL=1 LAMBDA=6 SEEDS=120 SEED0=$s \
    ./emerge_transfer > scratch_big_nofix_s${s}.out 2>&1 &
done
wait
echo "--- exact structural match, 240 seeds, LAMBDA=6:"
for tag in nofix fixed; do
  cat scratch_big_${tag}_s*.out | awk -v T=$tag '/^RAW/ && $2==1 {n++; if($4==3 && $6==1) e++} END{
    printf "    %-6s %d/%d (%.0f%%)\n", T, e, n, 100*e/n}'
done
echo "=== done"
