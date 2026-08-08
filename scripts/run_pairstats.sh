#!/bin/sh
# run_pairstats.sh -- regenerate every paired claim's PER-SEED data and run the named tests.
#
# WHY THIS EXISTS. The paper reported paired differences as mean +/- SEM with no named test, which
# both submission_plan.md and revision_notes_gpem.md flag as the most likely referee request. Every
# probe below already had the paired seeds; they just aggregated them away. Each now takes RAW=1 and
# emits one line per seed, which validation/pairstat.c turns into Wilcoxon signed-rank (continuous)
# or McNemar exact (binary solve counts), with Holm across the declared family.
#
# The aggregate tables are still printed by every probe, so an archived run stays comparable: RAW is
# an added mode, never a replacement.
#
# Run from the repo root:  nohup sh scripts/run_pairstats.sh > /dev/null 2>&1 &
# Results land in scratch_pairstat_*.out; raw per-seed data in scratch_raw_*.out.

set -e
cd "$(dirname "$0")/.."

# Wait for any in-flight shards so we do not oversubscribe the cores.
# NOTE: -x matches the process NAME exactly. Do NOT use `pgrep -f`: it matches whole command lines,
# so any shell whose command line merely mentions the probe (an echo, a comment, this script's own
# launcher) counts as a running job and the loop waits on itself forever. That happened.
# Use pgrep's EXIT STATUS (0 = something matched), not its printed count: `pgrep -c` prints 0 and
# also exits 1 when nothing matches, so a `|| echo 0` fallback emits two lines and the integer test
# becomes invalid.
while pgrep -x emerge_prove >/dev/null 2>&1; do sleep 30; done

make pairstat emerge_translate emerge_scale emerge_baseline emerge_develop \
     emerge_discover emerge_arch emerge_twoop emerge_staged

# ---- continuous outcomes: Wilcoxon signed-rank -------------------------------------------------
# Seed counts match the paper's stated values for each section.

SEEDS=40  RAW=1 ./emerge_translate > scratch_raw_translate.out 2>&1 &
SEEDS=250 RAW=1 ./emerge_scale     > scratch_raw_scale.out     2>&1 &
SEEDS=60  RAW=1 ./emerge_baseline  > scratch_raw_baseline.out  2>&1 &
SEEDS=40  RAW=1 ./emerge_develop   > scratch_raw_develop.out   2>&1 &
SEEDS=40  RAW=1 ./emerge_discover  > scratch_raw_discover.out  2>&1 &

# ---- binary outcomes: McNemar exact ------------------------------------------------------------
SEEDS=24  RAW=1 ./emerge_arch      > scratch_raw_arch.out      2>&1 &
SEEDS=24  RAW=1 ./emerge_twoop     > scratch_raw_twoop.out     2>&1 &
SEEDS=24  RAW=1 ./emerge_staged    > scratch_raw_staged.out    2>&1 &
wait

# ---- the tests ---------------------------------------------------------------------------------
# translate/scale/baseline: group = input size or train size, arms are columns 4 and 5.
GROUP=2 METRICS=shared_vs_indep:4:5 ./pairstat < scratch_raw_translate.out > scratch_pairstat_translate.out
GROUP=2 METRICS=shared_vs_indep:4:5 ./pairstat < scratch_raw_scale.out     > scratch_pairstat_scale.out
# baseline columns: 4 shared, 5 indep winner-only, 6 indep oracle-full. The claim is shared vs ORACLE.
GROUP=2 METRICS=shared_vs_oracle:4:6,shared_vs_winner:4:5 ./pairstat < scratch_raw_baseline.out \
        > scratch_pairstat_baseline.out
# develop phases: 4 scratch, 5 block+translate, 6 refine, 7 refine+jitter.
GROUP=2 METRICS=tiled_vs_scratch:5:4,jitter_cost:7:5,refine_vs_tiled:6:5 ./pairstat \
        < scratch_raw_develop.out > scratch_pairstat_develop.out
# discover: 4 discovered(C=2), 5 supervised curriculum, 6 unshared scratch.
GROUP=2 METRICS=curric_vs_disc:5:4,disc_vs_scratch:4:6 ./pairstat \
        < scratch_raw_discover.out > scratch_pairstat_discover.out

BINARY=1 GROUP=2 METRICS=free_vs_reuse:5:4 ./pairstat < scratch_raw_arch.out  > scratch_pairstat_arch.out
BINARY=1 GROUP=2 METRICS=free_vs_reuse:7:6 ./pairstat < scratch_raw_twoop.out > scratch_pairstat_twoop.out
BINARY=1 GROUP=2 METRICS=staged_vs_reuse:6:4,staged_vs_free:6:5 ./pairstat \
        < scratch_raw_staged.out > scratch_pairstat_staged.out

echo "done: $(date)" > scratch_pairstat_DONE
