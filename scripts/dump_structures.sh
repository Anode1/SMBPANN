#!/bin/sh
# dump_structures.sh -- genome dumps for the structure figure.
#
# Draws from the two conditions the paper compares, at the tariff where the planted design is the
# objective's preferred fixed structure (LAMBDA=6): a search judged by one held-out sample throughout,
# and one with both repairs. Seeds are split across processes with SEED0, whose partition is verified
# to reproduce a contiguous run exactly.
#
# A note on why this exists at all: tap counts and contiguity fractions cannot show whether a support
# is ALIGNED with the planted window, and alignment is the entire question. The weight matrix can.
cd "$(dirname "$0")/.." || exit 1

for s in 0 2 4 6; do
  DUMP=1 SEEDS=2 SEED0=$s GENS=50 LAMBDA=6 ROTPOS=0 FEVAL=1 \
    ./emerge_transfer > scratch_dump_nofix_s${s}.out 2>&1 &
done
for s in 0 2 4 6; do
  DUMP=1 SEEDS=2 SEED0=$s GENS=50 LAMBDA=6 ROTPOS=1 FEVAL=12 \
    ./emerge_transfer > scratch_dump_fixed_s${s}.out 2>&1 &
done
wait

cat scratch_dump_nofix_s0.out scratch_dump_nofix_s2.out \
    scratch_dump_nofix_s4.out scratch_dump_nofix_s6.out > scratch_dump_nofix.out
cat scratch_dump_fixed_s0.out scratch_dump_fixed_s2.out \
    scratch_dump_fixed_s4.out scratch_dump_fixed_s6.out > scratch_dump_fixed.out

n=`grep -c '^DUMP final' scratch_dump_nofix.out`
m=`grep -c '^DUMP final' scratch_dump_fixed.out`
echo "dumps written: nofix $n final supports, fixed $m final supports"
echo "(4 operator arms are dumped per seed; the figure uses the first of each group, flip-only)"
