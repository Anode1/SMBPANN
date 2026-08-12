#!/bin/sh
# nb101_clean.sh -- the depth sweep redone after the review, on the outcome that survives scrutiny.
#
# Four corrections are in force here and every earlier number is superseded, not merely re-derived:
#   1. TEST accuracy is the outcome. Validation quality was contaminated: since vmean is the mean of the
#      three runs, their deviations sum to zero, so an arm scoring on two of them was scoring partly on
#      the ground truth itself. No arm selects on test.
#   2. The PRNG is 64-bit. The 32-bit xorshift had a period of 4.29e9 and the largest cells drew 2.6e10,
#      so the stream wrapped and repetitions 16,385 apart were identical.
#   3. The sham arm is matched to rotation in STRUCTURE, hashing its offset from (architecture, parity),
#      not redrawing noise per evaluation, which was implicit averaging over time.
#   4. Arms are re-seeded per RUN, so run r of every arm really does face the same permutation and the
#      same initial population. Seeding once per arm did not pair anything past run 0.
#
# RANDOM at matched budget is included because without it the arm comparison cannot be shown to be a
# comparison among searches worth running.
cd "$(dirname "$0")/.." || exit 1
T=validation/nasbench101_trials.txt
for G in 800 3200; do
  JSIG=0.33 POP=24 GENS=$G RUNS=10000 ./nb101_search $T > scratch_clean_g${G}.out 2>&1 &
done
wait
echo "TEST REGRET, lower is better (clean outcome; no arm selects on it)"
for G in 50 200 800 3200; do
  awk -v G=$G '/FIXED   \(/{split($(NF-2),f,"+")} /SHAM    \(/{split($(NF-2),s,"+")}
       /ROTATE  \(/{split($(NF-2),r,"+")} /AVG2    \(/{split($(NF-2),a,"+")}
       /RANDOM  \(/{split($(NF-2),n,"+")}
       END{printf "  gens %-5s fixed %-8s sham %-8s rotate %-8s avg2 %-8s random %-8s\n",
                  G, f[1], s[1], r[1], a[1], n[1]}' scratch_clean_g${G}.out
done
