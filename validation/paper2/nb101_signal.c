/* nb101_signal.c -- does the toy probe's selection bias exist on REAL NAS-Bench-101 data?
 * Self-contained C99. Build: make nb101_signal
 *
 * WHAT THIS IS FOR. emerge_transfer.c showed a search overfitting WHICH held-out positions it had to
 * generalize to, detected by an exchangeability argument and removed by rotating them. That is a toy.
 * Its weakest sentence is that the mechanism "should transfer to any search selecting on a held-out
 * statistic", which is an argument, not a measurement. This probe makes it a measurement.
 *
 * WHY NAS-BENCH-101 CAN CARRY THE ARGUMENT. Every architecture is trained 3 independent times, so its
 * 3 validation accuracies are EXCHANGEABLE replicates: same architecture, different init and training
 * seed. That is exactly the structure the argument needs -- for an architecture not chosen using
 * replicate i, its expected accuracy on replicate i equals that on replicate j, so any excess is
 * created by selection and nothing else. The repo's shipped table averages the trials away, which is
 * the right table for "which architecture is best" and destroys the only structure this measurement
 * needs; nb101_trials.c keeps them.
 *
 * THE MEASUREMENT. Draw K architectures, let an arm pick the best one under its own scoring rule, then
 * ask what that architecture scores on a replicate NO arm was allowed to select on:
 *   inflation = (what the arm believed) - (held-out replicate)
 * Exchangeability sets the expectation of that to zero absent selection. Anything positive is the
 * portion of the search's apparent gain that is not real.
 *
 * ARMS, differing only in how many replicates the scoring rule averages:
 *   FIXED   score by trial perm[0]                    (what a single-run benchmark gives you)
 *   AVG2    score by mean(perm[0], perm[1])           (the cheap repair)
 * Both report on perm[2]. The trial permutation is redrawn per repetition, so which physical trial
 * plays which role cannot matter, and the two arms are paired on the same sample of K.
 *
 * Also recorded, because honest bookkeeping is not the interesting claim: the TRUE QUALITY of the
 * chosen architecture (mean of all 3 trials) and its REGRET against the best true quality present in
 * that sample. If averaging only lowers inflation without improving quality, the repair merely stops
 * you lying to yourself; if it also lowers regret, it changes what the search finds.
 *
 * env: KMIN KMAX REPS SEED  (K sweeps powers of two from KMIN to KMAX)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define MAXARCH 460000
#define NTRIAL  3

static float  vacc[MAXARCH][NTRIAL];      /* validation accuracy per trial */
static float  tacc[MAXARCH][NTRIAL];      /* test accuracy per trial      */
static float  vmean[MAXARCH];             /* mean validation over trials */
/* TEST accuracy is the only UNCONTAMINATED outcome available here, and the reason is algebraic. Since
 * vmean = (v0+v1+v2)/3, the three deviations from vmean sum to exactly zero, so an arm scoring on
 * mean(v0,v1) is scoring on 1.5*vmean - 0.5*v2: it is selecting partly ON the ground truth. Roughly
 * half of that arm's apparent advantage in vmean is therefore an artifact of the overlap rather than a
 * better choice of architecture. No arm selects on test accuracy at all, so test residuals carry no
 * such identity, and tmean is the outcome the conclusions should rest on. */
static float  tmean[MAXARCH];             /* mean TEST accuracy over trials: the clean outcome */
static int    narch = 0;

/* ---- PRNG (xorshift), same generator as the rest of the repo ---- */
/* 64-bit state, because the 32-bit xorshift this started with has a period of 2^32-1 = 4.29e9
 * draws and the largest cell here consumes 2.6e10 of them. The stream wrapped six times, so
 * repetitions about 16,385 apart were drawing IDENTICAL numbers: the effective sample size in that
 * cell was ~16k rather than 100k and its standard error was understated by roughly 2.5x. splitmix64
 * has a period of 2^64 and passes the usual test batteries, which removes the ceiling entirely.
 * Every number produced before this change is superseded, not merely re-derived. */
static uint64_t rs = 1u;
static uint32_t r32(void)
{
    uint64_t z = (rs += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return (uint32_t)((z ^ (z >> 31)) >> 32);
}
static void     rseed(uint32_t s){ rs = s ? (uint64_t)s * 0x9E3779B97F4A7C15ull : 1u; }
static uint32_t rbelow(uint32_t m){ return m ? r32()%m : 0u; }

static int envint(const char *k, int d){ const char *v=getenv(k); return v? atoi(v): d; }

/* Read nb101_trials.c's table: N adj op0..op{N-1} ntrial v0 t0 v1 t1 v2 t2.
 * Only the accuracies matter here -- the search is over the table's index set, so the graph
 * representation is not needed and is deliberately not parsed. */
static int load(const char *path)
{
    FILE *f = fopen(path, "r");
    char line[1024];
    if(!f){ fprintf(stderr, "nb101_signal: cannot open %s\n", path); return -1; }
    while(fgets(line, sizeof line, f)){
        char *p = line; int n, nt, k, skipped;
        if(*p == '#') continue;
        n = (int)strtol(p, &p, 10);
        if(n < 2) continue;
        while(*p == ' ') p++;
        while(*p && *p != ' ') p++;                    /* the adjacency bit string */
        for(k = 0; k < n; k++) (void)strtol(p, &p, 10);/* the op codes             */
        nt = (int)strtol(p, &p, 10);
        if(nt != NTRIAL) continue;                     /* keep the uniform-3-trial majority */
        if(narch >= MAXARCH) break;
        skipped = 0;
        for(k = 0; k < NTRIAL; k++){
            vacc[narch][k] = (float)strtod(p, &p);
            tacc[narch][k] = (float)strtod(p, &p);
            if(vacc[narch][k] <= 0.0f) skipped = 1;
        }
        if(skipped) continue;
        vmean[narch] = (vacc[narch][0] + vacc[narch][1] + vacc[narch][2]) / 3.0f;
        tmean[narch] = (tacc[narch][0] + tacc[narch][1] + tacc[narch][2]) / 3.0f;
        narch++;
    }
    fclose(f);
    return narch;
}

/* one repetition at a given K: both arms draw the SAME sample, so the comparison is paired */
typedef struct { double infl, qual, regret, believed, report, qtest, rtest; } Out;

static void one_rep(int K, Out *fixed, Out *avg2)
{
    int i, perm[NTRIAL] = {0,1,2}, bf = -1, ba = -1, bestidx = -1;
    double sf = -1e300, sa = -1e300, bestq = -1e300, bestt = -1e300;
    /* which physical trial plays selection / report is redrawn every repetition */
    for(i = NTRIAL-1; i > 0; i--){ int j = (int)rbelow((uint32_t)(i+1)), t = perm[i]; perm[i]=perm[j]; perm[j]=t; }

    for(i = 0; i < K; i++){
        int a = (int)rbelow((uint32_t)narch);
        double s1 = vacc[a][perm[0]];
        double s2 = 0.5*(vacc[a][perm[0]] + vacc[a][perm[1]]);
        if(s1 > sf){ sf = s1; bf = a; }
        if(s2 > sa){ sa = s2; ba = a; }
        if(vmean[a] > bestq){ bestq = vmean[a]; bestidx = a; }
        if(tmean[a] > bestt) bestt = tmean[a];      /* the clean reference for test regret */
    }
    (void)bestidx;
    fixed->believed = sf;  fixed->report = vacc[bf][perm[2]];
    fixed->infl     = sf - fixed->report;
    fixed->qual     = vmean[bf];
    fixed->regret   = bestq - vmean[bf];
    /* the HELD-OUT replicate, the only quantity no arm could see. tmean is not clean: t[k] shares its
     * training run with v[k], and the mean of three includes the ones selected on. */
    fixed->qtest    = tacc[bf][perm[2]];
    fixed->rtest    = vacc[bf][perm[2]];

    avg2->believed  = sa;  avg2->report = vacc[ba][perm[2]];
    avg2->infl      = sa - avg2->report;
    avg2->qual      = vmean[ba];
    avg2->regret    = bestq - vmean[ba];
    avg2->qtest     = tacc[ba][perm[2]];
    avg2->rtest     = vacc[ba][perm[2]];
}

static void acc(double *s, double *ss, double x){ *s += x; *ss += x*x; }
static double sd_of(double s, double ss, long n)
{ double m = s/n, v = ss/n - m*m; return v > 0 ? sqrt(v) : 0.0; }

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "validation/nasbench101_trials.txt";
    int kmin = envint("KMIN", 2), kmax = envint("KMAX", 65536);
    long reps = (long)envint("REPS", 20000);
    int K;

    if(load(path) < 0) return 1;
    rseed((uint32_t)envint("SEED", 12345));
    printf("nb101_signal -- selection bias on real NAS-Bench-101 replicates\n");
    printf("%d architectures, %d exchangeable trials each. %ld repetitions per K.\n",
           narch, NTRIAL, reps);
    printf("inflation = (score the arm selected on) - (held-out replicate it never saw);\n");
    printf("exchangeability puts its expectation at ZERO absent selection.\n");
    printf("quality = chosen architecture's mean over all 3 trials; regret vs the best in the sample.\n\n");
    printf("%7s | %-28s | %-28s | %s\n", "", "FIXED (one replicate)",
           "AVG2 (mean of two)", "paired");
    printf("%7s | %8s %10s | %8s %10s | %11s %11s\n",
           "K", "infl", "cleanTEST", "infl", "cleanTEST", "d(infl)", "d(cleanTEST)");
    for(K = kmin; K <= kmax; K *= 2){
        double fi=0, fis=0, fq=0, fr=0, ai=0, ais=0, aq=0, ar=0, di=0, dis=0;
        double ftr=0, atr=0, dt=0, dts=0;
        long r;
        for(r = 0; r < reps; r++){
            Out f, a;
            one_rep(K, &f, &a);
            acc(&fi,&fis,f.infl); fq += f.qual; ftr += f.qtest;
            acc(&ai,&ais,a.infl); aq += a.qual; atr += a.qtest;
            acc(&di,&dis,f.infl - a.infl);
            acc(&dt,&dts,a.qtest - f.qtest);   /* AVG2 minus FIXED on the clean outcome, paired */
        }
        printf("%7d | %8.4f %10.4f | %8.4f %10.4f | %+11.4f %+8.4f(%.4f)\n",
               K, fi/reps, ftr/reps, ai/reps, atr/reps,
               di/reps, dt/reps, sd_of(dt,dts,reps)/sqrt((double)reps));
        if(envint("RAW",0)){
            printf("RAWSD %d %.4f %.4f\n", K,
                   sd_of(fi,fis,reps), sd_of(ai,ais,reps));
        }
        fflush(stdout);
    }
    printf("\nP1 asks whether inflation grows with K; P2 whether AVG2 is strictly lower at every K;\n");
    printf("P3 whether AVG2 also has higher quality and lower regret (the repair changes the answer,\n");
    printf("not just the bookkeeping); P4 whether FIXED quality stalls while its believed score rises.\n");
    return 0;
}
