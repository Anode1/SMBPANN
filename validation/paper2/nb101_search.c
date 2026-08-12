/* nb101_search.c -- ROTATION vs AVERAGING on real NAS-Bench-101, the direct analogue of ROTPOS.
 * Self-contained C99. Build: make nb101_search
 *
 * WHAT THIS ADDS OVER nb101_signal.c. That probe measures selection bias for random search, which has
 * no generations, so it can only test AVERAGING replicates. The toy result in emerge_transfer.c turns
 * on a sharper distinction: ROTATING which held-out sample supplies the selection signal removes the
 * fixed target the search would otherwise memorise, WITHOUT reducing the variance of any single
 * evaluation. Rotation and averaging are therefore different interventions, and on the toy task they
 * behaved differently -- rotation closed the functional gap, averaging moved the structural one. Only
 * a generational search can test rotation, which is what this is.
 *
 * THE THREE ARMS differ only in the scoring rule the GA selects on. All three report on a replicate
 * none of them is allowed to select on, and the trial permutation is redrawn per run, so which
 * physical trial plays which role cannot matter.
 *   FIXED    score = v[perm[0]]                       one replicate, the whole run   (the bias)
 *   ROTATE   score = v[perm[gen & 1]]                 redrawn every generation       (the repair)
 *   AVG2     score = mean(v[perm[0]], v[perm[1]])     variance reduction             (the comparison)
 * ROTATE and AVG2 see exactly the same two replicates; they differ only in whether the signal is
 * averaged or alternated, so any difference between them is that distinction and nothing else. Note
 * ROTATE's per-generation signal is exactly as noisy as FIXED's -- it is not a variance reduction --
 * which is what makes it a clean test of the memorisation mechanism.
 *
 * Graph representation, canonicalisation and mutation are taken VERBATIM from validation/nb101.c so
 * the search space and the fitness oracle are identical to the published crossover experiment; only
 * the fitness table (which keeps all three trials) and the selection rule are new.
 *
 * env: POP GENS RUNS SEED ARM(0=fixed,1=rotate,2=avg2, -1=all)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define MAXN   7
#define PEDGE  38
#define MAXP   128            /* population cap; POP is clamped to it, never silently exceeded */
#define NTRIAL 3
#define TBITS  21
#define TSIZE  (1u << TBITS)

static int envint(const char *k, int d){ const char *v=getenv(k); return v? atoi(v): d; }

/* ---- PRNG (xorshift) ---- */
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

/* ---- a cell graph ---- */
typedef struct {
    int           n;              /* node count, 2..7 (input=0, output=n-1) */
    unsigned char adj[MAXN][MAXN];/* directed adjacency (edge a->b) */
    signed char   op[MAXN];       /* 0=input 1=c1x1 2=c3x3 3=maxpool 4=output */
} Graph;

/* ---- interior-node permutation tables (for canonicalization/alignment) ---- */
static int  perms[6][120][5];     /* perms[m] = the m! permutations of 0..m-1 */
static int  nperm[6];

static void gen_perms(void)
{
    int m;
    for (m = 0; m <= 5; m++) {
        int a[5], i, cnt = 0, c[5];
        for (i = 0; i < m; i++) a[i] = i;
        /* Heap's algorithm, iterative */
        for (i = 0; i < m; i++) c[i] = 0;
        { int k; for (k = 0; k < m; k++) perms[m][cnt][k] = a[k]; }
        cnt = 1;
        i = 0;
        while (i < m) {
            if (c[i] < i) {
                int sw = (i % 2 == 0) ? 0 : c[i];
                int t = a[sw]; a[sw] = a[i]; a[i] = t;
                { int k; for (k = 0; k < m; k++) perms[m][cnt][k] = a[k]; }
                cnt++;
                c[i]++;
                i = 0;
            } else { c[i] = 0; i++; }
        }
        nperm[m] = (m == 0) ? 1 : cnt;
    }
}

/* Pack a graph (with a GIVEN node labeling) into a comparable/uniqueness key:
 * low 3 bits = n, then the off-diagonal adjacency cells in fixed order, then the
 * interior op codes (2 bits each). Same n => same layout, so it is a total order. */
static uint64_t encode(const Graph *g)
{
    uint64_t key = (uint64_t)g->n;
    int shift = 3, a, b, k;
    for (a = 0; a < g->n; a++)
        for (b = 0; b < g->n; b++)
            if (a != b) { key |= (uint64_t)(g->adj[a][b] & 1u) << shift; shift++; }
    for (k = 1; k < g->n - 1; k++) {
        key |= (uint64_t)((g->op[k] - 1) & 3u) << shift; shift += 2;
    }
    return key;
}

/* Canonical key = min encode() over all relabelings that fix input(0) and
 * output(n-1) and permute the interior nodes. Isomorphic graphs => same key. */
static uint64_t canonical(const Graph *g)
{
    int m = g->n - 2;                 /* interior nodes at indices 1..n-2 */
    uint64_t best = ~(uint64_t)0;
    int pi;
    if (m < 0) m = 0;
    for (pi = 0; pi < nperm[m]; pi++) {
        Graph h; int a, b;
        int map[MAXN];                /* new index -> old index */
        map[0] = 0; map[g->n - 1] = g->n - 1;
        for (a = 0; a < m; a++) map[1 + a] = 1 + perms[m][pi][a];
        h.n = g->n;
        for (a = 0; a < g->n; a++) h.op[a] = g->op[map[a]];
        for (a = 0; a < g->n; a++)
            for (b = 0; b < g->n; b++)
                h.adj[a][b] = g->adj[map[a]][map[b]];
        { uint64_t e = encode(&h); if (e < best) best = e; }
    }
    return best;
}

/* edge count */
static int nedges(const Graph *g)
{
    int a, b, c = 0;
    for (a = 0; a < g->n; a++) for (b = 0; b < g->n; b++) c += g->adj[a][b] ? 1 : 0;
    return c;
}

/* Prune nodes that are not on any input->output path, compacting the survivors in
 * original index order. Returns 1 if the pruned graph is valid (input reaches
 * output, >=2 nodes), else 0. On success *out holds the reduced graph. */
static int prune(const Graph *g, Graph *out)
{
    int fromIn[MAXN] = {0}, toOut[MAXN] = {0}, keep[MAXN];
    int i, j, changed, cnt = 0, idx[MAXN];

    fromIn[0] = 1;
    do { changed = 0;
        for (i = 0; i < g->n; i++) if (fromIn[i])
            for (j = 0; j < g->n; j++) if (g->adj[i][j] && !fromIn[j]) { fromIn[j] = 1; changed = 1; }
    } while (changed);

    toOut[g->n - 1] = 1;
    do { changed = 0;
        for (i = 0; i < g->n; i++) if (toOut[i])
            for (j = 0; j < g->n; j++) if (g->adj[j][i] && !toOut[j]) { toOut[j] = 1; changed = 1; }
    } while (changed);

    for (i = 0; i < g->n; i++) {
        keep[i] = (fromIn[i] && toOut[i]);
        if (keep[i]) idx[cnt++] = i;
    }
    if (cnt < 2 || !keep[0] || !keep[g->n - 1]) return 0;   /* input must reach output */

    out->n = cnt;
    { int a, b, ra, rb;
      for (a = 0; a < cnt; a++) out->op[a] = g->op[idx[a]];
      for (ra = 0; ra < cnt; ra++) for (rb = 0; rb < cnt; rb++)
          out->adj[ra][rb] = 0;
      for (ra = 0; ra < cnt; ra++) for (rb = 0; rb < cnt; rb++)
          out->adj[ra][rb] = g->adj[idx[ra]][idx[rb]];
      (void)a; (void)b;
    }
    return 1;
}

static void random_raw(Graph *g)
{
    int i, j;
    g->n = MAXN; g->op[0] = 0; g->op[MAXN - 1] = 4;
    for (i = 1; i < MAXN - 1; i++) g->op[i] = (signed char)(1 + rbelow(3));
    for (i = 0; i < MAXN; i++) for (j = 0; j < MAXN; j++) g->adj[i][j] = 0;
    for (i = 0; i < MAXN; i++)
        for (j = i + 1; j < MAXN; j++)
            g->adj[i][j] = (unsigned char)(rbelow(100) < PEDGE);
}

/* one light mutation, shared by every GA arm (so arms differ ONLY in crossover):
 * flip a random edge, or re-roll a random interior op. */
static void mutate(Graph *g)
{
    if (rbelow(100) < 70u) {
        if (r32() & 1u) { int i = (int)rbelow(MAXN - 1); int j = i + 1 + (int)rbelow(MAXN - 1 - i);
                          g->adj[i][j] ^= 1u; }
        else            { int k = 1 + (int)rbelow(MAXN - 2); g->op[k] = (signed char)(1 + rbelow(3)); }
    }
}


/* ---- fitness table: canonical key -> all THREE validation trials (and their mean) ----
 * nb101.c stores one averaged accuracy, which is the right table for ranking architectures and the
 * wrong one for measuring an estimator: averaging the replicates destroys the exchangeable structure
 * the whole measurement rests on. */
/* TEST accuracy is kept because it is the only UNCONTAMINATED outcome. vmean = (v0+v1+v2)/3 makes the
 * three deviations sum to exactly zero, so an arm scoring on mean(v0,v1) is scoring on
 * 1.5*vmean - 0.5*v2, i.e. partly on the ground truth itself. No arm selects on test at all. */
typedef struct { uint64_t key; float v[NTRIAL], t[NTRIAL]; float vmean, tmean; } Slot;
static Slot *tab;
static long  ntab = 0;
static float g_bestmean = -1e30f;          /* best true quality in the whole catalogue */

static float g_besttmean = -1e30f;
static void tab_put(uint64_t key, const float *v, const float *t)
{
    uint32_t i = (uint32_t)((key * 11400714819323198485ull) >> (64 - TBITS));
    int k; float m = 0;
    while (tab[i].key) {
        if (tab[i].key == key) return;
        i = (i + 1u) & (TSIZE - 1u);
    }
    tab[i].key = key;
    for (k = 0; k < NTRIAL; k++){ tab[i].v[k] = v[k]; tab[i].t[k] = t[k]; m += v[k]; }
    tab[i].vmean = m / NTRIAL;
    tab[i].tmean = (t[0]+t[1]+t[2])/3.0f;
    if (tab[i].vmean > g_bestmean)  g_bestmean  = tab[i].vmean;
    if (tab[i].tmean > g_besttmean) g_besttmean = tab[i].tmean;
    ntab++;
}
static const Slot *tab_get(uint64_t key)
{
    uint32_t i = (uint32_t)((key * 11400714819323198485ull) >> (64 - TBITS));
    while (tab[i].key) {
        if (tab[i].key == key) return &tab[i];
        i = (i + 1u) & (TSIZE - 1u);
    }
    return NULL;
}
/* fitness oracle, identical in structure to nb101.c: validate -> prune -> <=9 edges -> canonical */
static const Slot *lookup_arch(const Graph *g)
{
    Graph p;
    if (g->n < 2 || g->n > MAXN) return NULL;
    if (!prune(g, &p)) return NULL;
    if (nedges(&p) > 9) return NULL;
    return tab_get(canonical(&p));
}

/* parse nb101_trials.c output: N adjbits op0..op{N-1} ntrial v0 t0 v1 t1 v2 t2 */
static long load(const char *path)
{
    FILE *f = fopen(path, "r");
    char line[1024];
    long n = 0;
    if (!f) { fprintf(stderr, "nb101_search: cannot open %s\n", path); return -1; }
    while (fgets(line, sizeof line, f)) {
        Graph g; float v[NTRIAL], t[NTRIAL]; int k, nt, bad = 0;
        const char *p = line; char *e;
        if (*p == '#') continue;
        g.n = (int)strtol(p, &e, 10); if (e == p || g.n < 2 || g.n > MAXN) continue;
        p = e; while (*p == ' ') p++;
        { int i, j;
          for (i = 0; i < MAXN; i++) for (j = 0; j < MAXN; j++) g.adj[i][j] = 0;
          for (i = 0; i < g.n; i++) for (j = 0; j < g.n; j++) {
              if (*p != '0' && *p != '1') { bad = 1; break; }
              g.adj[i][j] = (unsigned char)(*p++ - '0');
          }
          if (bad) continue; }
        for (k = 0; k < g.n; k++) { g.op[k] = (signed char)strtol(p, &e, 10); if (e == p) { bad = 1; break; } p = e; }
        if (bad) continue;
        nt = (int)strtol(p, &e, 10); p = e;
        if (nt != NTRIAL) continue;
        for (k = 0; k < NTRIAL; k++) {
            v[k] = (float)strtod(p, &e); p = e;      /* val  */
            t[k] = (float)strtod(p, &e); p = e;      /* test */
            if (v[k] <= 0.0f) bad = 1;
        }
        if (bad) continue;
        { Graph pr;
          if (!prune(&g, &pr)) continue;
          tab_put(canonical(&pr), v, t); }
        n++;
    }
    fclose(f);
    return n;
}

/* ---- the search ---- */
typedef struct { Graph g; const Slot *s; } Indiv;

/* score an individual under the arm's rule. gen is needed only by ROTATE.
 *
 * Arms 3 and 4 exist to kill an alternative explanation. Alternating the replicate reshuffles the
 * fitness ranking every generation, and noisy selection is independently known to slow a GA's
 * convergence and so preserve diversity. If that is why alternating wins, then ANY selection noise of
 * the same magnitude should win equally, even noise carrying no information about the architecture
 * being scored. JITTER supplies exactly that: it perturbs the fixed replicate by the replicate gap of a
 * DIFFERENT, randomly chosen architecture, which matches the perturbation's marginal distribution
 * exactly, heavy tail included, while being statistically independent of the candidate. If JITTER
 * reproduces ROTATE, the mechanism is noise-aided exploration and the information in the second
 * replicate is irrelevant. If JITTER behaves like FIXED, the information is what matters.
 * ROTEVAL alternates per EVALUATION rather than per generation, which decorrelates the noise at the
 * individual level and so separates "fresh noise" from "a fresh sample for the whole generation". */
static const Slot *g_pool;              /* unused by the scaled jitter; kept for reference */
static long        g_pooln;
static double      g_jsig = 0.33;       /* jitter SD in accuracy points; median within-arch spread */
/* Box-Muller, so the jitter is Gaussian with a SET scale rather than borrowed from a heavy tail.
 * The first version of this control perturbed by the replicate gap of a random architecture, which
 * matched the marginal distribution but injected roughly twenty times a typical architecture's own
 * spread (6.74 against 0.33), so it tested over-noising rather than matched noising. Sweeping the
 * scale is what actually separates noise-aided exploration from information. */
/* rnorm() removed: the sham it served drew fresh noise per evaluation, which is implicit
 * temporal averaging rather than a structural match to rotation. The replacement hashes the offset
 * from (architecture, generation parity), so no Gaussian generator is needed. */
static double score_of(const Slot *s, int arm, const int *perm, int gen)
{
    if (!s) return -1e30;
    if (arm == 0) return s->v[perm[0]];
    if (arm == 1) return s->v[perm[gen & 1]];
    if (arm == 2) return 0.5 * (s->v[perm[0]] + s->v[perm[1]]);
    if (arm == 3) {
        /* SHAM: matched to ROTATE in STRUCTURE, not merely in magnitude. Redrawing noise every
         * evaluation, as the first version did, is repeated independent re-evaluation and therefore a
         * different intervention: it averages over time. Under ROTATE a candidate's score is a
         * deterministic function of (architecture, generation parity), so the sham must be too. The
         * offset is hashed from the architecture's canonical key and the parity, which is reproducible
         * within a generation, changes between parities, and carries no information about quality. */
        uint64_t h = s->key ^ (0xD1B54A32D192ED03ull * (uint64_t)(gen & 1));
        h += 0x9E3779B97F4A7C15ull;
        h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ull;
        h = (h ^ (h >> 27)) * 0x94D049BB133111EBull;
        h =  h ^ (h >> 31);
        { double u1 = ((h >> 11) + 1.0) / 9007199254740993.0;
          double u2 = (((h * 0x2545F4914F6CDD1Dull) >> 11) + 1.0) / 9007199254740993.0;
          return s->v[perm[0]] + g_jsig * sqrt(-2.0*log(u1)) * cos(6.283185307179586*u2); }
    }
    if (arm == 4) return s->v[perm[(int)(r32() & 1u)]];   /* ROTEVAL */
    return s->v[perm[0]];                                 /* RANDOM uses the FIXED rule */
}

typedef struct { double believed, report, qual, regret, distinct, qtest, rtest; } Res;

/* RANDOM: the control the arm comparison needs. Draws pop*gens architectures uniformly from the same
 * rejection sampler the GA seeds from, scores them by the FIXED rule, and keeps the best. It is matched
 * to every other arm on number of evaluations AND on where the candidates come from, so it answers the
 * question that otherwise hangs over all of this: are these arms distinguishing better searches, or
 * ranking searches that a random draw would beat? */
static void run_random(int pop, int gens, Res *out)
{
    long budget = (long)pop * gens, i;
    const Slot *best = NULL; double bs = -1e30;
    int perm[NTRIAL] = {0,1,2}, k;
    for (k = NTRIAL-1; k > 0; k--) { int j = (int)rbelow((uint32_t)(k+1)), t = perm[k]; perm[k]=perm[j]; perm[j]=t; }
    for (i = 0; i < budget; i++) {
        Graph g; const Slot *sl;
        do { random_raw(&g); sl = lookup_arch(&g); } while (!sl);
        if (sl->v[perm[0]] > bs) { bs = sl->v[perm[0]]; best = sl; }
    }
    out->believed = bs;
    out->report   = best->v[perm[2]];
    out->qual     = best->vmean;
    out->regret   = g_bestmean  - best->vmean;
    out->qtest    = best->t[perm[2]];
    out->rtest    = best->v[perm[2]];
    out->distinct = 1;
}

static void run_one(int arm, int pop, int gens, Res *out)
{
    static Indiv P[MAXP], Q[MAXP];
    double fit[MAXP];
    int perm[NTRIAL] = {0,1,2};
    int i, gen, best = 0;
    for (i = NTRIAL-1; i > 0; i--) { int j = (int)rbelow((uint32_t)(i+1)), t = perm[i]; perm[i]=perm[j]; perm[j]=t; }

    for (i = 0; i < pop; i++) {
        do { random_raw(&P[i].g); P[i].s = lookup_arch(&P[i].g); } while (!P[i].s);
    }
    for (gen = 0; gen < gens; gen++) {
        int e, elite = pop / 4;
        int idx[MAXP];
        for (i = 0; i < pop; i++) { fit[i] = score_of(P[i].s, arm, perm, gen); idx[i] = i; }
        for (i = 0; i < pop; i++) for (e = i+1; e < pop; e++)
            if (fit[idx[e]] > fit[idx[i]]) { int t = idx[i]; idx[i] = idx[e]; idx[e] = t; }
        for (e = 0; e < elite; e++) Q[e] = P[idx[e]];
        for (e = elite; e < pop; e++) {
            int tries = 0;
            do { Q[e] = P[idx[(int)rbelow((uint32_t)elite)]];
                 mutate(&Q[e].g);
                 Q[e].s = lookup_arch(&Q[e].g);
            } while (!Q[e].s && ++tries < 40);
            if (!Q[e].s) Q[e] = P[idx[(int)rbelow((uint32_t)elite)]];
        }
        memcpy(P, Q, sizeof(Indiv) * (size_t)pop);
    }
    /* the run's answer: the individual the ARM's own rule ranks first at the end */
    { double bf = -1e30;
      for (i = 0; i < pop; i++) {
          double f = score_of(P[i].s, arm, perm, gens - 1);
          if (f > bf) { bf = f; best = i; }
      }
      out->believed = bf; }
    out->report = P[best].s->v[perm[2]];         /* a replicate no arm selected on */
    out->qual   = P[best].s->vmean;
    out->regret = g_bestmean - P[best].s->vmean;
    /* THE clean outcome. tmean is NOT clean: t[k] comes from the same training run as v[k], so its
     * noise is correlated with whichever replicate the arm selected on, and averaging all three keeps
     * that correlation. Only the perm[2]-indexed values were never available to any arm. */
    out->qtest  = P[best].s->t[perm[2]];
    out->rtest  = P[best].s->v[perm[2]];
    /* distinct architectures left in the final population: the diversity-collapse hypothesis predicts
     * this is near 1 for FIXED and larger for the arms that reshuffle the ranking */
    { int i2, j2, d = 0;
      for (i2 = 0; i2 < pop; i2++) {
          int dup = 0;
          for (j2 = 0; j2 < i2; j2++) if (P[j2].s == P[i2].s) { dup = 1; break; }
          if (!dup) d++;
      }
      out->distinct = d; }
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "validation/nasbench101_trials.txt";
    int pop = envint("POP", 24), gens = envint("GENS", 50), arm0, arm1, arm;
    /* An unchecked POP wrote past P[]/Q[]/fit[] and killed the POP=96 cell of the first sweep
     * (2026-08-12). Clamping and saying so is the fix; a bound that can be exceeded by an env var is
     * a defect even when it crashes rather than corrupting. */
    if (pop > MAXP) { fprintf(stderr, "nb101_search: POP %d exceeds MAXP %d, clamped\n", pop, MAXP); pop = MAXP; }
    if (pop < 4) pop = 4;
    long runs = (long)envint("RUNS", 2000), r;
    static const char *nm[6] = { "FIXED   (one replicate)", "ROTATE  (redrawn per gen)",
                                 "AVG2    (mean of two)",  "SHAM    (parity noise, no info)",
                                 "ROTEVAL (redrawn per eval)",
                                 "RANDOM  (matched budget)" };
    gen_perms();
    tab = calloc(TSIZE, sizeof *tab);
    if (!tab) { fprintf(stderr, "nb101_search: out of memory\n"); return 1; }
    if (load(path) < 0) return 1;
    /* a compact array of the catalogued architectures, for JITTER to draw its perturbation from */
    { uint32_t i2; g_pool = calloc((size_t)ntab, sizeof *g_pool);
      if (!g_pool) { fprintf(stderr, "nb101_search: out of memory\n"); return 1; }
      for (i2 = 0, g_pooln = 0; i2 < TSIZE; i2++) if (tab[i2].key) ((Slot*)g_pool)[g_pooln++] = tab[i2]; }
    rseed((uint32_t)envint("SEED", 7777));
    { const char *v = getenv("JSIG"); if (v) g_jsig = atof(v); }

    arm = envint("ARM", -1);
    arm0 = arm < 0 ? 0 : arm;  arm1 = arm < 0 ? 5 : arm;

    printf("nb101_search -- rotation vs averaging on real NAS-Bench-101 replicates\n");
    printf("%ld catalogued architectures, best true quality %.4f. pop %d, gens %d, %ld runs/arm.\n",
           ntab, g_bestmean, pop, gens, runs);
    printf("inflation = (score the arm selected on) - (a replicate it never saw).\n");
    printf("ROTATE and AVG2 see the SAME two replicates; only averaged vs alternated differs.\n\n");
    { const char *jv = getenv("JSIG"); if (jv) g_jsig = atof(jv); }

    /* Runs outer, arms inner. Seeding once per ARM pairs nothing past run 0, because the arms consume
     * different numbers of random draws and their streams desynchronise immediately. Re-seeding from a
     * run-indexed seed inside the run loop gives run r of every arm the same trial permutation and the
     * same initial population, which is what makes the paired column below valid.
     *
     * The reported outcome is the HELD-OUT replicate, t[perm[2]] and v[perm[2]]. Neither the mean over
     * all three runs nor the mean test accuracy is clean: v and t at the same index come from the same
     * training run, and the mean of three includes the ones the arm selected on, so both inherit the
     * selection noise. Only index perm[2] was unavailable to every arm. */
    printf("  %-30s %16s %16s %18s\n", "arm", "clean TEST +-SE", "clean VAL +-SE",
           "TEST vs FIXED (paired)");
    {
        static double sq[6], sqq[6], sv[6], svv[6], sd[6], dq[6], dqq[6], si[6];
        int a;
        for (r = 0; r < runs; r++) {
            Res o[6];
            for (a = arm0; a <= arm1; a++) {
                rseed((uint32_t)(envint("SEED", 7777) + 2654435761u * (uint32_t)r));
                if (a == 5) run_random(pop, gens, &o[a]);
                else        run_one(a, pop, gens, &o[a]);
                sq[a]  += o[a].qtest;  sqq[a] += o[a].qtest * o[a].qtest;
                sv[a]  += o[a].rtest;  svv[a] += o[a].rtest * o[a].rtest;
                sd[a]  += o[a].distinct;
                si[a]  += o[a].believed - o[a].report;
            }
            for (a = arm0; a <= arm1; a++) {
                double d = o[a].qtest - o[arm0].qtest;
                dq[a] += d; dqq[a] += d * d;
            }
        }
        for (a = arm0; a <= arm1; a++) {
            double m  = sq[a]/runs, v  = sqq[a]/runs - m*m;
            double mv = sv[a]/runs, vv = svv[a]/runs - mv*mv;
            double md = dq[a]/runs, vd = dqq[a]/runs - md*md;
            printf("  %-30s %8.4f+-%.4f %8.4f+-%.4f %+9.4f+-%.4f  infl %6.3f  distinct %5.2f\n",
                   nm[a], m,  sqrt(v  > 0 ? v /runs : 0.0),
                          mv, sqrt(vv > 0 ? vv/runs : 0.0),
                          md, sqrt(vd > 0 ? vd/runs : 0.0),
                   si[a]/runs, sd[a]/runs);
        }
    }
    printf("\nThe paired column is against FIXED on the held-out replicate. RANDOM is the matched-budget\n");
    printf("control: without it the arm ordering cannot be shown to rank searches worth running.\n");
    return 0;
}
