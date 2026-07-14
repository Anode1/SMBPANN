/* modevo.c -- does crossover WITH WEIGHT INHERITANCE help, where architecture-only
 * crossover did not?
 *
 * Earlier probes showed crossover of an architecture-only genome cannot help a
 * neural search: capacity is fungible and the solution lives in the weights, which a
 * retrain-from-scratch search does not inherit, so the genome carries nothing to
 * transfer. This probe adds the missing ingredient. A candidate is a MODULAR net:
 * G independent sub-networks, one per output-component, each a small MLP whose scalar
 * logit feeds that component's sigmoid. Components are hard (a high-frequency sine of
 * their input group), so training a subnet from a random init succeeds only
 * sometimes -- a lottery. A candidate CARRIES its subnets' trained weights across
 * generations (weight inheritance) and fine-tunes them further, so a useful
 * combination keeps evolving instead of restarting.
 *
 * A solved subnet is a self-contained trained part -- an independent building block.
 * Crossover recombines whole subnets (with their weights) from two parents, spreading
 * solved components across the population; mutation must re-roll the init lottery for
 * a stuck subnet within one lineage. We race crossover against mutation-only, both
 * with inheritance and fine-tuning, and count generations to solve all components.
 *
 * Self-contained C99 (own PRNG, doubles).  Build: make modevo   Run: ./modevo [G]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static uint32_t rng_state;
static uint32_t r32(void) { uint32_t x = rng_state; x ^= x << 13; x ^= x >> 17; x ^= x << 5; rng_state = x; return x; }
static void     rseed(uint32_t s) { rng_state = s ? s : 1u; }
static double   runif(double a, double b) { return a + (b - a) * ((double)r32() / 4294967296.0); }
static uint32_t rbelow(uint32_t m) { return m ? (r32() % m) : 0u; }

#define DMAX 24
#define GMAX 8
#define HMAX 12
#define POPMAX 128

static int d = 12, G = 4, groupsz;   /* inputs, components, inputs per component */
static int H = 8;                    /* hidden units per subnet */
static double FREQ = 4.0;            /* sine frequency (deception/hardness) */

/* one component subnet: input(d) -> hidden(H, tanh) -> 1 logit */
typedef struct {
    double W1[HMAX][DMAX], b1[HMAX];
    double W2[HMAX], b2;
    double a1[HMAX];                 /* forward cache */
} Sub;

typedef struct { Sub sub[GMAX]; double cerr[GMAX]; double fit; } Indiv;

static double sigmoid(double z) { return 1.0 / (1.0 + exp(-z)); }

/* each subnet reads only its own input group (groupsz inputs) */
static void sub_init(Sub *s)
{
    int h, j;
    for (h = 0; h < H; h++) {
        for (j = 0; j < groupsz; j++) s->W1[h][j] = runif(-2.4 / groupsz, 2.4 / groupsz);
        s->b1[h] = 0.0; s->W2[h] = runif(-2.4 / H, 2.4 / H);
    }
    s->b2 = 0.0;
}

static double sub_logit(Sub *s, const double *xg)
{
    int h, j; double z = s->b2;
    for (h = 0; h < H; h++) {
        double a = s->b1[h];
        for (j = 0; j < groupsz; j++) a += s->W1[h][j] * xg[j];
        s->a1[h] = tanh(a);
        z += s->W2[h] * s->a1[h];
    }
    return z;
}

/* one SGD step of subnet toward target t on its group xg (through the sigmoid) */
static void sub_step(Sub *s, const double *xg, double t, double lr)
{
    int h, j;
    double y = sigmoid(sub_logit(s, xg));
    double dz = (y - t) * y * (1.0 - y);
    for (h = 0; h < H; h++) {
        double dpre = dz * s->W2[h] * (1.0 - s->a1[h] * s->a1[h]);
        s->W2[h] -= lr * dz * s->a1[h];
        for (j = 0; j < groupsz; j++) s->W1[h][j] -= lr * dpre * xg[j];
        s->b1[h] -= lr * dpre;
    }
    s->b2 -= lr * dz;
}

/* ---- task: component i is a high-frequency sine of its own input group ------ */
static double TW[GMAX][DMAX];
static void task_init(uint32_t seed)
{
    int i, j; rseed(seed);
    groupsz = d / G;
    for (i = 0; i < G; i++) for (j = 0; j < groupsz; j++) TW[i][j] = runif(-1.5, 1.5);
}
static void sample(double *x, double *t)
{
    int i, j;
    for (j = 0; j < d; j++) x[j] = runif(-1.0, 1.0);
    for (i = 0; i < G; i++) {
        double s = 0.0;
        for (j = 0; j < groupsz; j++) s += TW[i][j] * x[i * groupsz + j];
        t[i] = (sin(FREQ * s) > 0.0) ? 1.0 : 0.0;
    }
}

/* fine-tune each subnet FT steps, then measure per-component validation MSE */
static void evaluate(Indiv *v, int ft)
{
    int e, i, s; double x[DMAX], t[GMAX];
    for (e = 0; e < ft; e++) { sample(x, t); for (i = 0; i < G; i++) sub_step(&v->sub[i], x + i * groupsz, t[i], 0.15); }
    for (i = 0; i < G; i++) v->cerr[i] = 0.0;
    for (s = 0; s < 200; s++) {
        sample(x, t);
        for (i = 0; i < G; i++) { double y = sigmoid(sub_logit(&v->sub[i], x + i * groupsz)); double d0 = y - t[i]; v->cerr[i] += d0 * d0; }
    }
    v->fit = 0.0;
    for (i = 0; i < G; i++) { v->cerr[i] /= 200.0; v->fit += v->cerr[i]; }
    v->fit /= G;
}

static int tourn(const Indiv *pop, int P)
{
    int a = (int)rbelow((uint32_t)P), b = (int)rbelow((uint32_t)P);
    return (pop[a].fit < pop[b].fit) ? a : b;   /* lower error wins */
}

static double g_bestsum; static int g_bestcnt;   /* calibration: best error reached */

/* one run; returns generation the whole task is solved (mean err < target), or -1 */
static int run(int P, int gens, int ft, int use_xover, double target, uint32_t seed)
{
    static Indiv pop[POPMAX], nxt[POPMAX];
    int g, i, k, best;
    double rbest = 1e9;
    rseed(seed ^ 0x9e3779b9u);
    for (i = 0; i < P; i++) { for (k = 0; k < G; k++) sub_init(&pop[i].sub[k]); }

    for (g = 1; g <= gens; g++) {
        best = 0;
        for (i = 0; i < P; i++) { evaluate(&pop[i], ft); if (pop[i].fit < pop[best].fit) best = i; }
        if (pop[best].fit < rbest) rbest = pop[best].fit;
        if (pop[best].fit < target) { g_bestsum += rbest; g_bestcnt++; return g; }

        nxt[0] = pop[best];                       /* elitism */
        for (i = 1; i < P; i++) {
            if (use_xover) {
                /* uniform crossover over subnets, inheriting weights: each
                 * component comes (whole, trained) from the better of two parents */
                int pa = tourn(pop, P), pb = tourn(pop, P);
                for (k = 0; k < G; k++)
                    nxt[i].sub[k] = (pop[pa].cerr[k] <= pop[pb].cerr[k]) ? pop[pa].sub[k] : pop[pb].sub[k];
            } else {
                /* mutation-only: inherit one parent whole, then re-roll (re-init)
                 * its worst subnet -- a fresh init lottery for the stuck component */
                int pa = tourn(pop, P), worst = 0;
                nxt[i] = pop[pa];
                for (k = 1; k < G; k++) if (pop[pa].cerr[k] > pop[pa].cerr[worst]) worst = k;
                sub_init(&nxt[i].sub[worst]);
            }
        }
        for (i = 0; i < P; i++) pop[i] = nxt[i];
    }
    g_bestsum += rbest; g_bestcnt++;
    return -1;
}

int main(int argc, char **argv)
{
    int P = 30, gens = 120, ft = 300, runs = 60, cond, r, reached; long sumg;
    double target = 0.08;
    if (argc > 1) G = atoi(argv[1]);
    if (argc > 2) FREQ = atof(argv[2]);
    groupsz = 3;                                  /* fixed per-component difficulty */
    d = G * groupsz;                              /* so only the NUMBER of blocks varies */
    task_init(777);                               /* one fixed task family; seeds vary the search */
    printf("modular weight-inheriting search: G=%d components (freq %.1f), d=%d, pop %d, "
           "fine-tune %d/gen, cap %d gens, %d seeds, solve target mean-MSE<%.2f\n",
           G, FREQ, d, P, ft, gens, runs, target);
    for (cond = 0; cond < 2; cond++) {
        reached = 0; sumg = 0; g_bestsum = 0.0; g_bestcnt = 0;
        for (r = 1; r <= runs; r++) {
            int gr = run(P, gens, ft, cond, target, (uint32_t)r);
            if (gr > 0) { reached++; sumg += gr; }
        }
        printf("%-24s solved %3d/%d", cond ? "crossover(inherit):" : "mutation(re-init):", reached, runs);
        if (reached) printf("   mean gens=%.1f", (double)sumg / reached);
        printf("   (mean best mean-MSE reached %.3f)", g_bestsum / g_bestcnt);
        printf("\n");
    }
    return 0;
}
