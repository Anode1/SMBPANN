/* bbtest.c -- does one-point crossover beat mutation when building blocks exist?
 *
 * A validation artifact, separate from the neural engine. On the neural task the
 * search's crossover operator (evolve -X) came out on-par with mutation; this
 * program checks whether that is the operator's fault or the landscape's, by
 * running the same idea (elitism + tournament selection + one-point crossover vs
 * mutation-only) on a problem with known, separable building-block structure.
 *
 * The problem is concatenated deceptive trap functions: the genome is M blocks of
 * K bits; a block scores K if all K bits are 1, else K-1 minus its number of ones
 * -- a deceptive gradient that lures a bit-flipping search toward all-zero. The
 * global optimum (all ones) can be reached only by assembling correctly-solved
 * blocks, which is what one-point crossover recombines and mutation cannot climb
 * to. We count how often and how fast each variant reaches the global optimum,
 * over many seeds, and scan the number of blocks M (the amount of structure).
 *
 * Result (see the paper): crossover's advantage grows with M -- the fingerprint
 * of a genuine building-block effect -- proving the operator is sound and the
 * neural on-par result is a property of that (rugged, non-modular) landscape.
 *
 * Build:  make bbtest      Run:  ./bbtest [M] [pop] [K]
 * Self-contained C99 (its own xorshift PRNG); no dependency on the engine.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static uint32_t rng_state;
static uint32_t r32(void) { uint32_t x = rng_state; x ^= x << 13; x ^= x >> 17; x ^= x << 5; rng_state = x; return x; }
static void     rseed(uint32_t s) { rng_state = s ? s : 1u; }
static double   rdbl(void) { return (double)r32() / 4294967296.0; }
static size_t   rbelow(size_t m) { return (m == 0) ? 0 : (size_t)(r32() % (uint32_t)m); }

#define MAXBITS 256
#define MAXPOP  4096

static int K = 4;   /* trap block size (deception strength) */
static int M = 5;   /* number of blocks */
static int L;       /* genome length = K*M */
static int TS = 3;  /* tournament size (selection pressure) */

static int fitness(const unsigned char *g)
{
    int total = 0, b, i;
    for (b = 0; b < M; b++) {
        int u = 0;
        for (i = 0; i < K; i++) u += g[b * K + i];
        total += (u == K) ? K : (K - 1 - u);   /* deceptive: fewer ones scores higher */
    }
    return total;
}

static void randgenome(unsigned char *g) { int i; for (i = 0; i < L; i++) g[i] = (unsigned char)(r32() & 1u); }
static void mutate(unsigned char *g, double pm) { int i; for (i = 0; i < L; i++) if (rdbl() < pm) g[i] ^= 1u; }

static void crossover(unsigned char *c, const unsigned char *a, const unsigned char *b)
{
    int cut = 1 + (int)rbelow((size_t)(L - 1));      /* one-point cut in 1..L-1 */
    int i;
    for (i = 0; i < L; i++) c[i] = (i < cut) ? a[i] : b[i];
}

/* tournament selection: best of TS random individuals (preserves diversity, so
 * different individuals keep different blocks solved for crossover to combine) */
static int tourn(const int *fit, int pop)
{
    int best = (int)rbelow((size_t)pop), t, c;
    for (t = 1; t < TS; t++) { c = (int)rbelow((size_t)pop); if (fit[c] > fit[best]) best = c; }
    return best;
}

/* one GA run; returns the generation the global optimum was first reached, or -1 */
static int run(int pop, int gens, int elite, double pc, double pm, uint32_t seed)
{
    static unsigned char P[MAXPOP][MAXBITS], Q[MAXPOP][MAXBITS];
    int fit[MAXPOP];
    int g, i, best;

    rseed(seed);
    for (i = 0; i < pop; i++) randgenome(P[i]);
    for (g = 1; g <= gens; g++) {
        for (i = 0; i < pop; i++) fit[i] = fitness(P[i]);
        best = 0;
        for (i = 1; i < pop; i++) if (fit[i] > fit[best]) best = i;
        if (fit[best] == K * M) return g;

        for (i = 0; i < elite; i++) memcpy(Q[i], P[best], (size_t)L);   /* keep the best */
        for (i = elite; i < pop; i++) {
            if (pc > 0.0 && rdbl() < pc)
                crossover(Q[i], P[tourn(fit, pop)], P[tourn(fit, pop)]);
            else
                memcpy(Q[i], P[tourn(fit, pop)], (size_t)L);
            mutate(Q[i], pm);
        }
        for (i = 0; i < pop; i++) memcpy(P[i], Q[i], (size_t)L);
    }
    return -1;
}

int main(int argc, char **argv)
{
    int    pop = 1000, gens = 500, elite = 1, runs = 200;
    double pm;
    int    cond, r, reached;
    long   sumg;

    if (argc > 1) M   = atoi(argv[1]);
    if (argc > 2) pop = atoi(argv[2]);
    if (argc > 3) K   = atoi(argv[3]);
    if (pop > MAXPOP) pop = MAXPOP;
    if (K * M > MAXBITS) M = MAXBITS / K;
    L  = K * M;
    pm = 1.0 / (double)L;

    printf("concatenated deceptive trap: %d blocks x %d bits = %d | pop %d, elite %d, "
           "cap %d gens, %d seeds, tournament %d, mut=1/L\n",
           M, K, L, pop, elite, gens, runs, TS);
    for (cond = 0; cond < 2; cond++) {
        double pc = cond ? 0.9 : 0.0;
        reached = 0; sumg = 0;
        for (r = 1; r <= runs; r++) {
            int gr = run(pop, gens, elite, pc, pm, (uint32_t)r);
            if (gr > 0) { reached++; sumg += gr; }
        }
        printf("%-20s reached global optimum %3d/%d",
               cond ? "crossover(0.9)+mut:" : "mutation-only:", reached, runs);
        if (reached) printf("   mean gens=%.1f", (double)sumg / reached);
        printf("\n");
    }
    return 0;
}
