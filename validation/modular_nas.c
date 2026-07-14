/* modular_nas.c -- does crossover of a MODULAR architecture assemble a solution
 * from parts, the way it assembled trap blocks in bbtest?
 *
 * The neural on-par result came from a sequential-layer genome, whose layers are
 * compositionally coupled (no separable building blocks). This probe gives the
 * search a genome that DOES have them: a set of parallel BRANCHES. The network is
 * B independent branches, each a small MLP reading the whole input; their linear
 * outputs are summed and passed through a sigmoid per output. Branches do not feed
 * each other, so a branch that solves one output-component is an independent block
 * -- and (unlike architecture-only NAS) a branch carries its trained weights, so
 * crossover transfers a SOLVED part, exactly as trap crossover transferred a
 * solved block.
 *
 * The task is additively separable and multi-output: the d inputs split into G
 * groups, and output i depends only on group i. Solving all G outputs needs G
 * specialised branches; mutation must grow and train them one lineage at a time,
 * while crossover can combine solved branches from two parents.
 *
 * This file is milestone 1: the branch network, its back-propagation, and a
 * finite-difference gradient check. Self-contained C99 (own PRNG, doubles).
 *
 * Build:  make modnas      Run:  ./modnas
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

#define DMAX 24        /* max inputs  */
#define GMAX 8         /* max outputs / groups */
#define HMAX 16        /* max hidden units per branch */
#define BMAX 16        /* max branches */

/* ---- one branch: input(d) -> hidden(H, tanh) -> output(G, linear) ---------- */
typedef struct {
    int    H;
    double W1[HMAX][DMAX], b1[HMAX];     /* input -> hidden  */
    double W2[GMAX][HMAX], b2[GMAX];     /* hidden -> output */
    double a1[HMAX];                     /* forward cache: hidden activations */
    double o[GMAX];                      /* forward cache: linear outputs */
} Branch;

/* ---- the modular network: B branches summed, then a per-output sigmoid ------ */
typedef struct {
    int    B, d, g;
    Branch br[BMAX];
    double z[GMAX], y[GMAX];             /* summed pre-activation, sigmoid output */
} Net;

static double sigmoid(double z) { return 1.0 / (1.0 + exp(-z)); }

static void branch_init(Branch *br, int H, int d, int g)
{
    int h, j, i;
    br->H = H;
    for (h = 0; h < H; h++) {
        for (j = 0; j < d; j++) br->W1[h][j] = runif(-2.4 / d, 2.4 / d);
        br->b1[h] = 0.0;
    }
    for (i = 0; i < g; i++) {
        for (h = 0; h < H; h++) br->W2[i][h] = runif(-2.4 / H, 2.4 / H);
        br->b2[i] = 0.0;
    }
}

static void net_init(Net *n, int B, int H, int d, int g)
{
    int b;
    n->B = B; n->d = d; n->g = g;
    for (b = 0; b < B; b++) branch_init(&n->br[b], H, d, g);
}

/* forward pass; fills per-branch caches and n->y (the prediction) */
static void net_forward(Net *n, const double *x)
{
    int b, h, j, i;
    for (i = 0; i < n->g; i++) n->z[i] = 0.0;
    for (b = 0; b < n->B; b++) {
        Branch *br = &n->br[b];
        for (h = 0; h < br->H; h++) {
            double s = br->b1[h];
            for (j = 0; j < n->d; j++) s += br->W1[h][j] * x[j];
            br->a1[h] = tanh(s);
        }
        for (i = 0; i < n->g; i++) {
            double s = br->b2[i];
            for (h = 0; h < br->H; h++) s += br->W2[i][h] * br->a1[h];
            br->o[i] = s;
            n->z[i] += s;                /* branches sum */
        }
    }
    for (i = 0; i < n->g; i++) n->y[i] = sigmoid(n->z[i]);
}

static double net_loss(Net *n, const double *x, const double *t)
{
    int i; double L = 0.0;
    net_forward(n, x);
    for (i = 0; i < n->g; i++) { double e = n->y[i] - t[i]; L += 0.5 * e * e; }
    return L;
}

/* one SGD step on (x,t); learning rate lr. Returns the loss before the step. */
static double net_train_step(Net *n, const double *x, const double *t, double lr)
{
    double dz[GMAX];
    int b, h, j, i;
    double L = net_loss(n, x, t);        /* also runs forward, filling caches */

    for (i = 0; i < n->g; i++)
        dz[i] = (n->y[i] - t[i]) * n->y[i] * (1.0 - n->y[i]);   /* through sigmoid */

    for (b = 0; b < n->B; b++) {
        Branch *br = &n->br[b];
        double da1[HMAX];
        for (h = 0; h < br->H; h++) da1[h] = 0.0;
        /* output layer: o_i = b2_i + sum_h W2[i][h] a1_h ; dL/do_i = dz_i */
        for (i = 0; i < n->g; i++) {
            for (h = 0; h < br->H; h++) {
                da1[h]        += dz[i] * br->W2[i][h];
                br->W2[i][h]  -= lr * dz[i] * br->a1[h];
            }
            br->b2[i] -= lr * dz[i];
        }
        /* hidden layer: a1_h = tanh(pre) ; dpre = da1 * (1 - a1^2) */
        for (h = 0; h < br->H; h++) {
            double dpre = da1[h] * (1.0 - br->a1[h] * br->a1[h]);
            for (j = 0; j < n->d; j++) br->W1[h][j] -= lr * dpre * x[j];
            br->b1[h] -= lr * dpre;
        }
    }
    return L;
}

/* ---- finite-difference gradient check on a random weight ------------------- */
static int grad_check(void)
{
    Net n; double x[DMAX], t[GMAX];
    int d = 6, g = 3, H = 5, B = 3, j, i, trial, fails = 0;
    rseed(12345);
    net_init(&n, B, H, d, g);
    for (j = 0; j < d; j++) x[j] = runif(-1.0, 1.0);
    for (i = 0; i < g; i++) t[i] = (runif(0, 1) > 0.5) ? 1.0 : 0.0;

    /* analytic gradient of the loss wrt each weight = -(update)/lr with lr=1,
     * momentum 0. We read it by taking a step with lr=1 on a COPY and comparing
     * the weight delta to a central finite difference of the loss. */
    for (trial = 0; trial < 40; trial++) {
        Net a = n, num = n;
        double eps = 1e-4, g_ana, g_num, tol;
        int b = (int)(r32() % (uint32_t)B);
        int which = (int)(r32() % 2u);   /* 0: a W1 weight, 1: a W2 weight */
        double *wp_ana, *wp_num;
        if (which == 0) { int hh=(int)(r32()%(uint32_t)H), jj=(int)(r32()%(uint32_t)d);
                          wp_ana=&a.br[b].W1[hh][jj]; wp_num=&num.br[b].W1[hh][jj]; }
        else            { int ii=(int)(r32()%(uint32_t)g), hh=(int)(r32()%(uint32_t)H);
                          wp_ana=&a.br[b].W2[ii][hh]; wp_num=&num.br[b].W2[ii][hh]; }

        {   double before = *wp_ana;
            net_train_step(&a, x, t, 1.0);          /* lr=1: delta = -dL/dw */
            g_ana = -(*wp_ana - before);
        }
        {   double w0 = *wp_num, Lp, Lm;
            *wp_num = w0 + eps; Lp = net_loss(&num, x, t);
            *wp_num = w0 - eps; Lm = net_loss(&num, x, t);
            *wp_num = w0;
            g_num = (Lp - Lm) / (2.0 * eps);
        }
        tol = 1e-4 + 1e-2 * fabs(g_ana);
        if (fabs(g_ana - g_num) > tol) {
            fails++;
            printf("  grad mismatch: ana=%.6g num=%.6g (which=%d)\n", g_ana, g_num, which);
        }
    }
    return fails;
}

/* ---- quick learnability demo: does a branch net train on a modular task? ---- */
static void task_sample(const double W[GMAX][DMAX], int d, int g, int groupsz,
                        double freq, double *x, double *t)
{
    int i, j;
    for (j = 0; j < d; j++) x[j] = runif(-1.0, 1.0);
    for (i = 0; i < g; i++) {           /* output i depends only on group i */
        double s = 0.0;
        for (j = 0; j < groupsz; j++) s += W[i][j] * x[i * groupsz + j];
        t[i] = (sin(freq * s) > 0.0) ? 1.0 : 0.0;
    }
}

static double demo_train(int B, int H, uint32_t seed)
{
    Net n; double W[GMAX][DMAX];
    int d = 6, g = 3, groupsz = 2, i, j, ep, s;
    double x[DMAX], t[GMAX], err;
    rseed(seed);
    for (i = 0; i < g; i++) for (j = 0; j < groupsz; j++) W[i][j] = runif(-1.5, 1.5);
    net_init(&n, B, H, d, g);
    for (ep = 0; ep < 4000; ep++) {
        task_sample(W, d, g, groupsz, 3.0, x, t);
        net_train_step(&n, x, t, 0.1);
    }
    err = 0.0;
    for (s = 0; s < 400; s++) {
        task_sample(W, d, g, groupsz, 3.0, x, t);
        net_forward(&n, x);
        for (i = 0; i < g; i++) { double e = n.y[i] - t[i]; err += e * e; }
    }
    return err / (400.0 * g);
}

int main(void)
{
    int fails = grad_check();
    int Hs[3] = {2, 3, 6}, Bs[4] = {1, 2, 3, 5}, hi, bi;
    printf("gradient check: %d/%d mismatches\n", fails, 40);
    printf("modular task (3 groups) test-MSE, rows=branch hidden width H, cols=branch count B:\n");
    printf("        B=1     B=2     B=3     B=5\n");
    for (hi = 0; hi < 3; hi++) {
        printf("  H=%d:", Hs[hi]);
        for (bi = 0; bi < 4; bi++)
            printf("  %.4f", demo_train(Bs[bi], Hs[hi], 7));
        printf("\n");
    }
    return fails ? 1 : 0;
}
