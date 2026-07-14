/* conv2d.c -- an image task where architecture is DECISIVE, not fungible.
 *
 * On the earlier synthetic tasks architecture barely mattered (capacity was
 * fungible), so architecture search and its crossover had nothing to exploit. This
 * probe builds the opposite: a translation-invariant image task, where the right
 * structure -- a 2D convolution -- generalizes and a dense network of similar size
 * does not. The label is the orientation of a small diagonal motif ("\" vs "/")
 * placed at a RANDOM position in a noisy 8x8 image. A convolution detects the motif
 * wherever it sits (shared weights, local receptive field); a dense network must
 * learn it separately at every position, so with limited data it fails to generalize.
 *
 * The network is one 2D conv layer (F filters, 3x3, valid, tanh) -> global max-pool
 * per filter -> a dense sigmoid. We gradient-check the convolutional backprop against
 * finite differences, then compare the conv network's test error to a dense network's
 * on the same data, showing architecture is decisive here.
 *
 * Self-contained C99 (own PRNG, doubles).  Build: make conv2d   Run: ./conv2d
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
static double   rnorm(void) { double u = runif(1e-9, 1), v = runif(0, 1); return sqrt(-2*log(u)) * cos(6.2831853*v); }

#define IMG 8               /* image side */
#define K   3               /* kernel side */
#define POS (IMG - K + 1)   /* conv output side (valid) = 6 */
#define FMAX 8              /* max filters */
#define HMAX 32             /* dense hidden units */

static double sigmoid(double z) { return 1.0 / (1.0 + exp(-z)); }

/* ---- task: a "\" (class 0) or "/" (class 1) 3x3 diagonal at a random spot ---- */
static void sample(double *img, double *label)
{
    int cls = (int)(r32() & 1u), r0, c0, i;
    for (i = 0; i < IMG * IMG; i++) img[i] = 0.15 * rnorm();     /* noise background */
    r0 = (int)(r32() % (uint32_t)(IMG - 2));
    c0 = (int)(r32() % (uint32_t)(IMG - 2));
    for (i = 0; i < 3; i++) {                                    /* draw the diagonal */
        int rr = r0 + i, cc = cls ? (c0 + 2 - i) : (c0 + i);     /* "\" vs "/"        */
        img[rr * IMG + cc] += 1.5;
    }
    *label = (double)cls;
}

/* ---- 2D conv (F filters, KxK, valid, tanh) -> global max-pool -> dense sigmoid -- */
typedef struct {
    int    F;
    double W[FMAX][K][K], b[FMAX];      /* conv filters + biases */
    double V[FMAX], b2;                 /* dense: pooled features -> logit */
    double fmap[FMAX][POS][POS];        /* forward cache: activations */
    double pooled[FMAX]; int ay[FMAX], ax[FMAX];   /* max-pool value + argmax */
} Conv;

static void conv_init(Conv *c, int F)
{
    int f, i, j;
    c->F = F;
    for (f = 0; f < F; f++) {
        for (i = 0; i < K; i++) for (j = 0; j < K; j++) c->W[f][i][j] = runif(-0.5, 0.5);
        c->b[f] = 0.0; c->V[f] = runif(-0.5, 0.5);
    }
    c->b2 = 0.0;
}

static double conv_forward(Conv *c, const double *img)
{
    int f, oy, ox, ky, kx; double logit = c->b2;
    for (f = 0; f < c->F; f++) {
        double best = -1e30; int by = 0, bx = 0;
        for (oy = 0; oy < POS; oy++) for (ox = 0; ox < POS; ox++) {
            double s = c->b[f];
            for (ky = 0; ky < K; ky++) for (kx = 0; kx < K; kx++)
                s += c->W[f][ky][kx] * img[(oy + ky) * IMG + (ox + kx)];
            s = tanh(s); c->fmap[f][oy][ox] = s;
            if (s > best) { best = s; by = oy; bx = ox; }
        }
        c->pooled[f] = best; c->ay[f] = by; c->ax[f] = bx;
        logit += c->V[f] * best;
    }
    return sigmoid(logit);
}

/* one SGD step; returns loss. Only the max-pool argmax position feeds gradient back. */
static double conv_step(Conv *c, const double *img, double t, double lr)
{
    int f, ky, kx;
    double y = conv_forward(c, img);
    double dz = (y - t) * y * (1.0 - y);
    for (f = 0; f < c->F; f++) {
        double dpool = dz * c->V[f];
        double pre = dpool * (1.0 - c->pooled[f] * c->pooled[f]);   /* tanh' at argmax */
        int oy = c->ay[f], ox = c->ax[f];
        c->V[f] -= lr * dz * c->pooled[f];
        for (ky = 0; ky < K; ky++) for (kx = 0; kx < K; kx++)
            c->W[f][ky][kx] -= lr * pre * img[(oy + ky) * IMG + (ox + kx)];
        c->b[f] -= lr * pre;
    }
    c->b2 -= lr * dz;
    return 0.5 * (y - t) * (y - t);
}

static double conv_loss(Conv *c, const double *img, double t)
{ double y = conv_forward(c, img); return 0.5 * (y - t) * (y - t); }

/* ---- gradient check on the conv weights via central finite differences -------- */
static int grad_check(void)
{
    Conv a, num; double img[IMG*IMG], t; int trial, fails = 0;
    rseed(4242); conv_init(&a, 4); { double l; sample(img, &l); t = l; }
    num = a;
    for (trial = 0; trial < 40; trial++) {
        Conv g = a; double *wp_g, *wp_n, eps = 1e-4, ga, gn, tol, before, w0, Lp, Lm;
        int f = (int)(r32() % 4u), ky = (int)(r32() % K), kx = (int)(r32() % K), pick = (int)(r32() % 5u);
        if (pick == 0) { wp_g = &g.V[f]; wp_n = &num.V[f]; }
        else           { wp_g = &g.W[f][ky][kx]; wp_n = &num.W[f][ky][kx]; }
        before = *wp_g; conv_step(&g, img, t, 1.0); ga = -(*wp_g - before);   /* lr=1 */
        w0 = *wp_n; *wp_n = w0 + eps; Lp = conv_loss(&num, img, t);
        *wp_n = w0 - eps; Lm = conv_loss(&num, img, t); *wp_n = w0;
        gn = (Lp - Lm) / (2 * eps);
        tol = 1e-4 + 1e-2 * fabs(ga);
        if (fabs(ga - gn) > tol) { fails++; printf("  mismatch ana=%.6g num=%.6g\n", ga, gn); }
    }
    return fails;
}

/* ---- dense baseline: flatten -> hidden(tanh) -> sigmoid ----------------------- */
typedef struct { int H; double W1[HMAX][IMG*IMG], b1[HMAX], W2[HMAX], b2, a1[HMAX]; } Dense;
static void dense_init(Dense *d, int H)
{
    int h, j; d->H = H;
    for (h = 0; h < H; h++) { for (j = 0; j < IMG*IMG; j++) d->W1[h][j] = runif(-0.3, 0.3);
        d->b1[h] = 0; d->W2[h] = runif(-0.3, 0.3); } d->b2 = 0;
}
static double dense_fwd(Dense *d, const double *x)
{
    int h, j;
    double z = d->b2;
    for (h = 0; h < d->H; h++) {
        double a = d->b1[h];
        for (j = 0; j < IMG*IMG; j++)
            a += d->W1[h][j] * x[j];
        d->a1[h] = tanh(a);
        z += d->W2[h] * d->a1[h];
    }
    return sigmoid(z);
}
static void dense_step(Dense *d, const double *x, double t, double lr)
{
    int h, j;
    double y = dense_fwd(d, x), dz = (y - t) * y * (1 - y);
    for (h = 0; h < d->H; h++) {
        double dpre = dz * d->W2[h] * (1 - d->a1[h]*d->a1[h]);
        d->W2[h] -= lr * dz * d->a1[h];
        for (j = 0; j < IMG*IMG; j++)
            d->W1[h][j] -= lr * dpre * x[j];
        d->b1[h] -= lr * dpre;
    }
  d->b2 -= lr * dz; }

/* train each net on TRAIN fresh samples, measure error on 400 held-out samples */
static void compare(int ntrain, uint32_t seed, double *conv_err, double *dense_err)
{
    Conv c; Dense d; double img[IMG*IMG], t; int e, s;
    rseed(seed); conv_init(&c, 6); dense_init(&d, 24);
    for (e = 0; e < ntrain; e++) { sample(img, &t); conv_step(&c, img, t, 0.05); dense_step(&d, img, t, 0.05); }
    *conv_err = 0; *dense_err = 0;
    for (s = 0; s < 400; s++) { double yc, yd; sample(img, &t);
        yc = conv_forward(&c, img); yd = dense_fwd(&d, img);
        *conv_err += ((yc > 0.5) != (t > 0.5)); *dense_err += ((yd > 0.5) != (t > 0.5)); }
    *conv_err /= 400.0; *dense_err /= 400.0;
}

int main(void)
{
    int fails = grad_check(), i;
    int trains[3] = {2000, 6000, 20000};
    printf("conv2d gradient check: %d/40 mismatches\n", fails);
    printf("translation-invariant motif task, test ERROR RATE (lower better), mean of 8 seeds:\n");
    printf("  train samples   conv(6 filt)   dense(24 hid)\n");
    for (i = 0; i < 3; i++) {
        double cc = 0, dd = 0; int sd;
        for (sd = 1; sd <= 8; sd++) { double ce, de; compare(trains[i], (uint32_t)(sd*7+i), &ce, &de); cc += ce; dd += de; }
        printf("  %-14d  %.3f          %.3f\n", trains[i], cc/8, dd/8);
    }
    return fails ? 1 : 0;
}
