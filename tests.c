/* tests.c -- in-process unit tests for the SMBPANN engine.
 *
 * Built with -DUNIT_TEST (see the Makefile), which also compiles out main.c's
 * main(), so this file provides the test entry point. Style follows AIS: linear,
 * inline, one comment per check saying what it verifies. The gate is the XOR
 * regression -- if backprop through a hidden layer learns the canonical
 * non-separable function, the forward pass, the delta rule, momentum, and the
 * allocation lifecycle are all exercised together.
 */
#ifdef UNIT_TEST

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "act.h"
#include "arena.h"
#include "ckpt.h"
#include "conv2f.h"
#include "common.h"
#include "data.h"
#include "genome.h"
#include "net.h"
#include "rng.h"
#include "train.h"

static int t_run, t_fail;

#define CHECK(cond, msg)                              \
    do {                                              \
        t_run++;                                      \
        if (cond) {                                   \
            printf("ok   %s\n", msg);                 \
        } else {                                      \
            t_fail++;                                 \
            printf("FAIL %s\n", msg);                 \
        }                                             \
    } while (0)

int main(void)
{
    /* rng: reproducible, in range, and 0-seed remapped (not stuck at zero) */
    {
        Rng g, h, z;
        smb_real u;
        rng_seed(&g, 42);
        rng_seed(&h, 42);
        CHECK(rng_u32(&g) == rng_u32(&h), "rng: same seed gives the same sequence");
        u = rng_uniform(&g, -1.0f, 1.0f);
        CHECK(u >= -1.0f && u < 1.0f, "rng: uniform stays in [lo, hi)");
        rng_seed(&z, 0);
        CHECK(rng_u32(&z) != 0, "rng: seed 0 is remapped, not degenerate");
    }

    /* act: sigmoid pins, saturation, and the a*(1-a) derivative */
    CHECK(act_sigmoid(0.0f) > 0.4999f && act_sigmoid(0.0f) < 0.5001f,
          "act: sigmoid(0) = 0.5");
    CHECK(act_sigmoid(100.0f) <= 1.0f && act_sigmoid(-100.0f) >= 0.0f,
          "act: sigmoid saturates within [0,1]");
    CHECK(act_sigmoid_deriv(0.5f) > 0.2499f && act_sigmoid_deriv(0.5f) < 0.2501f,
          "act: sigmoid' at a=0.5 is 0.25");

    /* net_new rejects a NULL dims array */
    CHECK(net_new(NULL, 3) == NULL, "net_new rejects NULL dims");

    /* net_new rejects fewer than two layers (need input + output) */
    {
        size_t d[1] = { 4 };
        CHECK(net_new(d, 1) == NULL, "net_new rejects < 2 layers");
    }

    /* net_new rejects a zero-width layer */
    {
        size_t d[3] = { 2, 0, 1 };
        CHECK(net_new(d, 3) == NULL, "net_new rejects a zero-width layer");
    }

    /* net_nweights counts every weight and no bias: 2*3 + 3*1 = 9 */
    {
        size_t d[3] = { 2, 3, 1 };
        Net *n = net_new(d, 3);
        CHECK(n != NULL && net_nweights(n) == 9, "net_nweights counts 2-3-1 as 9");
        net_free(n);
    }

    /* forward pass is deterministic and the sigmoid output lies in (0,1) */
    {
        size_t d[3] = { 2, 3, 1 };
        Net *n = net_new(d, 3);
        Rng r;
        smb_real x[2] = { 0.5f, 0.25f };
        smb_real first;
        const smb_real *y;
        rng_seed(&r, 7);
        net_init(n, &r);
        y = net_forward(n, x);
        first = y[0];
        y = net_forward(n, x);
        CHECK(first == y[0], "forward pass is deterministic");
        CHECK(first > 0.0f && first < 1.0f, "sigmoid output is in (0,1)");
        net_free(n);
    }

    /* the regression: a 2-4-1 net learns XOR below an error threshold and
     * classifies all four patterns correctly on the 0.5 boundary */
    {
        size_t d[3] = { 2, 4, 1 };
        Net *n = net_new(d, 3);
        Rng r;
        Trainer *t;
        const smb_real X[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };
        const smb_real D[4][1] = { {0},   {1},   {1},   {0}   };
        smb_real E = 0;
        int ep, i, correct = 1;

        rng_seed(&r, 1);
        net_init(n, &r);
        t = trainer_new(n, 0.5f, 0.9f);
        for (ep = 0; ep < 20000; ep++) {
            E = 0;
            for (i = 0; i < 4; i++)
                E += trainer_learn(t, X[i], D[i]);
        }
        CHECK(E < 0.02f, "XOR trains to sse < 0.02");
        for (i = 0; i < 4; i++) {
            const smb_real *y = net_forward(n, X[i]);
            if ((y[0] > 0.5f ? 1 : 0) != (int)D[i][0])
                correct = 0;
        }
        CHECK(correct, "XOR classifies all four patterns correctly");
        trainer_free(t);
        net_free(n);
    }

    /* arena: allocate via the pool, roll back to a mark, reset, and overflow */
    {
        Arena a;
        if (arena_init(&a, 64 * 1024) == 0) {
            smb_real *x, *y;
            size_t u1, m;
            CHECK(arena_used(&a) == 0, "arena starts empty");
            x = arena_alloc(&a, 100 * sizeof *x, sizeof *x);
            CHECK(x != NULL && arena_used(&a) > 0, "arena allocation advances the mark");
            x[49] = 1.5f;
            u1 = arena_used(&a);
            m = arena_mark(&a);
            y = arena_alloc(&a, 200 * sizeof *y, sizeof *y);
            y[199] = 2.0f;
            CHECK(arena_used(&a) > u1, "arena second allocation grows past the mark");
            CHECK(x[49] == 1.5f && y[199] == 2.0f, "arena buffers do not overlap");
            arena_release(&a, m);
            CHECK(arena_used(&a) == u1, "arena release rolls back to the mark");
            arena_reset(&a);
            CHECK(arena_used(&a) == 0, "arena reset empties the arena");
            CHECK(arena_alloc(&a, 1000000 * sizeof(smb_real), sizeof(smb_real)) == NULL,
                  "arena over-capacity allocation returns NULL");
            arena_free(&a);
        } else {
            CHECK(0, "arena_init succeeds");
        }
    }

    /* data: load a plain-text file (comment + integer tokens), then split it */
    {
        const char *path = "smbpann_data_test.tmp";
        FILE *tf = fopen(path, "w");
        Dataset ds;
        if (tf != NULL) {
            fputs("# xor sample set\n0 0 0\n0 1 1\n1 0 1\n1 1 0\n", tf);
            fclose(tf);
        }
        if (dataset_load(&ds, path, 2, 1) == 0) {
            const smb_real *in = dataset_input(&ds, 1);   /* row "0 1 1" */
            const smb_real *tg = dataset_target(&ds, 1);
            Rng r;
            Split sp;
            CHECK(ds.nsamples == 4, "data loads all rows, skips the comment");
            CHECK(ds.ninput == 2 && ds.noutput == 1, "data input/output widths");
            CHECK(in[0] == 0.0f && in[1] == 1.0f && tg[0] == 1.0f,
                  "data row values parsed (integer tokens accepted)");
            rng_seed(&r, 5);
            if (split_make(&sp, &ds, 0.75f, &r) == 0) {
                int seen[4] = { 0, 0, 0, 0 };
                int perm_ok = 1;
                size_t p;
                CHECK(split_train_count(&sp) == 3 && split_test_count(&sp) == 1,
                      "data 75% split -> 3 train, 1 test");
                for (p = 0; p < split_train_count(&sp); p++)
                    seen[split_train_index(&sp, p)] = 1;
                for (p = 0; p < split_test_count(&sp); p++)
                    seen[split_test_index(&sp, p)] = 1;
                for (p = 0; p < 4; p++)
                    if (!seen[p])
                        perm_ok = 0;
                CHECK(perm_ok, "data split covers every sample exactly once");
                split_free(&sp);
            } else {
                CHECK(0, "split_make succeeds");
            }
            dataset_free(&ds);
        } else {
            CHECK(0, "dataset_load succeeds");
        }
        remove(path);
    }

    /* genome: random within bounds, format/parse round-trip, safe mutation */
    {
        Rng r;
        Genome g, g2;
        char buf[256];
        int i, ok = 1;
        rng_seed(&r, 3);
        genome_random(&g, 2, 1, 3, 16, &r);
        CHECK(g.n >= 2 && g.n <= 5 && g.dim[0] == 2 && g.dim[g.n - 1] == 1,
              "genome: random keeps fixed endpoints and layer bound");
        genome_format(&g, buf, sizeof buf);
        CHECK(genome_parse(&g2, buf) == 0 && g2.n == g.n
              && g2.dim[0] == 2 && g2.dim[g2.n - 1] == 1,
              "genome: format/parse round-trips");
        for (i = 0; i < 200; i++) {
            Net *nn;
            genome_mutate(&g, 3, 16, &r);
            if (g.dim[0] != 2 || g.dim[g.n - 1] != 1 || g.n < 2 || g.n > 5)
                ok = 0;
            /* every mutated genome (dense or conv) must build a net whose
             * per-layer widths match the genome's -- the recompute invariant */
            nn = net_build(g.dim, g.kind, g.nfilt, g.ksize, g.n);
            if (nn == NULL) {
                ok = 0;
            } else {
                size_t j;
                for (j = 0; j < g.n; j++)
                    if (nn->dim[j] != g.dim[j])
                        ok = 0;
                net_free(nn);
            }
        }
        CHECK(ok, "genome: mutation stays valid and builds a matching net");
    }

    /* genome: self-adaptive reproduction keeps endpoints and a bounded rate */
    {
        Rng r;
        Genome parent, child;
        int i, ok = 1;
        rng_seed(&r, 9);
        genome_random(&parent, 2, 1, 3, 16, &r);
        parent.rate = 2.0f;
        for (i = 0; i < 200; i++) {
            genome_reproduce(&child, &parent, 3, 16, &r);
            if (child.dim[0] != 2 || child.dim[child.n - 1] != 1
                || child.n < 2 || child.n > 5)
                ok = 0;
            if (!(child.rate > 0.0f) || child.rate > 100.0f)
                ok = 0;
            parent = child;   /* let the rate drift over a lineage */
        }
        CHECK(ok, "genome: self-adaptive reproduce keeps endpoints and a bounded rate");
    }

    /* genome: hyper-parameters start in range, survive the spec, and stay
     * bounded as they co-evolve */
    {
        Rng r;
        Genome g, g2, c;
        char buf[256];
        int i, ok = 1;
        rng_seed(&r, 11);
        genome_random(&g, 2, 1, 3, 16, &r);
        CHECK(g.lrate >= 0.1f && g.lrate < 0.8f
              && g.momentum >= 0.5f && g.momentum < 0.95f
              && g.activation >= 0 && g.activation < ACT_COUNT,
              "genome: random hyper-parameters start in range");
        genome_format(&g, buf, sizeof buf);
        CHECK(genome_parse(&g2, buf) == 0 && g2.activation == g.activation,
              "genome: spec round-trips the activation gene");
        for (i = 0; i < 300; i++) {
            genome_reproduce(&c, &g, 3, 16, &r);
            if (c.lrate < 0.01f || c.lrate > 2.0f
                || c.momentum < 0.0f || c.momentum > 0.99f
                || c.activation < 0 || c.activation >= ACT_COUNT)
                ok = 0;
            g = c;
        }
        CHECK(ok, "genome: co-evolved hyper-parameters stay bounded");
    }

    /* genome: crossover splices two parents into a buildable child that keeps the
     * problem's fixed endpoints, respects the depth bound, and stays in range */
    {
        Rng r;
        Genome a, b, c;
        int i, ok = 1;
        rng_seed(&r, 21);
        for (i = 0; i < 300; i++) {
            Net *nn;
            genome_random(&a, 3, 1, 3, 16, &r);
            genome_random(&b, 3, 1, 3, 16, &r);
            genome_crossover(&c, &a, &b, 3, 16, &r);
            if (c.n < 2 || c.n > 5 || c.dim[0] != 3 || c.dim[c.n - 1] != 1)
                ok = 0;
            if (!(c.rate > 0.0f) || c.lrate < 0.01f || c.lrate > 2.0f
                || c.momentum < 0.0f || c.momentum > 0.99f
                || c.activation < 0 || c.activation >= ACT_COUNT)
                ok = 0;
            /* the spliced child must build a net whose widths match its genes */
            nn = net_build(c.dim, c.kind, c.nfilt, c.ksize, c.n);
            if (nn == NULL) {
                ok = 0;
            } else {
                size_t j;
                for (j = 0; j < c.n; j++)
                    if (nn->dim[j] != c.dim[j])
                        ok = 0;
                net_free(nn);
            }
        }
        CHECK(ok, "genome: crossover splices a valid, buildable child");
    }

    /* conv1d: forward (weight sharing), gradient correctness, and learning */
    {
        Rng r;

        /* forward with known weights: one filter, kernel {1,1}, so each local
         * window sums adjacent inputs -- the same two weights reused at each spot */
        {
            size_t   dims[3]  = { 4, 0, 1 };
            int      kind[3]  = { LAYER_DENSE, LAYER_CONV, LAYER_DENSE };
            size_t   nfilt[3] = { 0, 1, 0 };
            size_t   ksize[3] = { 0, 2, 0 };
            Net     *n = net_build(dims, kind, nfilt, ksize, 3);
            smb_real x[4] = { 1, 2, 3, 4 };
            int      ok = (n != NULL);
            if (ok) {
                n->w[1][0] = 1.0f; n->w[1][1] = 1.0f; n->b[1][0] = 0.0f;
                net_forward(n, x);
                ok = (n->dim[1] == 3)                       /* 1 filter x 3 pos */
                   && fabsf(n->a[1][0] - act_sigmoid(3.0f)) < 1e-5f
                   && fabsf(n->a[1][1] - act_sigmoid(5.0f)) < 1e-5f
                   && fabsf(n->a[1][2] - act_sigmoid(7.0f)) < 1e-5f;
                net_free(n);
            }
            CHECK(ok, "conv: forward shares weights over local windows");
        }

        /* numerical gradient check: at rate 1 with no momentum the applied update
         * is exactly -dE/dw, so compare it to central finite differences of E */
        {
            size_t dims[3]  = { 6, 0, 1 };
            int    kind[3]  = { LAYER_DENSE, LAYER_CONV, LAYER_DENSE };
            size_t nfilt[3] = { 0, 2, 0 };
            size_t ksize[3] = { 0, 3, 0 };
            Net   *n = net_build(dims, kind, nfilt, ksize, 3);
            int    ok = (n != NULL);
            if (ok) {
                smb_real  x[6] = { 0.3f, -0.1f, 0.7f, 0.2f, -0.5f, 0.4f };
                smb_real  d[1] = { 0.8f };
                smb_real  sw1[8], sw2[16], sb1[8], sb2[8];
                size_t    nw1 = net_layer_wsize(n, 1), nw2 = net_layer_wsize(n, 2);
                size_t    nb1 = net_layer_bsize(n, 1), nb2 = net_layer_bsize(n, 2);
                double    eps = 0.01;
                Trainer  *t;
                size_t    idx;
                rng_seed(&r, 5);
                net_init(n, &r);
                memcpy(sw1, n->w[1], nw1 * sizeof(smb_real));
                memcpy(sw2, n->w[2], nw2 * sizeof(smb_real));
                memcpy(sb1, n->b[1], nb1 * sizeof(smb_real));
                memcpy(sb2, n->b[2], nb2 * sizeof(smb_real));
                t = trainer_new(n, 1.0f, 0.0f);
                (void)trainer_learn(t, x, d);   /* applied dw = -dE/dw (rate 1) */
                memcpy(n->w[1], sw1, nw1 * sizeof(smb_real));   /* undo the step */
                memcpy(n->w[2], sw2, nw2 * sizeof(smb_real));
                memcpy(n->b[1], sb1, nb1 * sizeof(smb_real));
                memcpy(n->b[2], sb2, nb2 * sizeof(smb_real));
                for (idx = 0; idx < nw1 && ok; idx++) {
                    double          ana = -(double)t->dw[1][idx];
                    double          ep, em, num;
                    smb_real        save = n->w[1][idx];
                    const smb_real *y;
                    n->w[1][idx] = save + (smb_real)eps;
                    y = net_forward(n, x);
                    ep = 0.5 * ((double)d[0] - y[0]) * ((double)d[0] - y[0]);
                    n->w[1][idx] = save - (smb_real)eps;
                    y = net_forward(n, x);
                    em = 0.5 * ((double)d[0] - y[0]) * ((double)d[0] - y[0]);
                    n->w[1][idx] = save;
                    num = (ep - em) / (2.0 * eps);
                    if (fabs(num - ana) > 5e-3 + 0.02 * fabs(ana))
                        ok = 0;
                }
                trainer_free(t);
                net_free(n);
            }
            CHECK(ok, "conv: backprop gradients match finite differences");
        }

        /* a conv net actually learns: total error falls sharply while fitting */
        {
            size_t dims[3]  = { 8, 0, 1 };
            int    kind[3]  = { LAYER_DENSE, LAYER_CONV, LAYER_DENSE };
            size_t nfilt[3] = { 0, 3, 0 };
            size_t ksize[3] = { 0, 3, 0 };
            Net   *n = net_build(dims, kind, nfilt, ksize, 3);
            int    ok = (n != NULL);
            if (ok) {
                smb_real X[6][8], D[6][1];
                smb_real e0 = 0, e1 = 0;
                Trainer *t;
                int      s, ep, i;
                rng_seed(&r, 8);
                net_init(n, &r);
                for (s = 0; s < 6; s++) {
                    for (i = 0; i < 8; i++)
                        X[s][i] = rng_uniform(&r, 0.0f, 1.0f);
                    D[s][0] = (smb_real)(s & 1);
                }
                for (s = 0; s < 6; s++) {
                    const smb_real *y = net_forward(n, X[s]);
                    smb_real e = D[s][0] - y[0];
                    e0 += (smb_real)0.5 * e * e;
                }
                t = trainer_new(n, 0.3f, 0.9f);
                for (ep = 0; ep < 3000; ep++)
                    for (s = 0; s < 6; s++)
                        (void)trainer_learn(t, X[s], D[s]);
                for (s = 0; s < 6; s++) {
                    const smb_real *y = net_forward(n, X[s]);
                    smb_real e = D[s][0] - y[0];
                    e1 += (smb_real)0.5 * e * e;
                }
                ok = (e1 < (smb_real)0.25 * e0);
                trainer_free(t);
                net_free(n);
            }
            CHECK(ok, "conv: a convolutional net trains (error falls)");
        }
    }

    /* ckpt: save then warm-start restores identical weights (round-trip); an
     * incompatible target inherits nothing, keeping its own init (no corruption) */
    {
        const char *path = "/tmp/smb_ckpt_ut.tmp";
        size_t dims[3] = {3, 5, 2}, dims2[3] = {3, 6, 2};
        Rng r;
        Net *a, *b, *c;
        int same = 1, untouched = 1;
        size_t l, i;

        rng_seed(&r, 7);
        a = net_new(dims, 3); net_init(a, &r);
        b = net_new(dims, 3); rng_seed(&r, 99); net_init(b, &r);   /* different init */
        c = net_new(dims2, 3); rng_seed(&r, 99); net_init(c, &r);  /* incompatible mid layer */
        if (a && b && c && ckpt_save(a, path) == 0) {
            Net cbefore = *c;                     /* shallow snapshot of pointers */
            smb_real w0 = c->w[1][0];
            ckpt_warmstart(b, path);
            ckpt_warmstart(c, path);
            for (l = 1; l < 3; l++) {
                size_t ws = net_layer_wsize(a, l);
                for (i = 0; i < ws; i++)
                    if (a->w[l][i] != b->w[l][i]) same = 0;
            }
            if (c->w[1][0] != w0) untouched = 0;  /* layer 1 (5 vs 6) must not change */
            (void)cbefore;
            CHECK(same, "ckpt: warm-start restores identical weights");
            CHECK(untouched, "ckpt: incompatible layer keeps its own init");
        } else {
            CHECK(0, "ckpt: setup");
        }
        net_free(a); net_free(b); net_free(c);
        remove(path);
    }

    /* conv2f: the 2D conv front-end's analytic weight gradient matches central
     * finite differences of a toy loss on its pooled features */
    {
        Rng r;
        Conv2f base;
        smb_real img[64], tgt[CONV2F_MAXF];
        int trial, ok = 1, i;
        rng_seed(&r, 31);
        conv2f_init(&base, 4, 3, 8, &r);            /* 4 filters, 3x3, 8x8 image */
        for (i = 0; i < 64; i++) img[i] = rng_uniform(&r, -1.0f, 1.0f);
        for (i = 0; i < 4; i++)  tgt[i] = rng_uniform(&r, 0.0f, 1.0f);
        for (trial = 0; trial < 30; trial++) {
            Conv2f a = base, num = base;
            smb_real dfeat[CONV2F_MAXF], eps = 1e-3f, ana, gnum, tol, before, w0, Lp, Lm;
            smb_real *wp_a, *wp_n;
            int f = (int)(rng_u32(&r) % 4u), bias = (int)(rng_u32(&r) & 1u);
            int kk = (int)(rng_u32(&r) % 9u), j;
            if (bias) { wp_a = &a.b[f]; wp_n = &num.b[f]; }
            else      { wp_a = &a.w[f][kk]; wp_n = &num.w[f][kk]; }
            before = *wp_a;
            conv2f_forward(&a, img);
            for (j = 0; j < 4; j++) dfeat[j] = a.pooled[j] - tgt[j];
            conv2f_backward(&a, img, dfeat, 1.0f, 0.0f);   /* lr=1, mom=0: delta = grad */
            ana = -(*wp_a - before);
            w0 = *wp_n;
            *wp_n = w0 + eps; { const smb_real *p = conv2f_forward(&num, img);
                Lp = 0; for (j = 0; j < 4; j++) { smb_real e = p[j]-tgt[j]; Lp += 0.5f*e*e; } }
            *wp_n = w0 - eps; { const smb_real *p = conv2f_forward(&num, img);
                Lm = 0; for (j = 0; j < 4; j++) { smb_real e = p[j]-tgt[j]; Lm += 0.5f*e*e; } }
            *wp_n = w0;
            gnum = (Lp - Lm) / (2.0f * eps);
            tol = 5e-3f + 2e-2f * (smb_real)fabs((double)ana);
            if ((smb_real)fabs((double)(ana - gnum)) > tol) ok = 0;
        }
        CHECK(ok, "conv2f: 2D conv gradient matches finite differences");
    }

    printf("\n%d checks, %d failed\n", t_run, t_fail);
    return t_fail ? 1 : 0;
}

#else

/* In the production (non-test) build every source in the tree is compiled,
 * including this one, and the whole body above is preprocessed away. ISO C
 * forbids an empty translation unit, so leave one benign declaration behind. */
typedef int smb_tests_translation_unit;

#endif /* UNIT_TEST */
