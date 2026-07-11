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

#include <stdio.h>

#include "act.h"
#include "arena.h"
#include "common.h"
#include "data.h"
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

    printf("\n%d checks, %d failed\n", t_run, t_fail);
    return t_fail ? 1 : 0;
}

#else

/* In the production (non-test) build every source in the tree is compiled,
 * including this one, and the whole body above is preprocessed away. ISO C
 * forbids an empty translation unit, so leave one benign declaration behind. */
typedef int smb_tests_translation_unit;

#endif /* UNIT_TEST */
