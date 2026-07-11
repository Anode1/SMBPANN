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

#include "common.h"
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

    printf("\n%d checks, %d failed\n", t_run, t_fail);
    return t_fail ? 1 : 0;
}

#else

/* In the production (non-test) build every source in the tree is compiled,
 * including this one, and the whole body above is preprocessed away. ISO C
 * forbids an empty translation unit, so leave one benign declaration behind. */
typedef int smb_tests_translation_unit;

#endif /* UNIT_TEST */
