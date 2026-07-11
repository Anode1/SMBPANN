/* gentask.c -- generate a reproducible synthetic classification dataset.
 *
 * Writes a plain-text dataset (whitespace: `-d` input values then one 0/1 target)
 * to stdout, reproducibly from a seed (xorshift, so the same seed gives the same
 * data on any machine). The task: x uniform in [-1,1]^d; label 1 if
 * sin(freq * (w . x)) > 0 else 0, for a fixed random projection w, with `-e`%
 * label noise. Higher freq makes more decision stripes, so more network capacity
 * is needed to fit it -- a task where topology genuinely matters, unlike XOR, so
 * the GA-versus-random contest is non-trivial. Self-contained: no downloads.
 */
#ifndef UNIT_TEST

#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "common.h"
#include "rng.h"

#define GENTASK_MAX_DIM 1024

static void usage(const char *prog)
{
    fprintf(stderr,
        "usage: %s [-d dim] [-N samples] [-f freq] [-e noise%%] [-s seed]\n"
        "\n"
        "Writes a synthetic `dim`-input, 1-output classification task to stdout.\n"
        "  -d dim      input dimension            (default 4)\n"
        "  -N samples  number of rows             (default 600)\n"
        "  -f freq     decision-stripe frequency  (default 3)\n"
        "  -e noise    label-noise percent        (default 5)\n"
        "  -s seed     PRNG seed                  (default 1)\n", prog);
}

int main(int argc, char **argv)
{
    long   dim = 4, nsamples = 600, seed = 1;
    double freq = 3.0, noise = 5.0;
    int    c;
    long   s, j;
    double w[GENTASK_MAX_DIM];
    Rng    rng;

    while ((c = getopt(argc, argv, "d:N:f:e:s:h")) != -1) {
        switch (c) {
        case 'd': dim      = atol(optarg); break;
        case 'N': nsamples = atol(optarg); break;
        case 'f': freq     = atof(optarg); break;
        case 'e': noise    = atof(optarg); break;
        case 's': seed     = atol(optarg); break;
        case 'h': usage(argv[0]); return 0;
        default:  usage(argv[0]); return 2;
        }
    }
    if (dim < 1 || dim > GENTASK_MAX_DIM || nsamples < 1) {
        fprintf(stderr, "gentask: need 1 <= dim <= %d and samples >= 1\n",
                GENTASK_MAX_DIM);
        return 2;
    }

    rng_seed(&rng, (uint32_t)seed);
    /* Draw the fixed projection first, so the boundary depends only on the seed. */
    for (j = 0; j < dim; j++)
        w[j] = (double)rng_uniform(&rng, -1.0f, 1.0f);

    for (s = 0; s < nsamples; s++) {
        double z = 0.0;
        int    label;
        for (j = 0; j < dim; j++) {
            double xj = (double)rng_uniform(&rng, -1.0f, 1.0f);
            z += w[j] * xj;
            printf("%.5g ", xj);
        }
        label = (sin(freq * z) > 0.0) ? 1 : 0;
        if ((double)rng_uniform(&rng, 0.0f, 1.0f) < noise / 100.0)
            label = 1 - label;
        printf("%d\n", label);
    }
    return 0;
}

#endif /* !UNIT_TEST */
