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
        "  -s seed     PRNG seed                  (default 1)\n"
        "  -I side     2D image-motif task instead: a side*side image with a 3x3\n"
        "              diagonal motif (\\ vs /) at a random position; label is its\n"
        "              orientation. Translation-invariant, so convolution generalizes\n"
        "              and a dense net of similar size does not. Emits side*side\n"
        "              inputs then the label. (0 = the default sine task)\n", prog);
}

int main(int argc, char **argv)
{
    long   dim = 4, nsamples = 600, seed = 1, img_side = 0;
    double freq = 3.0, noise = 5.0;
    int    c;
    long   s, j;
    double w[GENTASK_MAX_DIM];
    Rng    rng;

    while ((c = getopt(argc, argv, "d:N:f:e:s:I:h")) != -1) {
        switch (c) {
        case 'd': dim      = atol(optarg); break;
        case 'N': nsamples = atol(optarg); break;
        case 'f': freq     = atof(optarg); break;
        case 'e': noise    = atof(optarg); break;
        case 's': seed     = atol(optarg); break;
        case 'I': img_side = atol(optarg); break;
        case 'h': usage(argv[0]); return 0;
        default:  usage(argv[0]); return 2;
        }
    }
    if (nsamples < 1) {
        fprintf(stderr, "gentask: need samples >= 1\n");
        return 2;
    }

    /* 2D image-motif task: a 3x3 diagonal at a random position (translation
     * invariant, so convolution is decisive), side*side inputs then the label. */
    if (img_side > 0) {
        long side = img_side, npix = side * side, i, k, r0, c0;
        if (side < 3 || npix > GENTASK_MAX_DIM) {
            fprintf(stderr, "gentask: need 3 <= side and side*side <= %d\n",
                    GENTASK_MAX_DIM);
            return 2;
        }
        rng_seed(&rng, (uint32_t)seed);
        for (s = 0; s < nsamples; s++) {
            int    cls = (int)(rng_u32(&rng) & 1u);
            double img[GENTASK_MAX_DIM];
            for (k = 0; k < npix; k++)
                img[k] = 0.15 * (double)rng_gaussian(&rng);   /* noise background */
            r0 = (long)(rng_u32(&rng) % (uint32_t)(side - 2));
            c0 = (long)(rng_u32(&rng) % (uint32_t)(side - 2));
            for (i = 0; i < 3; i++) {                          /* draw the motif */
                long rr = r0 + i, cc = cls ? (c0 + 2 - i) : (c0 + i);
                img[rr * side + cc] += 1.5;
            }
            for (k = 0; k < npix; k++)
                printf("%.5g ", img[k]);
            printf("%d\n", cls);
        }
        return 0;
    }

    if (dim < 1 || dim > GENTASK_MAX_DIM) {
        fprintf(stderr, "gentask: need 1 <= dim <= %d\n", GENTASK_MAX_DIM);
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
