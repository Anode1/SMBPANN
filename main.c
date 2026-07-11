/* main.c -- the SMBPANN command-line front end.
 *
 * The CLI is the only layer that turns a fatal condition into an exit or prints
 * to the user; the engine modules (net, train, ...) return codes and stay
 * silent (AIS STYLE.md). This milestone drives the built-in XOR demo -- the
 * thesis's canonical linearly-non-separable problem, unlearnable by a single
 * perceptron and the smallest proof that backprop through a hidden layer works.
 * Later milestones add plain-text datasets and the genetic architecture search.
 */
#ifndef UNIT_TEST

#define _POSIX_C_SOURCE 200809L  /* getopt, optarg (hidden by -std=c99 on glibc) */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "common.h"
#include "net.h"
#include "rng.h"
#include "train.h"

#ifndef SMB_VERSION
#define SMB_VERSION "0.0.0-dev"
#endif

static void usage(const char *prog)
{
    fprintf(stderr,
        "usage: %s [-e epochs] [-r rate] [-m momentum] [-s seed] [-H hidden]\n"
        "       %s -v | -h\n"
        "\n"
        "Trains a 2-H-1 network on XOR and prints the learned outputs.\n"
        "  -e epochs    training epochs           (default 20000)\n"
        "  -r rate      learning rate             (default 0.5)\n"
        "  -m momentum  momentum coefficient      (default 0.9)\n"
        "  -s seed      PRNG seed                 (default 1)\n"
        "  -H hidden    hidden-layer width        (default 4)\n"
        "  -v           print version and exit\n"
        "  -h           print this help and exit\n",
        prog, prog);
}

int main(int argc, char **argv)
{
    long   epochs = 20000, seed = 1, hidden = 4;
    double rate = 0.5, momentum = 0.9;
    int    c;

    /* XOR truth table: the four points of the unit square (thesis: no single
     * line separates them, so at least one hidden unit is required). */
    static const smb_real X[4][2] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    static const smb_real D[4][1] = { {0},    {1},    {1},    {0}    };

    size_t   dims[3];
    Net     *net;
    Trainer *t;
    Rng      rng;
    long     ep;
    int      i;

    while ((c = getopt(argc, argv, "e:r:m:s:H:vh")) != -1) {
        switch (c) {
        case 'e': epochs   = atol(optarg); break;
        case 'r': rate     = atof(optarg); break;
        case 'm': momentum = atof(optarg); break;
        case 's': seed     = atol(optarg); break;
        case 'H': hidden   = atol(optarg); break;
        case 'v': printf("smbpann %s\n", SMB_VERSION); return 0;
        case 'h': usage(argv[0]); return 0;
        default:  usage(argv[0]); return 2;
        }
    }
    if (epochs <= 0 || hidden <= 0) {
        fprintf(stderr, "smbpann: epochs and hidden must be positive\n");
        return 2;
    }

    dims[0] = 2;
    dims[1] = (size_t)hidden;
    dims[2] = 1;
    net = net_new(dims, 3);
    if (net == NULL) {
        fprintf(stderr, "smbpann: cannot build network\n");
        return 1;
    }
    rng_seed(&rng, (uint32_t)seed);
    net_init(net, &rng);

    t = trainer_new(net, (smb_real)rate, (smb_real)momentum);
    if (t == NULL) {
        net_free(net);
        fprintf(stderr, "smbpann: cannot build trainer\n");
        return 1;
    }

    printf("XOR  topology 2-%ld-1  weights %zu  rate %g  momentum %g  seed %ld\n",
           hidden, net_nweights(net), rate, momentum, seed);
    for (ep = 0; ep < epochs; ep++) {
        smb_real E = 0;
        for (i = 0; i < 4; i++)
            E += trainer_learn(t, X[i], D[i]);
        if (ep % (epochs / 10 > 0 ? epochs / 10 : 1) == 0)
            printf("  epoch %6ld   sse %.6f\n", ep, (double)E);
    }

    printf("learned:\n");
    for (i = 0; i < 4; i++) {
        const smb_real *y = net_forward(net, X[i]);
        printf("  %g XOR %g  ->  %.4f  (target %g)\n",
               (double)X[i][0], (double)X[i][1], (double)y[0], (double)D[i][0]);
    }

    trainer_free(t);
    net_free(net);
    return 0;
}

#endif /* !UNIT_TEST */
