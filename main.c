/* main.c -- the SMBPANN command-line front end and evaluation worker.
 *
 * The CLI is the only layer that turns a fatal condition into an exit or prints
 * to the user; the engine modules (net, train, ...) return codes and stay
 * silent (AIS STYLE.md).
 *
 * As a worker it does one thing: build a network of a given topology, train it
 * (on a plain-text dataset, or on the built-in XOR problem), and emit a single
 * machine-readable RESULT line carrying a fitness value (lower is better). The
 * shell coordinator (scripts/) launches one such worker process per candidate
 * and ranks them by that value. With no dataset it trains XOR -- the thesis's
 * canonical linearly-non-separable problem, the smallest proof that backprop
 * through a hidden layer works.
 */
#ifndef UNIT_TEST

#define _POSIX_C_SOURCE 200809L  /* getopt, optarg (hidden by -std=c99 on glibc) */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "common.h"
#include "data.h"
#include "genome.h"
#include "net.h"
#include "rng.h"
#include "train.h"

#ifndef SMB_VERSION
#define SMB_VERSION "0.0.0-dev"
#endif

static void usage(const char *prog)
{
    fprintf(stderr,
        "usage: %s [-t topology] [-f file -i in -o out [-p frac]]\n"
        "          [-e epochs] [-r rate] [-m momentum] [-s seed] [-H hidden] [-q]\n"
        "       %s -v | -h\n"
        "\n"
        "Trains a network and prints a RESULT line with its fitness (lower is\n"
        "better). With no -f it trains the built-in XOR problem.\n"
        "  -t topology  comma-separated layer widths, e.g. 2,4,1\n"
        "  -g spec      genome: topology|lrate|momentum|activation\n"
        "                 (e.g. 2,4,1|0.5|0.9|tanh); overrides -t/-r/-m\n"
        "  -f file      dataset: each line is `in` inputs then `out` targets\n"
        "  -i in        number of input values per sample   (with -f)\n"
        "  -o out       number of target values per sample  (with -f)\n"
        "  -p frac      train fraction of the split         (default 0.8)\n"
        "  -e epochs    training epochs                     (default 20000)\n"
        "  -r rate      learning rate                       (default 0.5)\n"
        "  -m momentum  momentum coefficient                (default 0.9)\n"
        "  -s seed      PRNG seed                           (default 1)\n"
        "  -H hidden    hidden width for the XOR default    (default 4)\n"
        "  -q           quiet: print only the RESULT line\n"
        "  -v           print version and exit\n"
        "  -h           print this help and exit\n",
        prog, prog);
}

/* Parse "2,4,1" into DIMS (capacity MAX); set *NOUT to the layer count.
 * Returns 0, or -1 on a malformed list, a non-positive width, or overflow. */
static int parse_topology(const char *s, size_t *dims, size_t max, size_t *nout)
{
    size_t      n = 0;
    const char *p = s;

    while (*p != '\0') {
        char *end;
        long  v = strtol(p, &end, 10);
        if (end == p || v <= 0)
            return -1;
        if (n >= max)
            return -1;
        dims[n++] = (size_t)v;
        p = end;
        while (*p == ',' || *p == ' ')
            p++;
    }
    if (n < 2)
        return -1;
    *nout = n;
    return 0;
}

/* Format DIMS as "2,4,1" into BUF. */
static void format_topology(const size_t *dims, size_t n, char *buf, size_t bufsz)
{
    size_t i, off = 0;

    for (i = 0; i < n && off < bufsz; i++) {
        int w = snprintf(buf + off, bufsz - off, "%s%zu",
                         (i > 0) ? "," : "", dims[i]);
        if (w < 0 || (size_t)w >= bufsz - off)
            break;
        off += (size_t)w;
    }
}

/* Mean squared error over the train (USE_TEST=0) or test (USE_TEST=1) half. */
static double eval_mse(Net *net, const Dataset *ds, const Split *sp, int use_test)
{
    size_t count = use_test ? split_test_count(sp) : split_train_count(sp);
    size_t p, k;
    double sum = 0.0;

    if (count == 0)
        return 0.0;
    for (p = 0; p < count; p++) {
        size_t          idx = use_test ? split_test_index(sp, p)
                                       : split_train_index(sp, p);
        const smb_real *x = dataset_input(ds, idx);
        const smb_real *d = dataset_target(ds, idx);
        const smb_real *y = net_forward(net, x);
        for (k = 0; k < ds->noutput; k++) {
            double e = (double)d[k] - (double)y[k];
            sum += e * e;
        }
    }
    return sum / (double)(count * ds->noutput);
}

/* Load FILE, split it, train NET/T on the train half, and report train and
 * test MSE. Returns 0, or -1 on load/split failure. */
static int run_dataset(Net *net, Trainer *t, const char *file,
                       size_t ni, size_t no, double frac, long epochs,
                       uint32_t seed, double *train_err, double *test_err)
{
    Dataset ds;
    Split   sp;
    Rng     r;
    long    ep;
    size_t  p;

    if (dataset_load(&ds, file, ni, no) != 0)
        return -1;
    rng_seed(&r, seed);
    if (split_make(&sp, &ds, (smb_real)frac, &r) != 0) {
        dataset_free(&ds);
        return -1;
    }
    for (ep = 0; ep < epochs; ep++)
        for (p = 0; p < split_train_count(&sp); p++) {
            size_t idx = split_train_index(&sp, p);
            (void)trainer_learn(t, dataset_input(&ds, idx),
                                dataset_target(&ds, idx));
        }
    *train_err = eval_mse(net, &ds, &sp, 0);
    *test_err  = eval_mse(net, &ds, &sp, 1);
    split_free(&sp);
    dataset_free(&ds);
    return 0;
}

/* Train the built-in XOR problem; report the final sum-squared error. */
static double run_xor(Trainer *t, long epochs)
{
    static const smb_real X[4][2] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    static const smb_real D[4][1] = { {0},    {1},    {1},    {0}    };
    smb_real E = 0;
    long     ep;
    int      i;

    for (ep = 0; ep < epochs; ep++) {
        E = 0;
        for (i = 0; i < 4; i++)
            E += trainer_learn(t, X[i], D[i]);
    }
    return (double)E;
}

int main(int argc, char **argv)
{
    long   epochs = 20000, seed = 1, hidden = 4;
    long   ninput = 0, noutput = 0;
    double rate = 0.5, momentum = 0.9, frac = 0.8;
    char  *topo_arg = NULL, *file = NULL, *spec_arg = NULL;
    int    activation = ACT_SIGMOID;
    int    quiet = 0, c;

    size_t   dims[SMB_MAX_LAYERS];
    size_t   nlayers = 0;
    char     topo[256];
    Genome   gen;
    int      have_spec = 0;
    Net     *net = NULL;
    Trainer *t = NULL;
    Rng      rng;
    double   train_err = 0.0, test_err = 0.0, fitness = 0.0;
    int      rc = 0;

    while ((c = getopt(argc, argv, "g:t:f:i:o:p:e:r:m:s:H:qvh")) != -1) {
        switch (c) {
        case 'g': spec_arg = optarg; break;
        case 't': topo_arg = optarg; break;
        case 'f': file     = optarg; break;
        case 'i': ninput   = atol(optarg); break;
        case 'o': noutput  = atol(optarg); break;
        case 'p': frac     = atof(optarg); break;
        case 'e': epochs   = atol(optarg); break;
        case 'r': rate     = atof(optarg); break;
        case 'm': momentum = atof(optarg); break;
        case 's': seed     = atol(optarg); break;
        case 'H': hidden   = atol(optarg); break;
        case 'q': quiet    = 1; break;
        case 'v': printf("smbpann %s\n", SMB_VERSION); return 0;
        case 'h': usage(argv[0]); return 0;
        default:  usage(argv[0]); return 2;
        }
    }
    if (epochs <= 0 || hidden <= 0) {
        fprintf(stderr, "smbpann: epochs and hidden must be positive\n");
        return 2;
    }

    /* A full candidate spec (-g) carries the topology AND the co-evolved
     * hyper-parameters, and overrides -t/-r/-m. Else -t gives the topology, else
     * the XOR default 2-hidden-1. */
    if (spec_arg != NULL) {
        size_t i;
        if (genome_parse(&gen, spec_arg) != 0) {
            fprintf(stderr, "smbpann: bad spec '%s'\n", spec_arg);
            return 2;
        }
        have_spec = 1;
        nlayers = gen.n;
        for (i = 0; i < nlayers; i++)
            dims[i] = gen.dim[i];
        rate       = (double)gen.lrate;
        momentum   = (double)gen.momentum;
        activation = gen.activation;
    } else if (topo_arg != NULL) {
        if (parse_topology(topo_arg, dims, SMB_MAX_LAYERS, &nlayers) != 0) {
            fprintf(stderr, "smbpann: bad topology '%s'\n", topo_arg);
            return 2;
        }
    } else {
        dims[0] = 2;
        dims[1] = (size_t)hidden;
        dims[2] = 1;
        nlayers = 3;
    }

    if (file != NULL) {
        if (ninput <= 0 || noutput <= 0) {
            fprintf(stderr, "smbpann: -f requires -i and -o\n");
            return 2;
        }
        if (dims[0] != (size_t)ninput || dims[nlayers - 1] != (size_t)noutput) {
            fprintf(stderr, "smbpann: topology ends (%zu,%zu) must match data "
                            "(%ld,%ld)\n",
                    dims[0], dims[nlayers - 1], ninput, noutput);
            return 2;
        }
    } else if (dims[0] != 2 || dims[nlayers - 1] != 1) {
        fprintf(stderr, "smbpann: the XOR default needs a 2-...-1 topology\n");
        return 2;
    }

    /* A spec may include conv layers, so build via net_build (which honours the
     * per-layer kinds); a bare -t/XOR run is all dense. */
    net = have_spec
        ? net_build(gen.dim, gen.kind, gen.nfilt, gen.ksize, gen.n)
        : net_new(dims, nlayers);
    if (net == NULL) {
        fprintf(stderr, "smbpann: cannot build network\n");
        return 1;
    }
    net->activation = activation;
    rng_seed(&rng, (uint32_t)seed);
    net_init(net, &rng);

    t = trainer_new(net, (smb_real)rate, (smb_real)momentum);
    if (t == NULL) {
        fprintf(stderr, "smbpann: cannot build trainer\n");
        rc = 1;
        goto cleanup;
    }

    if (file != NULL) {
        if (run_dataset(net, t, file, (size_t)ninput, (size_t)noutput, frac,
                        epochs, (uint32_t)seed, &train_err, &test_err) != 0) {
            fprintf(stderr, "smbpann: cannot load or split '%s'\n", file);
            rc = 1;
            goto cleanup;
        }
        fitness = test_err;
    } else {
        train_err = test_err = fitness = run_xor(t, epochs);
        if (!quiet) {
            static const smb_real X[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };
            int i;
            printf("learned:\n");
            for (i = 0; i < 4; i++) {
                const smb_real *y = net_forward(net, X[i]);
                printf("  %g XOR %g  ->  %.4f\n",
                       (double)X[i][0], (double)X[i][1], (double)y[0]);
            }
        }
    }

    format_topology(dims, nlayers, topo, sizeof topo);
    printf("RESULT topology=%s spec=%s weights=%zu seed=%ld epochs=%ld "
           "train=%.6g test=%.6g fitness=%.6g\n",
           topo, (spec_arg != NULL) ? spec_arg : topo,
           net_nweights(net), seed, epochs, train_err, test_err, fitness);

cleanup:
    trainer_free(t);
    net_free(net);
    return rc;
}

#endif /* !UNIT_TEST */
