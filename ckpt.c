/* ckpt.c -- network checkpoints. See ckpt.h. */
#include <stdio.h>

#include "ckpt.h"

int ckpt_save(const Net *net, const char *path)
{
    FILE  *f;
    size_t l, i;

    f = fopen(path, "w");
    if (f == NULL)
        return -1;
    fprintf(f, "SMBCKPT %zu %d\n", net->nlayers, net->activation);
    for (l = 1; l < net->nlayers; l++) {
        size_t ws = net_layer_wsize(net, l);
        size_t bs = net_layer_bsize(net, l);
        fprintf(f, "%d %zu %zu %zu %zu %zu %zu\n",
                net->kind[l], net->dim[l], net->dim[l - 1],
                net->nfilt[l], net->ksize[l], ws, bs);
        for (i = 0; i < ws; i++) fprintf(f, "%.9g\n", (double)net->w[l][i]);
        for (i = 0; i < bs; i++) fprintf(f, "%.9g\n", (double)net->b[l][i]);
    }
    if (fclose(f) != 0)
        return -1;
    return 0;
}

int ckpt_warmstart(Net *net, const char *path)
{
    FILE  *f;
    size_t cn, l;
    int    cact, compatible;

    f = fopen(path, "r");
    if (f == NULL)
        return -1;

    /* Pass 1 -- validate that the checkpoint's topology matches NET *exactly*.
     * Inheriting a weight only means something when the layer it lands in is the
     * same layer it was trained in. Any structural change -- a mutated width, an
     * inserted / pruned / flipped layer, a crossover splice -- shifts layers so
     * that a positional copy carries weights from an unrelated layer (and two
     * different shapes can share a buffer size, so a size check alone lets that
     * through). Rather than copy semantically meaningless weights, we refuse the
     * whole checkpoint and let the child train from its fresh init. Only an
     * unchanged architecture (a surviving elite) inherits, and it inherits whole. */
    if (fscanf(f, "SMBCKPT %zu %d", &cn, &cact) != 2) {
        fclose(f);
        return -1;
    }
    compatible = (cn == net->nlayers);
    for (l = 1; l < cn; l++) {
        int    ckind;
        size_t cdim, cprev, cnf, cks, cws, cbs, i;
        double v;

        if (fscanf(f, "%d %zu %zu %zu %zu %zu %zu",
                   &ckind, &cdim, &cprev, &cnf, &cks, &cws, &cbs) != 7) {
            fclose(f);
            return -1;
        }
        if (!(l < net->nlayers && net->kind[l] == ckind
              && net->dim[l] == cdim && net->dim[l - 1] == cprev
              && net->nfilt[l] == cnf && net->ksize[l] == cks
              && net_layer_wsize(net, l) == cws
              && net_layer_bsize(net, l) == cbs))
            compatible = 0;
        for (i = 0; i < cws + cbs; i++)             /* consume, decide later */
            if (fscanf(f, "%lf", &v) != 1) { fclose(f); return -1; }
    }
    if (!compatible) {                              /* topology differs: inherit nothing */
        fclose(f);
        return 0;
    }

    /* Pass 2 -- topologies match layer for layer; copy every weight and bias. */
    if (fseek(f, 0L, SEEK_SET) != 0
        || fscanf(f, "SMBCKPT %zu %d", &cn, &cact) != 2) {
        fclose(f);
        return -1;
    }
    for (l = 1; l < cn; l++) {
        int    ckind;
        size_t cdim, cprev, cnf, cks, cws, cbs, i;
        double v;

        if (fscanf(f, "%d %zu %zu %zu %zu %zu %zu",
                   &ckind, &cdim, &cprev, &cnf, &cks, &cws, &cbs) != 7) {
            fclose(f);
            return -1;
        }
        for (i = 0; i < cws; i++) {
            if (fscanf(f, "%lf", &v) != 1) { fclose(f); return -1; }
            net->w[l][i] = (smb_real)v;
        }
        for (i = 0; i < cbs; i++) {
            if (fscanf(f, "%lf", &v) != 1) { fclose(f); return -1; }
            net->b[l][i] = (smb_real)v;
        }
    }
    fclose(f);
    return 0;
}
