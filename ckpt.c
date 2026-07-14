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
    int    cact;

    f = fopen(path, "r");
    if (f == NULL)
        return -1;
    if (fscanf(f, "SMBCKPT %zu %d", &cn, &cact) != 2) {
        fclose(f);
        return -1;
    }
    for (l = 1; l < cn; l++) {
        int    ckind, compat;
        size_t cdim, cprev, cnf, cks, cws, cbs, i;
        double v;

        if (fscanf(f, "%d %zu %zu %zu %zu %zu %zu",
                   &ckind, &cdim, &cprev, &cnf, &cks, &cws, &cbs) != 7) {
            fclose(f);
            return -1;
        }
        /* copy into net's layer l only if it aligns and the buffers match */
        compat = (l < net->nlayers) && (net->kind[l] == ckind)
                 && (net_layer_wsize(net, l) == cws)
                 && (net_layer_bsize(net, l) == cbs);
        for (i = 0; i < cws; i++) {
            if (fscanf(f, "%lf", &v) != 1) { fclose(f); return -1; }
            if (compat) net->w[l][i] = (smb_real)v;
        }
        for (i = 0; i < cbs; i++) {
            if (fscanf(f, "%lf", &v) != 1) { fclose(f); return -1; }
            if (compat) net->b[l][i] = (smb_real)v;
        }
    }
    fclose(f);
    return 0;
}
