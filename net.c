/* net.c -- feed-forward network and the forward pass. See net.h.
 *
 * All allocation is here in net_build (once, at construction) and released in
 * net_free. net_forward and net_init touch no allocator. Layers are dense or
 * 1D-convolutional (weight-shared, local); the forward pass branches per layer. */
#include <stdlib.h>
#include <string.h>

#include "act.h"
#include "net.h"

size_t net_conv_positions(const Net *net, size_t l)
{
    size_t prev = net->dim[l - 1];
    return (prev >= net->ksize[l]) ? (prev - net->ksize[l] + 1) : 0;
}

size_t net_layer_wsize(const Net *net, size_t l)
{
    return (net->kind[l] == LAYER_CONV)
         ? net->nfilt[l] * net->ksize[l]           /* shared: F filters x K */
         : net->dim[l] * net->dim[l - 1];
}

size_t net_layer_bsize(const Net *net, size_t l)
{
    return (net->kind[l] == LAYER_CONV) ? net->nfilt[l] : net->dim[l];
}

Net *net_build(const size_t *dims, const int *kind,
               const size_t *nfilt, const size_t *ksize, size_t nlayers)
{
    Net   *net;
    size_t l;

    if (dims == NULL || nlayers < 2 || nlayers > SMB_MAX_LAYERS)
        return NULL;

    net = calloc(1, sizeof *net);   /* zero: pointers NULL, kinds DENSE */
    if (net == NULL)
        return NULL;
    net->nlayers = nlayers;

    if (dims[0] == 0)
        goto fail;
    net->dim[0] = dims[0];

    /* Resolve each layer's kind and width (conv widths are computed). The output
     * layer is always dense so its width equals the number of targets. */
    for (l = 1; l < nlayers; l++) {
        int k = (kind != NULL && l != nlayers - 1) ? kind[l] : LAYER_DENSE;
        if (k == LAYER_CONV) {
            size_t F = (nfilt != NULL) ? nfilt[l] : 0;
            size_t K = (ksize != NULL) ? ksize[l] : 0;
            if (F == 0 || K == 0 || K > net->dim[l - 1])
                goto fail;            /* kernel must fit the previous layer */
            net->kind[l]  = LAYER_CONV;
            net->nfilt[l] = F;
            net->ksize[l] = K;
            net->dim[l]   = F * (net->dim[l - 1] - K + 1);
        } else {
            if (dims[l] == 0)
                goto fail;
            net->kind[l] = LAYER_DENSE;
            net->dim[l]  = dims[l];
        }
    }

    net->a[0] = calloc(net->dim[0], sizeof *net->a[0]);
    if (net->a[0] == NULL)
        goto fail;
    for (l = 1; l < nlayers; l++) {
        net->w[l] = malloc(net_layer_wsize(net, l) * sizeof *net->w[l]);
        net->b[l] = calloc(net_layer_bsize(net, l), sizeof *net->b[l]);
        net->a[l] = calloc(net->dim[l], sizeof *net->a[l]);
        net->z[l] = calloc(net->dim[l], sizeof *net->z[l]);
        if (net->w[l] == NULL || net->b[l] == NULL ||
            net->a[l] == NULL || net->z[l] == NULL)
            goto fail;
    }
    return net;

fail:
    net_free(net);
    return NULL;
}

Net *net_new(const size_t *dims, size_t nlayers)
{
    return net_build(dims, NULL, NULL, NULL, nlayers);
}

void net_init(Net *net, Rng *rng)
{
    size_t l, i;

    for (l = 1; l < net->nlayers; l++) {
        /* Thesis init [-2.4/fan_in, +2.4/fan_in]; a conv unit's fan-in is the
         * kernel size, a dense unit's is the previous layer width. */
        size_t   fanin = (net->kind[l] == LAYER_CONV)
                       ? net->ksize[l] : net->dim[l - 1];
        smb_real range = (smb_real)(2.4 / (double)fanin);
        size_t   nw = net_layer_wsize(net, l);
        size_t   nb = net_layer_bsize(net, l);
        for (i = 0; i < nw; i++)
            net->w[l][i] = rng_uniform(rng, -range, range);
        for (i = 0; i < nb; i++)
            net->b[l][i] = rng_uniform(rng, -range, range);
    }
}

const smb_real *net_forward(Net *net, const smb_real *x)
{
    size_t l;
    size_t L = net->nlayers - 1;

    memcpy(net->a[0], x, net->dim[0] * sizeof *net->a[0]);

    for (l = 1; l < net->nlayers; l++) {
        const smb_real *prev = net->a[l - 1];

        if (net->kind[l] == LAYER_CONV) {
            size_t K = net->ksize[l], F = net->nfilt[l];
            size_t opos = net->dim[l - 1] - K + 1;
            size_t f, p, k;
            for (f = 0; f < F; f++) {
                const smb_real *wf = net->w[l] + f * K;   /* shared kernel */
                smb_real        bf = net->b[l][f];
                for (p = 0; p < opos; p++) {
                    smb_real s = bf;
                    size_t   idx = f * opos + p;
                    for (k = 0; k < K; k++)
                        s += wf[k] * prev[p + k];
                    net->z[l][idx] = s;
                    net->a[l][idx] = act_apply(net->activation, s);
                }
            }
        } else {
            size_t nprev = net->dim[l - 1];
            size_t i, j;
            for (i = 0; i < net->dim[l]; i++) {
                const smb_real *wi = net->w[l] + i * nprev;
                smb_real        s = net->b[l][i];
                for (j = 0; j < nprev; j++)
                    s += wi[j] * prev[j];
                net->z[l][i] = s;
                /* the output layer stays sigmoid so outputs remain in (0,1) */
                net->a[l][i] = (l == L) ? act_sigmoid(s)
                                        : act_apply(net->activation, s);
            }
        }
    }
    return net->a[L];
}

size_t net_nweights(const Net *net)
{
    size_t l, n = 0;

    for (l = 1; l < net->nlayers; l++)
        n += net_layer_wsize(net, l);
    return n;
}

void net_free(Net *net)
{
    size_t l;

    if (net == NULL)
        return;
    /* Loop the full fixed table: unused slots are NULL (calloc'd), free(NULL)
     * is a no-op, so this is safe on a partially-constructed net too. */
    for (l = 0; l < SMB_MAX_LAYERS; l++) {
        free(net->w[l]);
        free(net->b[l]);
        free(net->a[l]);
        free(net->z[l]);
    }
    free(net);
}
