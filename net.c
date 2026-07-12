/* net.c -- feed-forward network and the forward pass. See net.h.
 *
 * All allocation is here in net_new (once, at construction) and released in
 * net_free. net_forward and net_init touch no allocator. */
#include <stdlib.h>
#include <string.h>

#include "act.h"
#include "net.h"

Net *net_new(const size_t *dims, size_t nlayers)
{
    Net   *net;
    size_t l;

    if (dims == NULL || nlayers < 2 || nlayers > SMB_MAX_LAYERS)
        return NULL;

    net = calloc(1, sizeof *net);   /* zero: every pointer starts NULL */
    if (net == NULL)
        return NULL;
    net->nlayers = nlayers;

    for (l = 0; l < nlayers; l++) {
        if (dims[l] == 0)
            goto fail;
        net->dim[l] = dims[l];
    }

    /* Input layer holds only its activation vector (the copied input). */
    net->a[0] = calloc(net->dim[0], sizeof *net->a[0]);
    if (net->a[0] == NULL)
        goto fail;

    /* Each subsequent layer: weight matrix, biases, activations, pre-acts. */
    for (l = 1; l < nlayers; l++) {
        net->w[l] = malloc(net->dim[l] * net->dim[l - 1] * sizeof *net->w[l]);
        net->b[l] = calloc(net->dim[l], sizeof *net->b[l]);
        net->a[l] = calloc(net->dim[l], sizeof *net->a[l]);
        net->z[l] = calloc(net->dim[l], sizeof *net->z[l]);
        if (net->w[l] == NULL || net->b[l] == NULL ||
            net->a[l] == NULL || net->z[l] == NULL)
            goto fail;
    }
    return net;

fail:
    net_free(net);   /* frees the partial network and the struct */
    return NULL;
}

void net_init(Net *net, Rng *rng)
{
    size_t l, i, nw;

    for (l = 1; l < net->nlayers; l++) {
        /* Thesis initialization: uniform in [-2.4/fan_in, +2.4/fan_in], where
         * fan_in is the number of inputs to a unit (the sending layer width). */
        smb_real range = (smb_real)(2.4 / (double)net->dim[l - 1]);
        nw = net->dim[l] * net->dim[l - 1];
        for (i = 0; i < nw; i++)
            net->w[l][i] = rng_uniform(rng, -range, range);
        for (i = 0; i < net->dim[l]; i++)
            net->b[l][i] = rng_uniform(rng, -range, range);
    }
}

const smb_real *net_forward(Net *net, const smb_real *x)
{
    size_t l, i, j;

    memcpy(net->a[0], x, net->dim[0] * sizeof *net->a[0]);

    for (l = 1; l < net->nlayers; l++) {
        const smb_real *prev = net->a[l - 1];
        size_t          nprev = net->dim[l - 1];
        for (i = 0; i < net->dim[l]; i++) {
            /* z[l][i] = b[l][i] + sum_j w[l][i][j] * a[l-1][j] */
            const smb_real *wi = net->w[l] + i * nprev;
            smb_real        s = net->b[l][i];
            for (j = 0; j < nprev; j++)
                s += wi[j] * prev[j];
            net->z[l][i] = s;
            /* hidden layers use the net's chosen activation; the output layer
             * stays sigmoid so classification outputs remain in (0,1) */
            net->a[l][i] = (l == net->nlayers - 1)
                         ? act_sigmoid(s)
                         : act_apply(net->activation, s);
        }
    }
    return net->a[net->nlayers - 1];
}

size_t net_nweights(const Net *net)
{
    size_t l, n = 0;

    for (l = 1; l < net->nlayers; l++)
        n += net->dim[l] * net->dim[l - 1];
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
