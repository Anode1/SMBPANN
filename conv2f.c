/* conv2f.c -- a 2D convolutional front-end. See conv2f.h. */
#include <math.h>

#include "conv2f.h"

int conv2f_init(Conv2f *c, size_t F, size_t K, size_t S, Rng *rng)
{
    size_t f, k;
    smb_real lim;

    if (F < 1 || F > CONV2F_MAXF || K < 1 || K > CONV2F_MAXK || K > S)
        return -1;
    c->F = F; c->K = K; c->S = S; c->P = S - K + 1;
    lim = (smb_real)2.4 / (smb_real)(K * K);        /* thesis init, fan-in = K*K */
    for (f = 0; f < F; f++) {
        for (k = 0; k < K * K; k++) {
            c->w[f][k]  = rng_uniform(rng, -lim, lim);
            c->dw[f][k] = 0;
        }
        c->b[f]  = 0;
        c->db[f] = 0;
    }
    return 0;
}

const smb_real *conv2f_forward(Conv2f *c, const smb_real *img)
{
    size_t f, oy, ox, ky, kx;

    for (f = 0; f < c->F; f++) {
        smb_real best = (smb_real)-1e30;
        size_t   barg = 0;
        for (oy = 0; oy < c->P; oy++) {
            for (ox = 0; ox < c->P; ox++) {
                smb_real s = c->b[f];
                for (ky = 0; ky < c->K; ky++)
                    for (kx = 0; kx < c->K; kx++)
                        s += c->w[f][ky * c->K + kx]
                             * img[(oy + ky) * c->S + (ox + kx)];
                s = (smb_real)tanh((double)s);
                if (s > best) { best = s; barg = oy * c->P + ox; }
            }
        }
        c->pooled[f] = best;
        c->arg[f]    = barg;
    }
    return c->pooled;
}

void conv2f_backward(Conv2f *c, const smb_real *img, const smb_real *dfeat,
                     smb_real lr, smb_real mom)
{
    size_t f, ky, kx;

    for (f = 0; f < c->F; f++) {
        smb_real dpre = dfeat[f] * ((smb_real)1 - c->pooled[f] * c->pooled[f]); /* tanh' */
        size_t   ay = c->arg[f] / c->P, ax = c->arg[f] % c->P;
        for (ky = 0; ky < c->K; ky++) {
            for (kx = 0; kx < c->K; kx++) {
                smb_real g = dpre * img[(ay + ky) * c->S + (ax + kx)];
                size_t   idx = ky * c->K + kx;
                c->dw[f][idx] = lr * g + mom * c->dw[f][idx];
                c->w[f][idx] -= c->dw[f][idx];
            }
        }
        c->db[f] = lr * dpre + mom * c->db[f];
        c->b[f] -= c->db[f];
    }
}
