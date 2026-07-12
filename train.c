/* train.c -- backpropagation with momentum. See train.h.
 *
 * All allocation is in trainer_new; trainer_learn is pure arithmetic over the
 * pre-allocated scratch. */
#include <stdlib.h>

#include "act.h"
#include "train.h"

Trainer *trainer_new(Net *net, smb_real rate, smb_real momentum)
{
    Trainer *t;
    size_t   l;

    if (net == NULL)
        return NULL;

    t = calloc(1, sizeof *t);   /* zero: pointers NULL, deltas start at 0 */
    if (t == NULL)
        return NULL;
    t->net = net;
    t->rate = rate;
    t->momentum = momentum;

    /* One error-signal vector per non-input layer, and previous-delta buffers
     * mirroring the weights and biases (zeroed: the first step has no history). */
    for (l = 1; l < net->nlayers; l++) {
        t->beta[l] = calloc(net->dim[l], sizeof *t->beta[l]);
        t->dw[l]   = calloc(net_layer_wsize(net, l), sizeof *t->dw[l]);
        t->db[l]   = calloc(net_layer_bsize(net, l), sizeof *t->db[l]);
        if (t->beta[l] == NULL || t->dw[l] == NULL || t->db[l] == NULL)
            goto fail;
    }
    return t;

fail:
    trainer_free(t);
    return NULL;
}

smb_real trainer_learn(Trainer *t, const smb_real *x, const smb_real *d)
{
    Net   *net = t->net;
    size_t L = net->nlayers - 1;   /* output layer index */
    size_t l, i, j;
    smb_real E = 0;

    const smb_real *out = net_forward(net, x);

    /* Output layer: beta = (d - a) * g'(z), and accumulate the squared error. */
    for (i = 0; i < net->dim[L]; i++) {
        smb_real e = d[i] - out[i];
        E += (smb_real)0.5 * e * e;
        t->beta[L][i] = e * act_sigmoid_deriv(net->a[L][i]);
    }

    /* Hidden layers, output-to-input: beta[l][i] = g'(z) * sum_k w[l+1][k][i]
     * beta[l+1][k]. All betas are computed before any weight changes, so the
     * w[l+1] used here are still the pre-update values. l runs L-1..1; at l==1
     * the decrement yields 0 and the size_t guard `l >= 1` stops the loop. */
    for (l = L - 1; l >= 1; l--) {
        size_t nthis = net->dim[l];
        if (net->kind[l + 1] == LAYER_CONV) {
            /* conv l+1 convolves over layer l; propagate its error back to each
             * unit i of layer l by the conv-transpose: sum over filters f and
             * kernel taps k of w[f][k] * beta[l+1][f, p] where p = i - k. */
            size_t K = net->ksize[l + 1], F = net->nfilt[l + 1];
            size_t opos = net->dim[l] - K + 1;
            for (i = 0; i < nthis; i++) {
                smb_real s = 0;
                size_t   f, k;
                for (f = 0; f < F; f++) {
                    const smb_real *wf = net->w[l + 1] + f * K;
                    for (k = 0; k < K; k++) {
                        if (i >= k && i - k < opos)
                            s += wf[k] * t->beta[l + 1][f * opos + (i - k)];
                    }
                }
                t->beta[l][i] = act_deriv(net->activation, net->a[l][i]) * s;
            }
        } else {
            size_t nnext = net->dim[l + 1];
            for (i = 0; i < nthis; i++) {
                smb_real s = 0;
                for (j = 0; j < nnext; j++)
                    s += net->w[l + 1][j * nthis + i] * t->beta[l + 1][j];
                t->beta[l][i] = act_deriv(net->activation, net->a[l][i]) * s;
            }
        }
    }

    /* Apply the updates: dw = rate*grad + momentum*dw_prev; retain dw. */
    for (l = 1; l <= L; l++) {
        const smb_real *prev = net->a[l - 1];

        if (net->kind[l] == LAYER_CONV) {
            /* shared weights: each kernel tap's gradient sums over all positions
             * it was applied at (that is what makes the weights shared). */
            size_t K = net->ksize[l], F = net->nfilt[l];
            size_t opos = net->dim[l - 1] - K + 1;
            size_t f, k, p;
            for (f = 0; f < F; f++) {
                smb_real *wf  = net->w[l] + f * K;
                smb_real *dwf = t->dw[l] + f * K;
                smb_real  gb  = 0;
                for (k = 0; k < K; k++) {
                    smb_real g = 0, delta;
                    for (p = 0; p < opos; p++)
                        g += t->beta[l][f * opos + p] * prev[p + k];
                    delta = t->rate * g + t->momentum * dwf[k];
                    wf[k]  += delta;
                    dwf[k]  = delta;
                }
                for (p = 0; p < opos; p++)
                    gb += t->beta[l][f * opos + p];
                {
                    smb_real dbf = t->rate * gb + t->momentum * t->db[l][f];
                    net->b[l][f] += dbf;
                    t->db[l][f]   = dbf;
                }
            }
        } else {
            size_t nprev = net->dim[l - 1];
            for (i = 0; i < net->dim[l]; i++) {
                smb_real  bi  = t->beta[l][i];
                smb_real *wi  = net->w[l] + i * nprev;
                smb_real *dwi = t->dw[l] + i * nprev;
                smb_real  dbi;
                for (j = 0; j < nprev; j++) {
                    smb_real delta = t->rate * bi * prev[j] + t->momentum * dwi[j];
                    wi[j]  += delta;
                    dwi[j]  = delta;
                }
                dbi = t->rate * bi + t->momentum * t->db[l][i];   /* bias: a_prev=1 */
                net->b[l][i] += dbi;
                t->db[l][i]   = dbi;
            }
        }
    }
    return E;
}

void trainer_free(Trainer *t)
{
    size_t l;

    if (t == NULL)
        return;
    for (l = 0; l < SMB_MAX_LAYERS; l++) {
        free(t->beta[l]);
        free(t->dw[l]);
        free(t->db[l]);
    }
    free(t);
}
