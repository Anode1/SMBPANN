/* conv2f.h -- a 2D convolutional front-end (LeCun's operation) for the search.
 *
 * The engine's LAYER_CONV is one-dimensional. This is a native 2D convolution the
 * genome can put in front of the network: F filters of K x K swept over the
 * S x S input image (valid, stride 1), tanh, then a global max-pool per filter to
 * F features that become the network's input. Local receptive fields, weights
 * shared across positions, and the pooling that gives shift invariance -- the
 * inductive bias LeCun designed for images (thesis: shift invariance), here a move
 * the search can reach. Weights live in fixed arrays (no heap), like the rest of
 * the engine; only the max-pool argmax position feeds gradient back, so a filter
 * learns from wherever it fired strongest.
 */
#ifndef SMB_CONV2F_H
#define SMB_CONV2F_H

#include <stddef.h>

#include "common.h"
#include "rng.h"

#define CONV2F_MAXF 8                 /* max filters */
#define CONV2F_MAXK 6                 /* max kernel side */

typedef struct {
    size_t   F, K, S, P;              /* filters, kernel, input side, out side S-K+1 */
    smb_real w[CONV2F_MAXF][CONV2F_MAXK * CONV2F_MAXK];   /* per-filter kernel */
    smb_real b[CONV2F_MAXF];
    smb_real dw[CONV2F_MAXF][CONV2F_MAXK * CONV2F_MAXK];  /* momentum buffers */
    smb_real db[CONV2F_MAXF];
    smb_real pooled[CONV2F_MAXF];     /* forward cache: max-pooled feature */
    size_t   arg[CONV2F_MAXF];        /* forward cache: argmax position ay*P+ax */
} Conv2f;

/* Initialize F filters of K x K over an S x S image; weights uniform in the thesis
 * range [-2.4/fan_in, 2.4/fan_in] with fan_in = K*K. Returns 0, or -1 if the sizes
 * are out of range or the kernel does not fit the image. */
int conv2f_init(Conv2f *c, size_t F, size_t K, size_t S, Rng *rng);

/* Forward pass on the S*S image IMG: returns the F pooled features (owned by c),
 * caching the pooled value and argmax position for the backward pass. */
const smb_real *conv2f_forward(Conv2f *c, const smb_real *img);

/* Backward pass: given DFEAT = dL/d(pooled feature) for each of the F filters,
 * accumulate momentum and update the filter weights (learning rate LR, momentum
 * MOM). Only the max-pool argmax position of each filter contributes. IMG is the
 * same image passed to the forward pass. The image itself is the fixed input, so no
 * gradient is propagated to it. */
void conv2f_backward(Conv2f *c, const smb_real *img, const smb_real *dfeat,
                     smb_real lr, smb_real mom);

#endif /* SMB_CONV2F_H */
