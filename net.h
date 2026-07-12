/* net.h -- a feed-forward network: flat weight buffers and the forward pass.
 *
 * The Java original modelled Network -> Layer -> Neuron -> Edge as an object
 * graph. In C, for the memory discipline and the cache behaviour SMBPANN needs
 * at scale (see doc/dev/STYLE.md and doc/dev/WHY-FLAT.md), that graph collapses
 * into flat, contiguous weight matrices: layer l holds a dim[l] x dim[l-1]
 * matrix in row-major order, one weight per (receiving unit, sending unit) pair.
 * "Row receives, column sends" -- the convention of the 1997 thesis.
 *
 * Feed-forward computation (thesis section "Feed-Forward Networks"):
 *     z[l] = W[l] a[l-1] + b[l]
 *     a[l] = g(z[l])
 * z[l] is retained because backpropagation needs the pre-activation (train.c).
 *
 * Memory: every buffer is allocated ONCE in net_new and freed in net_free.
 * net_forward allocates nothing. See doc/dev/STYLE.md for why this single
 * construction-time allocation is the sanctioned departure from AIS's no-heap
 * core: a network's weights ARE its data.
 */
#ifndef SMB_NET_H
#define SMB_NET_H

#include <stddef.h>

#include "common.h"
#include "rng.h"

/* Layer kinds. A DENSE layer is fully connected (w is dim[l] x dim[l-1]). A CONV
 * layer is a 1D convolution: nfilt filters of kernel size ksize, stride 1, valid
 * padding, applied over the previous layer as a signal -- weights are SHARED
 * across positions, connectivity is LOCAL. That sharing plus locality is the
 * convolution inductive bias (thesis: LeCun's shift invariance), here an
 * evolvable structural choice rather than a hand-set one. */
enum { LAYER_DENSE = 0, LAYER_CONV = 1 };

/* A network of `nlayers` layers including the input layer (index 0, which has
 * no weights or biases of its own). Pointer tables are fixed-size (bounded by
 * SMB_MAX_LAYERS); the buffers they point at are heap, sized by topology. */
typedef struct {
    size_t    nlayers;               /* total layers, input included (>= 2) */
    size_t    dim[SMB_MAX_LAYERS];   /* units in each layer (computed for conv)*/
    int       kind[SMB_MAX_LAYERS];  /* LAYER_DENSE | LAYER_CONV (l >= 1)    */
    size_t    nfilt[SMB_MAX_LAYERS]; /* conv: number of filters             */
    size_t    ksize[SMB_MAX_LAYERS]; /* conv: kernel size                   */
    smb_real *w[SMB_MAX_LAYERS];     /* dense: dim[l] x dim[l-1]; conv: F x K */
    smb_real *b[SMB_MAX_LAYERS];     /* dense: dim[l]; conv: F biases        */
    smb_real *a[SMB_MAX_LAYERS];     /* a[l]: dim[l] activations (a[0]=input)*/
    smb_real *z[SMB_MAX_LAYERS];     /* z[l]: dim[l] pre-activations (l>=1)  */
    int       activation;            /* hidden-layer activation (act.h kinds)*/
} Net;

/* Build a network with layer sizes dims[0..nlayers-1] (dims[0] = inputs,
 * dims[nlayers-1] = outputs). Allocates all buffers once. Returns NULL on bad
 * arguments (dims NULL, nlayers outside [2, SMB_MAX_LAYERS], any dim 0) or on
 * allocation failure -- and on failure frees whatever it had taken. All layers
 * are dense; use net_build for a mix that includes convolutional layers. */
Net *net_new(const size_t *dims, size_t nlayers);

/* Build a network with per-layer kinds. For layer l>=1: kind[l]==LAYER_DENSE
 * has width dims[l]; kind[l]==LAYER_CONV has nfilt[l] filters of kernel ksize[l]
 * and a computed width (the kernel must fit the previous layer). kind/nfilt/ksize
 * may be NULL to mean all-dense. The output layer is always dense. */
Net *net_build(const size_t *dims, const int *kind,
               const size_t *nfilt, const size_t *ksize, size_t nlayers);

/* Weights / biases / conv output positions in layer l (l >= 1). */
size_t net_layer_wsize(const Net *net, size_t l);
size_t net_layer_bsize(const Net *net, size_t l);
size_t net_conv_positions(const Net *net, size_t l);

/* Initialize weights and biases to uniform random values in the thesis range
 * [-2.4/fan_in, +2.4/fan_in], where fan_in is the sending layer's width. */
void net_init(Net *net, Rng *rng);

/* Forward pass: copy the dim[0] inputs from x, propagate through every layer,
 * and return the output activation vector a[nlayers-1] (dim[nlayers-1] values,
 * owned by the net). Allocates nothing. */
const smb_real *net_forward(Net *net, const smb_real *x);

/* Total number of weights across all layers (topology size; excludes biases). */
size_t net_nweights(const Net *net);

/* Release all buffers. NULL-safe; safe on a partially-constructed net. */
void net_free(Net *net);

#endif /* SMB_NET_H */
