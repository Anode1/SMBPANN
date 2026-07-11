/* train.h -- backpropagation: the generalized delta rule with momentum.
 *
 * This is the 1997 thesis's derivation made executable (thesis sections
 * "Backpropagation" through "Learning rate and momentum"):
 *
 *   output units:  beta[i] = (d[i] - a[i]) * g'(z[i])
 *   hidden units:  beta[i] = g'(z[i]) * sum_k w[l+1][k][i] * beta[l+1][k]
 *   update:        dw = rate * beta[i] * a_prev[j] + momentum * dw_prev
 *                  w  += dw            (biases: a_prev == 1)
 *
 * "beta" is the thesis's name for the local error signal (the delta of the
 * standard literature). Momentum carries a fraction of the previous weight
 * change (thesis eq. for Delta w(t+1)); it needs the previous dw/db retained,
 * so a Trainer owns that scratch. Like Net, a Trainer allocates ONCE here and
 * the per-example update touches no allocator.
 *
 * The Trainer is separate from the Net on purpose: the Net is the model (what a
 * trained network exports and what inference needs); the Trainer is transient
 * learning machinery. In the GA, many workers infer over shared network shapes
 * without paying for backprop scratch they will not use.
 */
#ifndef SMB_TRAIN_H
#define SMB_TRAIN_H

#include "common.h"
#include "net.h"

/* Backpropagation state over one network: the hyper-parameters plus the scratch
 * the delta rule needs (error signals and the previous weight/bias deltas for
 * momentum). Pointer tables fixed-size; buffers heap, sized by the net. */
typedef struct {
    Net     *net;                     /* the network being trained (borrowed) */
    smb_real rate;                    /* learning rate r                      */
    smb_real momentum;                /* momentum coefficient alpha in [0,1)  */
    smb_real *beta[SMB_MAX_LAYERS];   /* beta[l]: dim[l] error signals        */
    smb_real *dw[SMB_MAX_LAYERS];     /* previous weight deltas, w's shape     */
    smb_real *db[SMB_MAX_LAYERS];     /* previous bias deltas, b's shape       */
} Trainer;

/* Create a trainer over `net` with the given learning rate and momentum.
 * Allocates its scratch once (previous deltas zeroed). Returns NULL on bad
 * arguments or allocation failure, freeing any partial allocation. */
Trainer *trainer_new(Net *net, smb_real rate, smb_real momentum);

/* One online (stochastic) backprop step on example (x, d): forward pass,
 * back-propagate the error, update every weight and bias in place. Returns the
 * example's error E = 0.5 * sum_i (d[i] - a[i])^2 measured BEFORE the update.
 * Allocates nothing. */
smb_real trainer_learn(Trainer *t, const smb_real *x, const smb_real *d);

/* Release the trainer's scratch. NULL-safe. Does NOT free the borrowed net. */
void trainer_free(Trainer *t);

#endif /* SMB_TRAIN_H */
