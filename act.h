/* act.h -- activation functions and their derivatives.
 *
 * The 1997 thesis derives backpropagation with the logistic sigmoid, whose
 * derivative has the convenient closed form g'(z) = a(1-a) in terms of the
 * activation a = g(z) itself (thesis section "Sigmoid-specific form"). That is
 * why the derivative here takes the ALREADY-COMPUTED activation, not z: the
 * forward pass stores a, and backprop reuses it without recomputing g.
 *
 * Its own file so a future ReLU/tanh (the modern defaults noted in the thesis's
 * 2026 editorial asides) drops in beside it without touching net.c or train.c.
 * Pure functions: no allocation, no state.
 */
#ifndef SMB_ACT_H
#define SMB_ACT_H

#include "common.h"

/* Logistic sigmoid g(z) = 1 / (1 + e^-z), mapping the reals into (0, 1). */
smb_real act_sigmoid(smb_real z);

/* Derivative g'(z) expressed via the activation a = g(z): a * (1 - a). */
smb_real act_sigmoid_deriv(smb_real a);

/* Selectable activations, so a network can carry (and evolve) its own. Applied
 * to hidden layers; the output layer stays sigmoid, keeping classification
 * outputs in (0,1). */
enum { ACT_SIGMOID = 0, ACT_TANH = 1, ACT_RELU = 2, ACT_COUNT = 3 };

/* Apply / differentiate activation KIND. act_deriv takes the activation a=g(z),
 * as the sigmoid form does (for ReLU, a>0 agrees with z>0). */
smb_real    act_apply(int kind, smb_real z);
smb_real    act_deriv(int kind, smb_real a);
const char *act_name(int kind);              /* "sigmoid" | "tanh" | "relu" */
int         act_from_name(const char *name); /* kind, or -1 if unknown */

#endif /* SMB_ACT_H */
