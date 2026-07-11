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

#endif /* SMB_ACT_H */
