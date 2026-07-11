/* act.c -- activation functions. See act.h. Pure, no allocation. */
#include <math.h>

#include "act.h"

smb_real act_sigmoid(smb_real z)
{
    return (smb_real)(1.0 / (1.0 + exp(-(double)z)));
}

smb_real act_sigmoid_deriv(smb_real a)
{
    return a * ((smb_real)1.0 - a);
}
