/* act.c -- activation functions. See act.h. Pure, no allocation. */
#include <math.h>
#include <string.h>

#include "act.h"

/* Numerically stable logistic: never Exp() a large positive argument. */
static smb_real logistic(double z)
{
    if (z >= 0.0)
        return (smb_real)(1.0 / (1.0 + exp(-z)));
    else {
        double e = exp(z);
        return (smb_real)(e / (1.0 + e));
    }
}

smb_real act_sigmoid(smb_real z)
{
    return logistic((double)z);
}

smb_real act_sigmoid_deriv(smb_real a)
{
    return a * ((smb_real)1.0 - a);
}

smb_real act_apply(int kind, smb_real z)
{
    switch (kind) {
    case ACT_TANH: return (smb_real)tanh((double)z);
    case ACT_RELU: return (z > (smb_real)0.0) ? z : (smb_real)0.0;
    default:       return logistic((double)z);
    }
}

smb_real act_deriv(int kind, smb_real a)
{
    switch (kind) {
    case ACT_TANH: return (smb_real)1.0 - a * a;
    case ACT_RELU: return (a > (smb_real)0.0) ? (smb_real)1.0 : (smb_real)0.0;
    default:       return a * ((smb_real)1.0 - a);
    }
}

const char *act_name(int kind)
{
    switch (kind) {
    case ACT_TANH: return "tanh";
    case ACT_RELU: return "relu";
    default:       return "sigmoid";
    }
}

int act_from_name(const char *name)
{
    if (name == NULL)
        return -1;
    if (strcmp(name, "sigmoid") == 0) return ACT_SIGMOID;
    if (strcmp(name, "tanh") == 0)    return ACT_TANH;
    if (strcmp(name, "relu") == 0)    return ACT_RELU;
    return -1;
}
