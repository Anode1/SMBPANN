/* rng.c -- 32-bit xorshift PRNG. See rng.h. Pure, no allocation. */
#include "rng.h"

void rng_seed(Rng *r, uint32_t seed)
{
    /* xorshift cannot escape the all-zero state, so map 0 to a fixed nonzero
     * constant: a 0 seed stays reproducible instead of producing all zeros. */
    r->s = (seed != 0u) ? seed : 0x9e3779b9u;
}

uint32_t rng_u32(Rng *r)
{
    uint32_t x = r->s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    r->s = x;
    return x;
}

smb_real rng_uniform(Rng *r, smb_real lo, smb_real hi)
{
    /* Map the 32-bit word to [0,1) with the full 2^32 divisor, then scale. */
    smb_real u = (smb_real)(rng_u32(r) / 4294967296.0);
    return lo + u * (hi - lo);
}

smb_real rng_gaussian(Rng *r)
{
    /* Sum of 12 uniforms in [0,1) has mean 6 and variance 1, so subtracting 6
     * gives an approximate N(0,1) without a libm call. */
    double s = 0.0;
    int    i;

    for (i = 0; i < 12; i++)
        s += (double)rng_uniform(r, 0.0f, 1.0f);
    return (smb_real)(s - 6.0);
}
