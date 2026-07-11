/* rng.h -- a small deterministic pseudo-random generator.
 *
 * Reproducibility is a first-class requirement: the same seed must give the
 * same weight initialization and the same GA run on every machine, so results
 * are comparable and bugs are reproducible. This is a 32-bit xorshift (Marsaglia
 * 2003): tiny, fast, no dependencies, good enough for weight init and evolution.
 * It is NOT cryptographic. Pure: no allocation, caller owns the state.
 */
#ifndef SMB_RNG_H
#define SMB_RNG_H

#include <stdint.h>

#include "common.h"

/* The whole generator state: one 32-bit word. Copyable by value. */
typedef struct {
    uint32_t s;
} Rng;

/* Seed the generator. Any seed is accepted; 0 is remapped (xorshift cannot
 * leave the zero state), so a 0 seed is reproducible, not degenerate. */
void rng_seed(Rng *r, uint32_t seed);

/* Next 32-bit value, advancing the state. */
uint32_t rng_u32(Rng *r);

/* Uniform real in [lo, hi). */
smb_real rng_uniform(Rng *r, smb_real lo, smb_real hi);

/* Approximately standard-normal N(0,1), via the central-limit sum of 12 uniforms
 * (no libm dependency); adequate for self-adaptive mutation. */
smb_real rng_gaussian(Rng *r);

#endif /* SMB_RNG_H */
