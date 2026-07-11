/* genome.h -- a network-topology genome for the architecture search.
 *
 * A genome is a topology: a list of layer widths, with the input and output
 * widths fixed by the problem and the hidden layers evolvable. Mutation makes
 * one small, local change (perturb a hidden width by one, or add or remove a
 * single hidden layer), so the search moves in small steps -- the mutation-only
 * evolutionary-NAS approach (Real et al.), not topology crossover. All
 * randomness flows through rng.h, so a seed reproduces a whole run.
 */
#ifndef SMB_GENOME_H
#define SMB_GENOME_H

#include <stddef.h>

#include "common.h"
#include "rng.h"

typedef struct {
    size_t n;                     /* layers, input and output included (>= 2) */
    size_t dim[SMB_MAX_LAYERS];   /* widths; dim[0]=inputs, dim[n-1]=outputs  */
} Genome;

/* A random genome: NINPUT inputs, NOUTPUT outputs, and 0..MAXHID hidden layers
 * of width 1..MAXWIDTH. */
void genome_random(Genome *g, size_t ninput, size_t noutput,
                   size_t maxhid, size_t maxwidth, Rng *rng);

/* One small mutation in place: perturb a hidden width by one, or add or remove
 * a single hidden layer, staying within [1, MAXWIDTH] and 0..MAXHID hidden
 * layers. The input and output widths never change. */
void genome_mutate(Genome *g, size_t maxhid, size_t maxwidth, Rng *rng);

/* Format G as "2,4,1" into BUF. */
void genome_format(const Genome *g, char *buf, size_t bufsz);

/* Parse "2,4,1" into G. Returns 0, or -1 on a malformed or too-long list. */
int genome_parse(Genome *g, const char *s);

#endif /* SMB_GENOME_H */
