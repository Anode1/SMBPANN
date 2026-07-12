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

#include "act.h"
#include "common.h"
#include "rng.h"

/* A genome carries the topology AND its own training hyper-parameters, so the
 * search discovers them instead of being told (the self_modifying_predict goal):
 * network shape, plus learning rate, momentum, and hidden activation. All of them
 * co-evolve; the mutation rate itself is self-adaptive. */
typedef struct {
    size_t   n;                     /* layers, input and output included (>= 2) */
    size_t   dim[SMB_MAX_LAYERS];   /* widths; dim[0]=inputs, dim[n-1]=outputs */
    smb_real rate;                  /* self-adaptive mutation rate (moves/child)*/
    smb_real lrate;                 /* learning rate  (co-evolved)              */
    smb_real momentum;              /* momentum       (co-evolved)              */
    int      activation;            /* hidden activation, act.h kind (co-evolved)*/
} Genome;

/* A random genome: NINPUT inputs, NOUTPUT outputs, and 0..MAXHID hidden layers
 * of width 1..MAXWIDTH. */
void genome_random(Genome *g, size_t ninput, size_t noutput,
                   size_t maxhid, size_t maxwidth, Rng *rng);

/* One small mutation in place: perturb a hidden width by one, or add or remove
 * a single hidden layer, staying within [1, MAXWIDTH] and 0..MAXHID hidden
 * layers. The input and output widths never change. Does not touch the rate. */
void genome_mutate(Genome *g, size_t maxhid, size_t maxwidth, Rng *rng);

/* Self-adaptive reproduction: copy PARENT into CHILD, log-normally perturb the
 * inherited mutation rate, apply round(rate) topology mutations, and co-evolve
 * the hyper-parameters (log-normal steps on learning rate and momentum, an
 * occasional switch of activation). Selection on the offspring tunes them all. */
void genome_reproduce(Genome *child, const Genome *parent,
                      size_t maxhid, size_t maxwidth, Rng *rng);

/* Format G as a candidate spec "topology|lrate|momentum|activation" into BUF,
 * e.g. "2,4,1|0.5|0.9|sigmoid". */
void genome_format(const Genome *g, char *buf, size_t bufsz);

/* Parse a candidate spec into G. Accepts a bare topology ("2,4,1", with default
 * hyper-parameters) or the full "2,4,1|lrate|momentum|activation". Returns 0, or
 * -1 on a malformed or too-long list. */
int genome_parse(Genome *g, const char *s);

#endif /* SMB_GENOME_H */
