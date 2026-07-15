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
#include "net.h"
#include "rng.h"

/* A genome carries the topology AND its own training hyper-parameters, so the
 * search discovers them instead of being told (the self_modifying_predict goal):
 * network shape, plus learning rate, momentum, and hidden activation. All of them
 * co-evolve; the mutation rate itself is self-adaptive. */
typedef struct {
    size_t   n;                     /* layers, input and output included (>= 2) */
    size_t   dim[SMB_MAX_LAYERS];   /* widths; dim[0]=inputs, dim[n-1]=outputs;
                                       a conv layer's width is derived, not free */
    int      kind[SMB_MAX_LAYERS];  /* LAYER_DENSE | LAYER_CONV per layer (l>=1) */
    size_t   nfilt[SMB_MAX_LAYERS]; /* conv: filters                            */
    size_t   ksize[SMB_MAX_LAYERS]; /* conv: kernel size                        */
    smb_real rate;                  /* self-adaptive mutation rate (moves/child)*/
    smb_real lrate;                 /* learning rate  (co-evolved)              */
    smb_real momentum;              /* momentum       (co-evolved)              */
    int      activation;            /* hidden activation, act.h kind (co-evolved)*/
    size_t   ninput;                /* problem input width (S*S if a 2D image)  */
    size_t   c2filt;                /* 2D conv front-end filters (0 = none);    */
    size_t   c2ksize;               /* when >0, dim[0] is the F pooled features  */
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

/* Sexual reproduction: recombine two parents into CHILD by UNIFORM crossover of
 * their homologous hidden layers -- layers align by depth from the input, and over
 * the aligned prefix each whole layer is drawn from a uniformly-chosen parent. The
 * child's depth (and any tail beyond the shorter parent) is inherited whole from
 * one parent; the 2D front-end and each object hyper-parameter (learning rate,
 * momentum, activation) are likewise inherited discretely from a random parent,
 * while the self-adaptive mutation rate blends (geometric mean). Then the same
 * self-adaptive step as genome_reproduce. Input and output widths stay fixed. */
void genome_crossover(Genome *child, const Genome *a, const Genome *b,
                      size_t maxhid, size_t maxwidth, Rng *rng);

/* Format G as a candidate spec "topology|lrate|momentum|activation" into BUF,
 * e.g. "2,4,1|0.5|0.9|sigmoid". */
void genome_format(const Genome *g, char *buf, size_t bufsz);

/* Parse a candidate spec into G. Accepts a bare topology ("2,4,1", with default
 * hyper-parameters) or the full "2,4,1|lrate|momentum|activation". Returns 0, or
 * -1 on a malformed or too-long list. */
int genome_parse(Genome *g, const char *s);

#endif /* SMB_GENOME_H */
