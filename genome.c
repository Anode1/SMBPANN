/* genome.c -- topology genome and its mutation operators. See genome.h. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "genome.h"

/* Self-adaptation constants: TAU is the log-normal step for the rate; the rate
 * is clamped so it never vanishes or explodes. */
#define GENOME_TAU       0.35
#define GENOME_RATE_MIN  0.5
#define GENOME_RATE_MAX  12.0

/* Co-evolution of the training hyper-parameters: a gentler log-normal step for
 * the learning rate and momentum, clamped to sane ranges; an occasional switch
 * of the activation. */
#define GENOME_HP_TAU     0.20
#define GENOME_LR_MIN     0.01
#define GENOME_LR_MAX     2.0
#define GENOME_MOM_MIN    0.0
#define GENOME_MOM_MAX    0.99
#define GENOME_ACT_SWITCH 10      /* percent chance to switch activation */

/* uniform integer in [0, m) via the PRNG */
static size_t below(Rng *rng, size_t m)
{
    return (m == 0) ? 0 : (size_t)(rng_u32(rng) % (uint32_t)m);
}

void genome_random(Genome *g, size_t ninput, size_t noutput,
                   size_t maxhid, size_t maxwidth, Rng *rng)
{
    size_t nhid, i;

    if (maxhid > SMB_MAX_LAYERS - 2)
        maxhid = SMB_MAX_LAYERS - 2;
    nhid = below(rng, maxhid + 1);           /* 0 .. maxhid */

    g->dim[0] = ninput;
    for (i = 0; i < nhid; i++)
        g->dim[1 + i] = 1 + below(rng, maxwidth);
    g->dim[1 + nhid] = noutput;
    g->n = nhid + 2;
    g->rate = 1.0f;                          /* a caller may override the seed */

    /* random starting hyper-parameters, so the population (and the random
     * control) explore them from the first generation */
    g->lrate      = rng_uniform(rng, 0.1f, 0.8f);
    g->momentum   = rng_uniform(rng, 0.5f, 0.95f);
    g->activation = (int)below(rng, ACT_COUNT);
}

void genome_reproduce(Genome *child, const Genome *parent,
                      size_t maxhid, size_t maxwidth, Rng *rng)
{
    long moves, k;

    *child = *parent;
    /* log-normal self-adaptation of the inherited rate, then clamp */
    child->rate *= (smb_real)exp(GENOME_TAU * (double)rng_gaussian(rng));
    if (child->rate < (smb_real)GENOME_RATE_MIN)
        child->rate = (smb_real)GENOME_RATE_MIN;
    if (child->rate > (smb_real)GENOME_RATE_MAX)
        child->rate = (smb_real)GENOME_RATE_MAX;

    /* apply round(rate) topology mutations, at least one */
    moves = (long)((double)child->rate + 0.5);
    if (moves < 1)
        moves = 1;
    for (k = 0; k < moves; k++)
        genome_mutate(child, maxhid, maxwidth, rng);

    /* co-evolve the training hyper-parameters: log-normal steps on the learning
     * rate and momentum, an occasional switch of activation, all clamped */
    child->lrate *= (smb_real)exp(GENOME_HP_TAU * (double)rng_gaussian(rng));
    if (child->lrate < (smb_real)GENOME_LR_MIN) child->lrate = (smb_real)GENOME_LR_MIN;
    if (child->lrate > (smb_real)GENOME_LR_MAX) child->lrate = (smb_real)GENOME_LR_MAX;

    child->momentum *= (smb_real)exp(GENOME_HP_TAU * (double)rng_gaussian(rng));
    if (child->momentum < (smb_real)GENOME_MOM_MIN) child->momentum = (smb_real)GENOME_MOM_MIN;
    if (child->momentum > (smb_real)GENOME_MOM_MAX) child->momentum = (smb_real)GENOME_MOM_MAX;

    if ((rng_u32(rng) % 100u) < (uint32_t)GENOME_ACT_SWITCH)
        child->activation = (int)below(rng, ACT_COUNT);
}

void genome_mutate(Genome *g, size_t maxhid, size_t maxwidth, Rng *rng)
{
    size_t nhid = g->n - 2;
    uint32_t roll;

    if (maxhid > SMB_MAX_LAYERS - 2)
        maxhid = SMB_MAX_LAYERS - 2;

    /* with no hidden layers the only meaningful move is to add one */
    if (nhid == 0) {
        if (maxhid == 0)
            return;
        g->dim[2] = g->dim[1];               /* shift output to index 2 */
        g->dim[1] = 1 + below(rng, maxwidth);
        g->n = 3;
        return;
    }

    roll = rng_u32(rng) % 100u;
    if (roll < 70) {
        /* perturb one hidden width by +/-1, clamped */
        size_t li = 1 + below(rng, nhid);
        if ((rng_u32(rng) & 1u) != 0u) {
            if (g->dim[li] < maxwidth)
                g->dim[li]++;
        } else if (g->dim[li] > 1) {
            g->dim[li]--;
        }
    } else if (roll < 85 && nhid < maxhid && g->n < SMB_MAX_LAYERS) {
        /* grow: insert a hidden layer at a random hidden position */
        size_t pos = 1 + below(rng, nhid + 1);
        size_t w = 1 + below(rng, maxwidth);
        size_t i;
        for (i = g->n; i > pos; i--)
            g->dim[i] = g->dim[i - 1];
        g->dim[pos] = w;
        g->n++;
    } else {
        /* prune: remove a random hidden layer */
        size_t li = 1 + below(rng, nhid);
        size_t i;
        for (i = li; i < g->n - 1; i++)
            g->dim[i] = g->dim[i + 1];
        g->n--;
    }
}

void genome_format(const Genome *g, char *buf, size_t bufsz)
{
    size_t i, off = 0;

    for (i = 0; i < g->n && off < bufsz; i++) {
        int w = snprintf(buf + off, bufsz - off, "%s%zu",
                         (i > 0) ? "," : "", g->dim[i]);
        if (w < 0 || (size_t)w >= bufsz - off)
            return;
        off += (size_t)w;
    }
    /* append the hyper-parameters: |lrate|momentum|activation */
    if (off < bufsz)
        snprintf(buf + off, bufsz - off, "|%.4g|%.4g|%s",
                 (double)g->lrate, (double)g->momentum, act_name(g->activation));
}

int genome_parse(Genome *g, const char *s)
{
    size_t      n = 0;
    const char *p = s;

    /* defaults, used for a bare-topology spec and overwritten if present */
    g->rate       = 1.0f;
    g->lrate      = 0.5f;
    g->momentum   = 0.9f;
    g->activation = ACT_SIGMOID;

    while (*p != '\0' && *p != '|') {
        char *end;
        long  v = strtol(p, &end, 10);
        if (end == p || v <= 0)
            return -1;
        if (n >= SMB_MAX_LAYERS)
            return -1;
        g->dim[n++] = (size_t)v;
        p = end;
        while (*p == ',' || *p == ' ' || *p == '\t' || *p == '\n')
            p++;
    }
    if (n < 2)
        return -1;
    g->n = n;

    /* optional hyper-parameters: |lrate|momentum|activation */
    if (*p == '|') {
        char   act[32];
        double lr = 0.0, mom = 0.0;
        int    got = sscanf(p, "|%lf|%lf|%31[a-z]", &lr, &mom, act);
        if (got >= 1) g->lrate    = (smb_real)lr;
        if (got >= 2) g->momentum = (smb_real)mom;
        if (got >= 3) {
            int kind = act_from_name(act);
            if (kind >= 0)
                g->activation = kind;
        }
    }
    return 0;
}
