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
            break;
        off += (size_t)w;
    }
}

int genome_parse(Genome *g, const char *s)
{
    size_t      n = 0;
    const char *p = s;

    while (*p != '\0') {
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
    g->rate = 1.0f;   /* topology strings carry no rate; default it */
    return 0;
}
