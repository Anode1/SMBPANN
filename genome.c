/* genome.c -- topology genome and its mutation operators. See genome.h.
 *
 * A layer is dense (a free width) or conv (nfilt filters of kernel ksize, whose
 * width is derived from the previous layer). Because conv widths are derived,
 * every structural change is followed by genome_recompute, which fills them in
 * and clamps kernels that no longer fit. The output layer is always dense. */
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

/* Conv-layer bounds. */
#define GENOME_MAXFILT   8
#define GENOME_MINK      2
#define GENOME_MAXK      5

/* 2D conv front-end bounds (a perfect-square input is treated as an S x S image). */
#define GENOME_C2_MINK     2
#define GENOME_C2_MAXK     5
#define GENOME_C2_MINSIDE  4

/* integer floor(sqrt(n)) */
static size_t isqrt_(size_t n)
{
    size_t r = 0;
    while ((r + 1) * (r + 1) <= n) r++;
    return r;
}

/* uniform integer in [0, m) via the PRNG */
static size_t below(Rng *rng, size_t m)
{
    return (m == 0) ? 0 : (size_t)(rng_u32(rng) % (uint32_t)m);
}

/* Recompute the width of every conv layer from the chain, clamping any kernel
 * that no longer fits the (possibly mutated) previous layer. Dense widths are
 * left as they are. Keeps a genome always buildable by net_build. */
static void genome_recompute(Genome *g)
{
    size_t l;

    /* a 2D conv front-end feeds the F pooled features to a dense head, so layer 1
     * must be dense, and dim[0] is the front-end's filter count */
    if (g->c2filt > 0) {
        g->dim[0] = g->c2filt;
        if (g->n >= 2 && g->kind[1] == LAYER_CONV)
            g->kind[1] = LAYER_DENSE;
    }
    for (l = 1; l < g->n; l++) {
        if (g->kind[l] == LAYER_CONV) {
            size_t prev = g->dim[l - 1];
            if (g->nfilt[l] < 1)     g->nfilt[l] = 1;
            if (g->ksize[l] < 1)     g->ksize[l] = 1;
            if (g->ksize[l] > prev)  g->ksize[l] = prev;   /* kernel must fit */
            g->dim[l] = g->nfilt[l] * (prev - g->ksize[l] + 1);
        }
    }
}

/* Make layer L a random conv layer (kind/filters/kernel; width filled later). */
static void set_conv(Genome *g, size_t l, Rng *rng)
{
    g->kind[l]  = LAYER_CONV;
    g->nfilt[l] = 1 + below(rng, GENOME_MAXFILT);
    g->ksize[l] = GENOME_MINK + below(rng, GENOME_MAXK - GENOME_MINK + 1);
}

void genome_random(Genome *g, size_t ninput, size_t noutput,
                   size_t maxhid, size_t maxwidth, Rng *rng)
{
    size_t nhid, i;

    g->ninput  = ninput;                     /* init before any genome_recompute */
    g->c2filt  = 0;
    g->c2ksize = 0;

    if (maxhid > SMB_MAX_LAYERS - 2)
        maxhid = SMB_MAX_LAYERS - 2;
    nhid = below(rng, maxhid + 1);           /* 0 .. maxhid */

    g->dim[0]  = ninput;
    g->kind[0] = LAYER_DENSE;
    for (i = 0; i < nhid; i++) {
        size_t l = 1 + i;
        if (below(rng, 2) == 0) {            /* half the hidden layers are conv */
            set_conv(g, l, rng);
        } else {
            g->kind[l] = LAYER_DENSE;
            g->dim[l]  = 1 + below(rng, maxwidth);
        }
    }
    g->kind[1 + nhid] = LAYER_DENSE;         /* output is always dense */
    g->dim[1 + nhid]  = noutput;
    g->n = nhid + 2;
    genome_recompute(g);

    g->rate       = 1.0f;                    /* a caller may override the seed */
    g->lrate      = rng_uniform(rng, 0.1f, 0.8f);
    g->momentum   = rng_uniform(rng, 0.5f, 0.95f);
    g->activation = (int)below(rng, ACT_COUNT);

    /* 2D conv front-end: only for a square (image) input, about 40% of the time.
     * The guard short-circuits before drawing for non-square inputs, so the PRNG
     * stream (and hence every existing non-image run) is unchanged. */
    {
        size_t side = isqrt_(ninput);
        if (side * side == ninput && side >= GENOME_C2_MINSIDE
            && below(rng, 5) < 2) {
            g->c2ksize = GENOME_C2_MINK + below(rng, GENOME_C2_MAXK - GENOME_C2_MINK + 1);
            if (g->c2ksize > side) g->c2ksize = side;
            g->c2filt  = 1 + below(rng, GENOME_MAXFILT);
            if (g->kind[1] == LAYER_CONV) {        /* the front-end feeds a dense head */
                g->kind[1] = LAYER_DENSE;
                g->dim[1]  = 1 + below(rng, maxwidth);
            }
            g->dim[0] = g->c2filt;                 /* net input = F pooled features */
            genome_recompute(g);
        }
    }
}

/* The self-adaptive step shared by asexual reproduction and crossover: perturb
 * the (already-set) mutation rate log-normally, apply round(rate) topology
 * mutations, and co-evolve the training hyper-parameters. Operates in place on a
 * child whose genes are already filled in. */
static void selfadapt(Genome *child, size_t maxhid, size_t maxwidth, Rng *rng)
{
    long moves, k;

    /* log-normal self-adaptation of the mutation rate, then clamp */
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

void genome_reproduce(Genome *child, const Genome *parent,
                      size_t maxhid, size_t maxwidth, Rng *rng)
{
    *child = *parent;
    selfadapt(child, maxhid, maxwidth, rng);
}

void genome_crossover(Genome *child, const Genome *a, const Genome *b,
                      size_t maxhid, size_t maxwidth, Rng *rng)
{
    size_t ahid = a->n - 2, bhid = b->n - 2;    /* hidden-layer counts */
    size_t ca = below(rng, ahid + 1);           /* take hidden 1..ca from A */
    size_t cb = below(rng, bhid + 1);           /* then hidden cb.. from B  */
    size_t nhid, l, i;

    if (maxhid > SMB_MAX_LAYERS - 2)
        maxhid = SMB_MAX_LAYERS - 2;
    nhid = ca + (bhid - cb);                     /* one-point spliced depth */
    if (nhid > maxhid)
        nhid = maxhid;

    /* the 2D conv front-end rides with parent A; set before any genome_recompute */
    child->ninput  = a->ninput;
    child->c2filt  = a->c2filt;
    child->c2ksize = a->c2ksize;

    child->dim[0]  = a->dim[0];                  /* input fixed by the problem */
    child->kind[0] = LAYER_DENSE;
    l = 1;
    for (i = 0; i < ca && (l - 1) < nhid; i++, l++) {         /* prefix of A */
        child->kind[l]  = a->kind[1 + i];
        child->dim[l]   = a->dim[1 + i];
        child->nfilt[l] = a->nfilt[1 + i];
        child->ksize[l] = a->ksize[1 + i];
    }
    for (i = 0; (cb + i) < bhid && (l - 1) < nhid; i++, l++) { /* suffix of B */
        size_t src = 1 + cb + i;
        child->kind[l]  = b->kind[src];
        child->dim[l]   = b->dim[src];
        child->nfilt[l] = b->nfilt[src];
        child->ksize[l] = b->ksize[src];
    }
    child->kind[l] = LAYER_DENSE;                /* output fixed by the problem */
    child->dim[l]  = a->dim[a->n - 1];
    child->n = l + 1;
    genome_recompute(child);                     /* derive any conv widths */

    /* blend the training hyper-parameters: geometric mean for the positive
     * scale ones, arithmetic for momentum, one parent's activation gene */
    child->rate       = (smb_real)sqrt((double)a->rate * (double)b->rate);
    child->lrate      = (smb_real)sqrt((double)a->lrate * (double)b->lrate);
    child->momentum   = (smb_real)(0.5 * ((double)a->momentum + (double)b->momentum));
    child->activation = ((rng_u32(rng) & 1u) != 0u) ? a->activation : b->activation;

    selfadapt(child, maxhid, maxwidth, rng);
}

/* shift all four per-layer arrays right by one from index POS (for a grow). */
static void shift_right(Genome *g, size_t pos)
{
    size_t i;
    for (i = g->n; i > pos; i--) {
        g->dim[i]   = g->dim[i - 1];
        g->kind[i]  = g->kind[i - 1];
        g->nfilt[i] = g->nfilt[i - 1];
        g->ksize[i] = g->ksize[i - 1];
    }
}

/* shift all four per-layer arrays left by one over index LI (for a prune). */
static void shift_left(Genome *g, size_t li)
{
    size_t i;
    for (i = li; i < g->n - 1; i++) {
        g->dim[i]   = g->dim[i + 1];
        g->kind[i]  = g->kind[i + 1];
        g->nfilt[i] = g->nfilt[i + 1];
        g->ksize[i] = g->ksize[i + 1];
    }
}

void genome_mutate(Genome *g, size_t maxhid, size_t maxwidth, Rng *rng)
{
    size_t   nhid = g->n - 2;
    uint32_t roll;

    if (maxhid > SMB_MAX_LAYERS - 2)
        maxhid = SMB_MAX_LAYERS - 2;

    /* 2D conv front-end move (only for a square image input; the guard
     * short-circuits before drawing otherwise, so non-image runs are unchanged):
     * toggle the front-end on/off, or nudge its filter count or kernel size. */
    {
        size_t side = isqrt_(g->ninput);
        if (side * side == g->ninput && side >= GENOME_C2_MINSIDE
            && (rng_u32(rng) % 100u) < 12u) {
            if (g->c2filt == 0) {                          /* turn it on */
                g->c2ksize = GENOME_C2_MINK
                             + below(rng, GENOME_C2_MAXK - GENOME_C2_MINK + 1);
                if (g->c2ksize > side) g->c2ksize = side;
                g->c2filt  = 1 + below(rng, GENOME_MAXFILT);
                if (g->kind[1] == LAYER_CONV) {
                    g->kind[1] = LAYER_DENSE;
                    g->dim[1]  = 1 + below(rng, maxwidth);
                }
                g->dim[0] = g->c2filt;
            } else if ((rng_u32(rng) & 3u) == 0u) {        /* turn it off */
                g->c2filt = 0; g->c2ksize = 0;
                g->dim[0] = g->ninput;
            } else if ((rng_u32(rng) & 1u) != 0u) {        /* nudge filters */
                if ((rng_u32(rng) & 1u) && g->c2filt < GENOME_MAXFILT) g->c2filt++;
                else if (g->c2filt > 1) g->c2filt--;
                g->dim[0] = g->c2filt;
            } else {                                       /* nudge kernel */
                if ((rng_u32(rng) & 1u) && g->c2ksize < GENOME_C2_MAXK
                    && g->c2ksize < side) g->c2ksize++;
                else if (g->c2ksize > GENOME_C2_MINK) g->c2ksize--;
            }
            genome_recompute(g);
            return;                                        /* one move per call */
        }
    }

    /* with no hidden layers the only meaningful move is to add one (dense) */
    if (nhid == 0) {
        if (maxhid == 0)
            return;
        shift_right(g, 1);                   /* push the output to index 2 */
        g->kind[1] = LAYER_DENSE;
        g->dim[1]  = 1 + below(rng, maxwidth);
        g->n = 3;
        genome_recompute(g);
        return;
    }

    roll = rng_u32(rng) % 100u;
    if (roll < 50) {
        /* perturb a random hidden layer's parameter (kind-dependent) */
        size_t li = 1 + below(rng, nhid);
        if (g->kind[li] == LAYER_CONV) {
            if ((rng_u32(rng) & 1u) != 0u) {                  /* filters +/-1 */
                if ((rng_u32(rng) & 1u) && g->nfilt[li] < GENOME_MAXFILT)
                    g->nfilt[li]++;
                else if (g->nfilt[li] > 1)
                    g->nfilt[li]--;
            } else {                                          /* kernel +/-1 */
                if ((rng_u32(rng) & 1u) && g->ksize[li] < GENOME_MAXK)
                    g->ksize[li]++;
                else if (g->ksize[li] > 1)
                    g->ksize[li]--;
            }
        } else {
            if ((rng_u32(rng) & 1u) != 0u) {
                if (g->dim[li] < maxwidth) g->dim[li]++;
            } else if (g->dim[li] > 1) {
                g->dim[li]--;
            }
        }
    } else if (roll < 65) {
        /* flip a hidden layer between dense and conv */
        size_t li = 1 + below(rng, nhid);
        if (g->kind[li] == LAYER_CONV) {
            g->kind[li] = LAYER_DENSE;
            g->dim[li]  = 1 + below(rng, maxwidth);
        } else {
            set_conv(g, li, rng);
        }
    } else if (roll < 82 && nhid < maxhid && g->n < SMB_MAX_LAYERS) {
        /* grow: insert a hidden layer (random kind) at a random hidden spot */
        size_t pos = 1 + below(rng, nhid + 1);
        shift_right(g, pos);
        if (below(rng, 2) == 0) {
            set_conv(g, pos, rng);
        } else {
            g->kind[pos] = LAYER_DENSE;
            g->dim[pos]  = 1 + below(rng, maxwidth);
        }
        g->n++;
    } else {
        /* prune a random hidden layer */
        size_t li = 1 + below(rng, nhid);
        shift_left(g, li);
        g->n--;
    }
    genome_recompute(g);
}

void genome_format(const Genome *g, char *buf, size_t bufsz)
{
    size_t i, off = 0;

    if (g->c2filt > 0 && off < bufsz) {              /* 2D conv front-end prefix */
        int w = snprintf(buf + off, bufsz - off, "C%zu:%zu;",
                         g->c2filt, g->c2ksize);
        if (w > 0 && (size_t)w < bufsz - off)
            off += (size_t)w;
    }
    for (i = 0; i < g->n && off < bufsz; i++) {
        int w;
        if (g->kind[i] == LAYER_CONV)
            w = snprintf(buf + off, bufsz - off, "%sc%zu:%zu",
                         (i > 0) ? "," : "", g->nfilt[i], g->ksize[i]);
        else
            w = snprintf(buf + off, bufsz - off, "%s%zu",
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
    g->ninput     = 0;
    g->c2filt     = 0;
    g->c2ksize    = 0;

    /* optional 2D conv front-end prefix "C<filters>:<kernel>;" */
    if (*p == 'C') {
        char *end;
        long  F = strtol(p + 1, &end, 10), K;
        if (end != p + 1 && F > 0 && *end == ':') {
            K = strtol(end + 1, &end, 10);
            if (K > 0 && *end == ';') {
                g->c2filt  = (size_t)F;
                g->c2ksize = (size_t)K;
                p = end + 1;
            }
        }
    }

    while (*p != '\0' && *p != '|') {
        if (n >= SMB_MAX_LAYERS)
            return -1;
        if (*p == 'c') {                     /* conv token: cF:K */
            char *end;
            long  F, K;
            F = strtol(p + 1, &end, 10);
            if (end == p + 1 || F <= 0 || *end != ':')
                return -1;
            p = end + 1;
            K = strtol(p, &end, 10);
            if (end == p || K <= 0)
                return -1;
            g->kind[n]  = LAYER_CONV;
            g->nfilt[n] = (size_t)F;
            g->ksize[n] = (size_t)K;
            g->dim[n]   = 0;                 /* filled by genome_recompute */
            p = end;
        } else {                             /* dense width */
            char *end;
            long  v = strtol(p, &end, 10);
            if (end == p || v <= 0)
                return -1;
            g->kind[n] = LAYER_DENSE;
            g->dim[n]  = (size_t)v;
            p = end;
        }
        n++;
        while (*p == ',' || *p == ' ' || *p == '\t' || *p == '\n')
            p++;
    }
    if (n < 2)
        return -1;
    g->n = n;
    genome_recompute(g);                     /* derive conv widths from the chain */

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
