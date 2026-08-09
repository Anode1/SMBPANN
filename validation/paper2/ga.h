/* ga.h -- the outer search, shared by paper-2 probes. C99, header-only, no dependencies.
 *
 * WHY THIS IS A COMPILE-TIME TEMPLATE AND NOT AN INTERFACE. A shared GA is the one piece of common
 * code that CAN change a result, because it owns the order in which random numbers are drawn. The
 * usual way to share it -- a struct of function pointers -- would buy that risk twice over: an
 * indirection at every call, and a probe you can no longer read top to bottom. So instead the probe
 * defines its own symbols and includes this file LAST. Everything below binds at compile time,
 * inlines, and reads as if it had been written in the probe.
 *
 * THE CONTRACT. Before including this header a probe must have defined, with exactly these names:
 *
 *     POP, ELITE                 population size and elite count (macros)
 *     Indiv                      the genome type (a plain struct; copied by assignment)
 *     void   seed_individual(Indiv *g)                     one member of the initial population
 *     void   mutate(Indiv *g)                              in place; owns ALL of its RNG calls
 *     double fitness(const Indiv *g, double acc)           the outer objective
 *     double train_eval(const Indiv *g, uint32_t s, int t) the inner learner; t=1 selects the test set
 *     void   ga_summarize(const Indiv *pop, const double *fit, int best, double *out)
 *                                                          probe-specific statistics of a finished run
 *     int g_gens, g_raw, g_arm                             knobs read here
 *
 * A missing symbol is a compile error naming it, which is the whole appeal: the contract is checked
 * by the compiler rather than documented and hoped for.
 *
 * WHAT IS DELIBERATELY NOT HERE. The genome, its mutation, the inner learner and the fitness. Those
 * ARE the experiment. This file only decides who gets copied and in what order, which is the part
 * every probe agrees on and no probe should be re-deriving.
 *
 * FAITHFULNESS IS TESTED, NOT ASSERTED. The loop below is the hand-written loop from
 * emerge_rewire.c (version 1), unchanged in structure, so the RNG call order is identical:
 * one r32() to pick a parent, then the probe's mutate(). emerge_rewire2 must reproduce version 1 and
 * the archived per-seed data in ../../scratch_rewire_*.out exactly. If a change here ever breaks that,
 * this file is wrong -- and because it is shared, it would be wrong for every paper-2 probe at once.
 * Rerun the acceptance test after touching anything below.
 *
 * GENS=0 leaves the seeded population untouched and is therefore the no-evolution control, free.
 */
#ifndef SMB_PAPER_GA_H
#define SMB_PAPER_GA_H

static void run_ga(uint32_t seed, int sd, double *out)
{
    static Indiv pop[POP], nxt[POP];
    double fit[POP];
    int idx[POP], g, p, q, best = 0;
    double bf = -1e300, test;

    rseed(seed);
    for(p=0;p<POP;p++) seed_individual(&pop[p]);
    for(p=0;p<POP;p++) fit[p] = fitness(&pop[p], train_eval(&pop[p],(uint32_t)(seed+p*2654435761u+1u),0));

    for(g=0; g<g_gens; g++){
        /* rank by fitness; no randomness in the sort, so ties resolve identically every run */
        for(p=0;p<POP;p++) idx[p]=p;
        for(p=0;p<POP;p++) for(q=p+1;q<POP;q++)
            if(fit[idx[q]]>fit[idx[p]]){ int t=idx[p]; idx[p]=idx[q]; idx[q]=t; }

        for(p=0;p<ELITE;p++) nxt[p]=pop[idx[p]];          /* elites survive unchanged */
        for(p=ELITE;p<POP;p++){                           /* the rest: copy an elite, mutate it */
            int a = idx[(int)(r32()%ELITE)];
            nxt[p] = pop[a];
            mutate(&nxt[p]);
        }
        memcpy(pop, nxt, sizeof pop);

        for(p=0;p<POP;p++) fit[p] = fitness(&pop[p], train_eval(&pop[p],(uint32_t)(seed+(uint32_t)(g*POP+p)+7u),0));
    }

    for(p=0;p<POP;p++) if(fit[p]>bf){ bf=fit[p]; best=p; }
    test = train_eval(&pop[best], seed+999u, 1);          /* held out; weight stream only */
    ga_summarize(pop, fit, best, test, out);

    /* PROTOCOL item 2: one row per run, carrying whatever free variables the probe named, so an arm
     * effect is read as an increment over them rather than as a raw difference. */
    if(g_raw) ga_raw_row(pop, best, sd, test);
}

#endif /* SMB_PAPER_GA_H */
