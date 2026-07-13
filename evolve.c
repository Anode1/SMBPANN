/* evolve.c -- the evolutionary architecture search, with a random control.
 *
 * The genetic search each generation: evaluate a population of topologies (in
 * parallel, via scripts/evaluate.sh, one worker process per candidate), keep the
 * best few (elite), and refill by mutating them. Alongside it, a matched-compute
 * RANDOM SEARCH evaluates the same number of freshly-random topologies each
 * generation. Both track best-so-far, so the run answers the project's actual
 * question: does evolution beat random search at equal compute?
 *
 * Evaluation is delegated to the shell coordinator; this program owns only the
 * reproducible genome logic (rng.h + genome.h). Run it from the repo root, or
 * set SMB_EVAL (the coordinator) and SMB (the worker) to absolute paths.
 */
#ifndef UNIT_TEST

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "common.h"
#include "genome.h"
#include "rng.h"

#define EV_MAX_POP 64

static void usage(const char *prog)
{
    fprintf(stderr,
        "usage: %s -i in -o out [-P pop] [-G gens] [-k elite] [-s seed]\n"
        "          [-L maxhidden] [-W maxwidth] [-M rate] [-A 0|1] [-E error]\n"
        "          [-X crossover%%]\n"
        "\n"
        "Evolves network topologies and races the GA against random search.\n"
        "Fitness comes from scripts/evaluate.sh (set COMMON for a dataset; with\n"
        "none it trains XOR, so use -i 2 -o 1).\n"
        "  -E error  stop when the GA's best validation error reaches this target\n"
        "            (self_modifying_predict: search until good enough); -G is then\n"
        "            the safety cap. Default 0 = run the full -G generations.\n", prog);
}

/* uniform index in [0, m) */
static size_t below_pop(Rng *rng, size_t m)
{
    return (m == 0) ? 0 : (size_t)(rng_u32(rng) % (uint32_t)m);
}

/* Write a population as one topology per line. Returns 0 or -1. */
static int write_pop(const char *path, const Genome *pop, size_t n)
{
    FILE  *f = fopen(path, "w");
    char   buf[256];
    size_t i;

    if (f == NULL)
        return -1;
    for (i = 0; i < n; i++) {
        genome_format(&pop[i], buf, sizeof buf);
        fprintf(f, "%s\n", buf);
    }
    fclose(f);
    return 0;
}

/* Run the coordinator on POPFILE, writing its sorted leaderboard to RESFILE,
 * then read the best fitness and the top-K topology strings (sorted best-first).
 * Topology strings only: the genome's rate lives in memory and does not cross
 * the shell seam, so the caller recovers it by matching. Returns 0 (and
 * *NTOP >= 1), or -1 if nothing evaluated. */
static int evaluate(const char *evalcmd, const char *popfile, const char *resfile,
                    char (*topos)[256], size_t k, size_t *ntop, double *best)
{
    char   cmd[1024];
    char   line[512];
    FILE  *f;
    size_t got = 0;

    *best = 0.0;
    snprintf(cmd, sizeof cmd, "%s %s > %s 2>/dev/null", evalcmd, popfile, resfile);
    if (system(cmd) == -1)
        return -1;
    f = fopen(resfile, "r");
    if (f == NULL)
        return -1;
    while (fgets(line, sizeof line, f) != NULL) {
        char       *tp = strstr(line, "spec=");
        const char *fp = strstr(line, "fitness=");
        if (tp == NULL || fp == NULL)
            continue;
        if (got == 0)
            *best = atof(fp + 8);            /* "fitness=" is 8 chars; best first */
        if (got < k && sscanf(tp + 5, "%255[^ \t\n]", topos[got]) == 1)
            got++;                            /* "spec=" is 5 chars; full genome */
    }
    fclose(f);
    *ntop = got;
    return (got > 0) ? 0 : -1;
}

int main(int argc, char **argv)
{
    long   ninput = 0, noutput = 0;
    long   pop = 8, gens = 10, elite = 2, seed = 1, maxhid = 3, maxwidth = 16;
    long   mutations = 1;   /* the mutation rate (initial, if self-adaptive) */
    long   adapt = 1;       /* 1 = self-adaptive rate, 0 = fixed rate         */
    long   xover = 0;       /* percent of offspring made by crossover (0=off) */
    double target = 0.0;    /* stop when GA best <= this; 0 = disabled        */
    int    ga_hit = 0, rand_hit = 0;
    long   ga_hit_gen = 0, rand_hit_gen = 0;   /* gen each first met the target */
    int    c;
    long   gen, total_evals = 0;
    size_t i;

    const char *evalcmd;
    char        popfile[64], resfile[64];
    Rng         ga_rng, rand_rng;
    Genome      gapop[EV_MAX_POP], randpop[EV_MAX_POP], eliteg[EV_MAX_POP];
    char        etop[EV_MAX_POP][256];
    double      ga_best = 0.0, rand_best = 0.0;
    smb_real    ga_rate = 1.0f;
    char        ga_topo[256] = "", rand_topo[256] = "";
    int         have_ga = 0, have_rand = 0;

    while ((c = getopt(argc, argv, "i:o:P:G:k:s:L:W:M:A:E:X:h")) != -1) {
        switch (c) {
        case 'i': ninput    = atol(optarg); break;
        case 'o': noutput   = atol(optarg); break;
        case 'P': pop       = atol(optarg); break;
        case 'G': gens      = atol(optarg); break;
        case 'k': elite     = atol(optarg); break;
        case 's': seed      = atol(optarg); break;
        case 'L': maxhid    = atol(optarg); break;
        case 'W': maxwidth  = atol(optarg); break;
        case 'M': mutations = atol(optarg); break;
        case 'A': adapt     = atol(optarg); break;
        case 'E': target    = atof(optarg); break;
        case 'X': xover     = atol(optarg); break;
        case 'h': usage(argv[0]); return 0;
        default:  usage(argv[0]); return 2;
        }
    }
    if (ninput <= 0 || noutput <= 0) {
        fprintf(stderr, "evolve: -i and -o are required and positive\n");
        return 2;
    }
    if (pop < 2 || pop > EV_MAX_POP || elite < 1 || elite >= pop
        || gens < 1 || maxwidth < 1 || mutations < 1
        || (adapt != 0 && adapt != 1) || xover < 0 || xover > 100) {
        fprintf(stderr, "evolve: need 2 <= pop <= %d, 1 <= elite < pop, "
                        "gens >= 1, maxwidth >= 1, mutations >= 1, A in {0,1}, "
                        "0 <= X <= 100\n",
                EV_MAX_POP);
        return 2;
    }

    evalcmd = getenv("SMB_EVAL");
    if (evalcmd == NULL)
        evalcmd = "scripts/evaluate.sh";
    snprintf(popfile, sizeof popfile, "/tmp/smb_evolve_%ld.pop", (long)getpid());
    snprintf(resfile, sizeof resfile, "/tmp/smb_evolve_%ld.res", (long)getpid());

    /* Separate PRNG streams for the search and the control, both seeded from -s
     * so the whole race is reproducible. */
    rng_seed(&ga_rng, (uint32_t)seed);
    rng_seed(&rand_rng, (uint32_t)seed ^ 0x5bd1e995u);

    for (i = 0; i < (size_t)pop; i++) {
        genome_random(&gapop[i], (size_t)ninput, (size_t)noutput,
                      (size_t)maxhid, (size_t)maxwidth, &ga_rng);
        gapop[i].rate = (smb_real)mutations;   /* seed the self-adaptive rate */
    }

    if (target > 0.0)
        printf("evolving %ld topologies until error <= %g "
               "(cap %ld generations; elite %ld, %s rate %ld, seed %ld)\n",
               pop, target, gens, elite,
               adapt ? "initial self-adaptive" : "fixed", mutations, seed);
    else
        printf("evolving %ld topologies over %ld generations "
               "(elite %ld, %s rate %ld, seed %ld)\n",
               pop, gens, elite, adapt ? "initial self-adaptive" : "fixed",
               mutations, seed);
    if (xover > 0)
        printf("  crossover: %ld%% of offspring recombine two elites\n", xover);

    for (gen = 1; gen <= gens; gen++) {
        size_t ne, e;
        double gbest, rbest;
        char   rtop[1][256];
        size_t rne;

        /* --- the genetic search --- */
        if (write_pop(popfile, gapop, (size_t)pop) != 0
            || evaluate(evalcmd, popfile, resfile, etop, (size_t)elite,
                        &ne, &gbest) != 0) {
            fprintf(stderr, "evolve: evaluation failed (is %s runnable?)\n",
                    evalcmd);
            return 1;
        }
        total_evals += pop;

        /* Recover the elite genomes -- with their evolved rates -- by matching
         * the leaderboard's top topologies back to the population we submitted.
         * (The rate lives only in memory; it never crosses the shell seam.) */
        for (e = 0; e < ne; e++) {
            char   buf[256];
            size_t j;
            int    found = 0;
            for (j = 0; j < (size_t)pop; j++) {
                genome_format(&gapop[j], buf, sizeof buf);
                if (strcmp(buf, etop[e]) == 0) {
                    eliteg[e] = gapop[j];
                    found = 1;
                    break;
                }
            }
            if (!found) {                       /* should not happen; be safe */
                genome_parse(&eliteg[e], etop[e]);
                eliteg[e].rate = (smb_real)mutations;
            }
        }

        if (!have_ga || gbest < ga_best) {
            ga_best = gbest;
            snprintf(ga_topo, sizeof ga_topo, "%s", etop[0]);
            have_ga = 1;
        }
        ga_rate = eliteg[0].rate;               /* current best individual's rate */

        /* next generation: keep the elite, refill by self-adaptive reproduction */
        for (i = 0; i < ne; i++)
            gapop[i] = eliteg[i];
        for (i = ne; i < (size_t)pop; i++) {
            /* a fraction of offspring are made by crossover of two elites; the
             * rest by asexual (self-adaptive or fixed) reproduction. With xover=0
             * the first test short-circuits without drawing, so behaviour and the
             * PRNG stream are identical to the mutation-only search. */
            if (xover > 0 && (rng_u32(&ga_rng) % 100u) < (uint32_t)xover) {
                const Genome *pa = &eliteg[below_pop(&ga_rng, ne)];
                const Genome *pb = &eliteg[below_pop(&ga_rng, ne)];
                genome_crossover(&gapop[i], pa, pb,
                                 (size_t)maxhid, (size_t)maxwidth, &ga_rng);
            } else {
                const Genome *parent = &eliteg[below_pop(&ga_rng, ne)];
                if (adapt) {
                    genome_reproduce(&gapop[i], parent,
                                     (size_t)maxhid, (size_t)maxwidth, &ga_rng);
                } else {
                    long mv;
                    gapop[i] = *parent;
                    for (mv = 0; mv < mutations; mv++)
                        genome_mutate(&gapop[i], (size_t)maxhid, (size_t)maxwidth,
                                      &ga_rng);
                }
            }
        }

        /* --- matched-compute random control: pop fresh random genomes --- */
        for (i = 0; i < (size_t)pop; i++)
            genome_random(&randpop[i], (size_t)ninput, (size_t)noutput,
                          (size_t)maxhid, (size_t)maxwidth, &rand_rng);
        if (write_pop(popfile, randpop, (size_t)pop) == 0
            && evaluate(evalcmd, popfile, resfile, rtop, 1, &rne, &rbest) == 0
            && (!have_rand || rbest < rand_best)) {
            rand_best = rbest;
            snprintf(rand_topo, sizeof rand_topo, "%s", rtop[0]);
            have_rand = 1;
        }

        printf("gen %3ld   GA best=%.6g (%s) rate=%.2f   RAND best=%.6g (%s)\n",
               gen, ga_best, ga_topo, (double)ga_rate, rand_best, rand_topo);

        /* error control: note the first generation each method reaches the target
         * (the Error argument of self_modifying_predict). Recording both lets one
         * run measure the time-to-target race. Stop once both are good enough (or
         * at the -G safety cap). */
        if (target > 0.0) {
            if (!ga_hit && ga_best <= target) { ga_hit = 1; ga_hit_gen = gen; }
            if (!rand_hit && rand_best <= target) { rand_hit = 1; rand_hit_gen = gen; }
            if (ga_hit && rand_hit)
                break;
        }
    }

    if (target > 0.0) {
        /* one machine-readable line per run for the multi-seed benchmark:
         * evals-to-target = gens-to-target * pop; 0 gens means never reached. */
        printf("\nTARGET error=%g seed=%ld ga_gens=%ld ga_evals=%ld "
               "rand_gens=%ld rand_evals=%ld\n",
               target, seed,
               ga_hit ? ga_hit_gen : 0, ga_hit ? ga_hit_gen * pop : 0,
               rand_hit ? rand_hit_gen : 0, rand_hit ? rand_hit_gen * pop : 0);
    }
    printf("\nfinal:  GA %.6g (%s) rate=%.2f  vs  RAND %.6g (%s)   "
           "[%ld evaluations each]\n",
           ga_best, ga_topo, (double)ga_rate, rand_best, rand_topo, total_evals);

    remove(popfile);
    remove(resfile);
    return 0;
}

#endif /* !UNIT_TEST */
