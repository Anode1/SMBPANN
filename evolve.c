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
        "          [-L maxhidden] [-W maxwidth]\n"
        "\n"
        "Evolves network topologies and races the GA against random search.\n"
        "Fitness comes from scripts/evaluate.sh (set COMMON for a dataset; with\n"
        "none it trains XOR, so use -i 2 -o 1).\n", prog);
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
 * then read the best fitness and the top-K topologies (elite). The leaderboard
 * is sorted best-first, so the first line is the best and the first K are the
 * elite. Returns 0 (and *NELITE >= 1), or -1 if nothing evaluated. */
static int evaluate(const char *evalcmd, const char *popfile, const char *resfile,
                    Genome *elite, size_t k, size_t *nelite, double *best)
{
    char  cmd[1024];
    char  line[512];
    FILE *f;
    size_t got = 0;

    *best = 0.0;
    snprintf(cmd, sizeof cmd, "%s %s > %s 2>/dev/null", evalcmd, popfile, resfile);
    if (system(cmd) == -1)
        return -1;
    f = fopen(resfile, "r");
    if (f == NULL)
        return -1;
    while (fgets(line, sizeof line, f) != NULL) {
        char       *tp = strstr(line, "topology=");
        const char *fp = strstr(line, "fitness=");
        if (tp == NULL || fp == NULL)
            continue;
        if (got == 0)
            *best = atof(fp + 8);            /* "fitness=" is 8 chars; best first */
        if (got < k) {
            char topo[256];
            if (sscanf(tp + 9, "%255[0-9,]", topo) == 1   /* "topology=" is 9 */
                && genome_parse(&elite[got], topo) == 0)
                got++;
        }
    }
    fclose(f);
    *nelite = got;
    return (got > 0) ? 0 : -1;
}

int main(int argc, char **argv)
{
    long   ninput = 0, noutput = 0;
    long   pop = 8, gens = 10, elite = 2, seed = 1, maxhid = 3, maxwidth = 16;
    int    c;
    long   gen, total_evals = 0;
    size_t i;

    const char *evalcmd;
    char        popfile[64], resfile[64];
    Rng         ga_rng, rand_rng;
    Genome      gapop[EV_MAX_POP], randpop[EV_MAX_POP], eliteg[EV_MAX_POP];
    double      ga_best = 0.0, rand_best = 0.0;
    char        ga_topo[256] = "", rand_topo[256] = "";
    int         have_ga = 0, have_rand = 0;

    while ((c = getopt(argc, argv, "i:o:P:G:k:s:L:W:h")) != -1) {
        switch (c) {
        case 'i': ninput   = atol(optarg); break;
        case 'o': noutput  = atol(optarg); break;
        case 'P': pop      = atol(optarg); break;
        case 'G': gens     = atol(optarg); break;
        case 'k': elite    = atol(optarg); break;
        case 's': seed     = atol(optarg); break;
        case 'L': maxhid   = atol(optarg); break;
        case 'W': maxwidth = atol(optarg); break;
        case 'h': usage(argv[0]); return 0;
        default:  usage(argv[0]); return 2;
        }
    }
    if (ninput <= 0 || noutput <= 0) {
        fprintf(stderr, "evolve: -i and -o are required and positive\n");
        return 2;
    }
    if (pop < 2 || pop > EV_MAX_POP || elite < 1 || elite >= pop
        || gens < 1 || maxwidth < 1) {
        fprintf(stderr, "evolve: need 2 <= pop <= %d, 1 <= elite < pop, "
                        "gens >= 1, maxwidth >= 1\n", EV_MAX_POP);
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

    for (i = 0; i < (size_t)pop; i++)
        genome_random(&gapop[i], (size_t)ninput, (size_t)noutput,
                      (size_t)maxhid, (size_t)maxwidth, &ga_rng);

    printf("evolving %ld topologies over %ld generations "
           "(elite %ld, seed %ld)\n", pop, gens, elite, seed);

    for (gen = 1; gen <= gens; gen++) {
        size_t ne;
        double gbest, rbest;

        /* --- the genetic search --- */
        if (write_pop(popfile, gapop, (size_t)pop) != 0
            || evaluate(evalcmd, popfile, resfile, eliteg, (size_t)elite,
                        &ne, &gbest) != 0) {
            fprintf(stderr, "evolve: evaluation failed (is %s runnable?)\n",
                    evalcmd);
            return 1;
        }
        total_evals += pop;
        if (!have_ga || gbest < ga_best) {
            ga_best = gbest;
            genome_format(&eliteg[0], ga_topo, sizeof ga_topo);
            have_ga = 1;
        }
        /* next generation: keep the elite, refill with mutated elite */
        for (i = 0; i < ne; i++)
            gapop[i] = eliteg[i];
        for (i = ne; i < (size_t)pop; i++) {
            gapop[i] = eliteg[below_pop(&ga_rng, ne)];
            genome_mutate(&gapop[i], (size_t)maxhid, (size_t)maxwidth, &ga_rng);
        }

        /* --- matched-compute random control: pop fresh random genomes --- */
        for (i = 0; i < (size_t)pop; i++)
            genome_random(&randpop[i], (size_t)ninput, (size_t)noutput,
                          (size_t)maxhid, (size_t)maxwidth, &rand_rng);
        if (write_pop(popfile, randpop, (size_t)pop) == 0) {
            Genome rbestg[EV_MAX_POP];
            size_t rne;
            if (evaluate(evalcmd, popfile, resfile, rbestg, 1, &rne, &rbest) == 0
                && (!have_rand || rbest < rand_best)) {
                rand_best = rbest;
                genome_format(&rbestg[0], rand_topo, sizeof rand_topo);
                have_rand = 1;
            }
        }

        printf("gen %3ld   GA best=%.6g (%s)   RAND best=%.6g (%s)\n",
               gen, ga_best, ga_topo, rand_best, rand_topo);
    }

    printf("\nfinal:  GA %.6g (%s)  vs  RAND %.6g (%s)   "
           "[%ld evaluations each]\n",
           ga_best, ga_topo, rand_best, rand_topo, total_evals);

    remove(popfile);
    remove(resfile);
    return 0;
}

#endif /* !UNIT_TEST */
