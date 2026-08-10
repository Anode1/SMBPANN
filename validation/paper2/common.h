/* common.h -- shared leaves for paper-2 (and later) probes. C99, header-only, no dependencies.
 *
 * THE RULE THIS HEADER OBEYS, and the only one that matters:
 *
 *     SHARE WHAT CANNOT CHANGE A NUMBER. DUPLICATE WHAT CAN.
 *
 * Nothing in here ever calls into a probe, and nothing in here draws a random number on its own. It
 * provides the generator; the probe decides when to turn the crank. So moving a probe onto this
 * header cannot alter the order of RNG calls, and therefore cannot alter a single result. That is
 * checkable rather than asserted: emerge_rewire2.c must reproduce emerge_rewire.c and the archived
 * per-seed data in ../../scratch_rewire_*.out exactly.
 *
 * WHAT LIVES HERE (number-neutral):
 *     the two RNG streams, env-knob parsing, run-header and table printing, the RAW row,
 *     and the convergence report over a learning curve.
 *
 * WHAT DOES NOT, EVER (these ARE the experiment, and a shared version would hide the thing under test):
 *     the task, the genome, its mutation operators, the inner learner, and the fitness function.
 *     A probe that cannot show you its genome and its objective in its own file is not a probe.
 *
 * Everything here is `static inline`: a probe that uses only part of the header (emerge_relax runs no
 * search, so it needs the RNG and the env knobs but none of the protocol reporters) must still compile
 * warning-clean. Behaviour is unchanged, and the oracle was rerun to prove it.
 *
 * NO INDIRECTION ON PURPOSE. There are no function pointers and no inversion of control. The probe
 * owns main() and owns its search loop; this header is a box of leaves it calls. That is why a probe
 * still reads top to bottom, and why a bug here cannot silently change what a probe means -- only
 * what it prints.
 *
 * SCALING TO PAPER 3, 4. Copy the directory, not this file: validation/paper3/ includes
 * ../paper2/common.h if the leaves still fit, or grows its own if the reporting changes. The Makefile
 * needs no edit per probe -- it globs the paperN directory -- so a new experiment is one new file.
 */
#ifndef SMB_PAPER_COMMON_H
#define SMB_PAPER_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* -- Two RNG streams. SEARCH decisions and WEIGHT init are kept apart so that retraining a network
 * never perturbs the evolutionary trajectory. Identical xorshift to every probe in this repo, so a
 * probe moved onto this header reproduces its own history. This header calls neither. -- */
static uint32_t smb_rs = 1u;
static inline uint32_t r32(void){ uint32_t x=smb_rs; x^=x<<13; x^=x>>17; x^=x<<5; smb_rs=x; return x; }
static inline void     rseed(uint32_t s){ smb_rs = s?s:1u; }
static inline double   runif(void){ return (double)r32()/4294967296.0*2.0-1.0; }
static inline double   rprob(void){ return (double)r32()/4294967296.0; }

static uint32_t smb_wr = 1u;
static inline uint32_t wr32(void){ uint32_t x=smb_wr; x^=x<<13; x^=x>>17; x^=x<<5; smb_wr=x; return x; }
static inline void     wseed(uint32_t s){ smb_wr = s?s:1u; }
static inline double   wunif(void){ return (double)wr32()/4294967296.0*2.0-1.0; }

/* -- Knobs. PROTOCOL item 3: a constant that is not an env knob is a constant nobody will ever sweep,
 * and an unswept constant has already decided one result in this project. -- */
static inline int    envint(const char*k,int d){ const char*e=getenv(k); return e&&*e?atoi(e):d; }
static inline double envdbl(const char*k,double d){ const char*e=getenv(k); return e&&*e?atof(e):d; }
static inline int    envis (const char*k,const char*v){ const char*e=getenv(k); return e && !strcmp(e,v); }

/* -- Reporting. Printing only; no state, no randomness. A probe prints its own run header, because
 * the header is where its knobs and its objective are declared and those differ per experiment. -- */

/* Convergence over a learning curve: how much accuracy the inner learner still gained over the second
 * half of its epoch budget. This decides whether the ENDPOINT is an honest summary of the learner:
 *   converged     -> the endpoint is a sufficient statistic; trajectory fitness adds only noise
 *                    (measured on the locality probe: placement signal-to-noise 1.29 for AULC
 *                     against 1.42 for the endpoint, 16 placements x 24 seeds)
 *   not converged -> the endpoint reports how far TRAINING got, not how good the TOPOLOGY is, and any
 *                    result read off it moves when the budget moves (measured on the depth probe:
 *                    +0.217 accuracy at depth when epochs and data were raised)
 * Same code, opposite answers on two probes, which is why this has to be measured per probe. */
static inline double smb_curve_gain(const double *curve, int n)
{ return (n>=2) ? curve[n-1] - curve[n/2] : 0.0; }

static inline void smb_convergence_report(double mean_gain)
{
    printf("PROTOCOL  convergence   accuracy gained over the second half of the budget: %+.4f\n", mean_gain);
    printf("PROTOCOL                -> %s\n", fabs(mean_gain) < 0.01
        ? "CONVERGED: endpoint summarises the learner; trajectory fitness would add only noise"
        : "NOT CONVERGED: endpoint reports how far TRAINING got, not how good the TOPOLOGY is");
}

/* PROTOCOL item 1, the printing half. The probe builds the genomes and computes the span, because
 * only the probe knows what its outcome axes are; this just says it the same way every time. */
static inline int smb_sensitivity_report(const char *axis, double lo, double hi, double elo, double ehi)
{
    printf("PROTOCOL  sensitivity %-10s fitness %.4f..%.4f (span %.4f)  energy %.6f..%.6f\n",
           axis, lo, hi, hi-lo, elo, ehi);
    if(hi-lo < 1e-6){
        printf("PROTOCOL  FAIL: fitness is FLAT along %s. Nothing the search does could be credited\n"
               "          for a %s outcome. Fix the objective, not the operator.\n", axis, axis);
        return 0;
    }
    return 1;
}

#endif /* SMB_PAPER_COMMON_H */
