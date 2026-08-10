/* emerge_rewire2.c -- rewiring vs add/remove, version 2.  Paper 2 (smbpann2).
 * Self-contained C99, no dependencies, no indirection: read it top to bottom.
 * Build: make emerge_rewire2
 *
 * THE SHAPE OF THE EXPERIMENT, which is the shape of this file:
 *
 *      TASK          a labelled dataset drawn from a seed
 *      GENOME        a topology: per-unit window [start, start+w), plus a weight-sharing gene
 *      INNER LEARNER backprop trains the net that a genome describes, over a FIXED epoch budget
 *      FITNESS       the outer objective: how good is the learner that genome produced
 *      OUTER SEARCH  a GA over genomes, whose fitness is the line above
 *      PROTOCOL      checks that must pass before any of it is allowed to report a number
 *
 * The two loops are the point. The inner loop trains weights and knows nothing about evolution. The
 * outer loop evolves topologies and knows nothing about backprop. FITNESS is the only connector, and
 * it is one named function so that it can be read, checked, and changed in one place -- in version 1
 * of this line the objective was an expression buried in the middle of a probe, which is how a step
 * function nobody had looked at ended up deciding a result.
 *
 * ---------------------------------------------------------------------------------------------
 * VERSION 2 CHANGES, stated so they are not confused with emerge_rewire.c (version 1, unchanged).
 *
 *  1. LAYOUT. Same algorithm, reorganised into the six banners above. Nothing about the search or
 *     the arms changed, and the RNG call order is identical, so version 2 REPRODUCES version 1's
 *     numbers exactly. That is this file's acceptance test, against the archived per-seed data in
 *     ../../scratch_rewire_*.out (commit 70e6ea0). If it ever stops reproducing them, this file is
 *     wrong, not the archive.
 *
 *  2. THE INNER LEARNER NOW RECORDS A LEARNING CURVE, not just its endpoint. Accuracy is measured
 *     every CHECK epochs during training. The returned value is still the final point, so (1) holds.
 *
 *  3. NEW PROTOCOL CHECK: CONVERGENCE. The run reports whether the inner learner has converged
 *     inside its epoch budget, by comparing accuracy at half budget with accuracy at full budget.
 *     This decides whether the endpoint is an honest summary of the learner:
 *        converged     -> the endpoint is a sufficient statistic; trajectory fitness adds only noise
 *        not converged -> the endpoint reports how far TRAINING got, not how good the ARCHITECTURE is,
 *                         and any result conditioned on it moves when the budget moves
 *     This exists because both failure modes were measured. On this probe the learner converges, and
 *     area-under-the-curve fitness came out strictly worse than the endpoint (placement signal-to-noise
 *     1.29 against 1.42, by a 16-placement x 24-seed variance decomposition). On emerge_compose it does
 *     NOT converge, and accuracy at depth rose +0.217 when the budget was raised. Same code, opposite
 *     answers, so the check has to be per-probe and automatic rather than remembered.
 *
 *  4. FITNESS IS SELECTABLE, so the finding above can be re-tested on a probe where the learner does
 *     not converge, without rewriting anything: FITNESS=final (default) or FITNESS=aulc. It is a plain
 *     if, not a function pointer. Default reproduces version 1.
 *
 *  5. Every constant is an env knob and every knob is printed in the run header (PROTOCOL item 3).
 *     Version 1 already did this for the search; version 2 extends it to the learner (EPOCHS, CHECK).
 * ---------------------------------------------------------------------------------------------
 *
 * THE EXPERIMENT ITSELF (unchanged from version 1). Energy charges free parameters: with sharing on
 * it is the number of distinct tied weights, i.e. the max window width; with sharing off it is the
 * total connection count. NEITHER READS start. So moving a window at fixed width is exactly
 * energy-neutral, and the equal-energy genomes form a connected neutral network. Add/remove cannot
 * travel it; rewiring can. Three arms, identical width mutation, differing only in placement:
 *      0 no-rewire     start never mutates                    (the add/remove-only control)
 *      1 slide-1       start += +-1          at rate PSLIDE   (neighbour-only; O(d^2) to move d)
 *      2 rewire-rand   start := uniform valid at rate PSLIDE  (no spatial prior; O(1) to move d)
 * Arm 1 imposes a spatial neighbourhood, which is the prior under test, so arm 2 is the primary
 * result and arm 1 measures what the prior buys.
 *
 * env: SEEDS GENS EPOCHS CHECK LAMBDA PSLIDE PGROW PSHARE FITNESS RAW
 */
#include "common.h"     /* RNG, env knobs, reporting: the leaves that cannot change a number */

#define N   12                 /* inputs                                   */
#define K   3                  /* true filter width                        */
#define H   (N - K + 1)        /* hidden units = 10                        */
#define NTR   64
#define NVAL  300
#define NTE   1000
#define MAXEP 400              /* ceiling for EPOCHS, sizes the curve array */
#define LR    0.1
#define POP   24

/* -- knobs (PROTOCOL item 3: no hidden constants) -- */
static int    g_seeds  = 30, g_gens = 150, g_epochs = 50, g_check = 5, g_raw = 0, g_arm = 2;
static int    g_elite  = 4;   /* ELITE: selection pressure. Low = fast convergence, and a
                                 converged population gives recombination nothing to work on. */
static int    g_aulc   = 0;                /* FITNESS=aulc  */
static double g_lambda = 1.0, g_pslide = 0.15, g_pgrow = 0.10, g_pshare = 0.05;
/* RNG (two streams), envint/envdbl and the reporting helpers come from common.h. Everything below
 * this line is what makes THIS probe a different experiment from any other: task, genome, mutation
 * operators, inner learner, fitness. */

/* ============================== TASK ============================== */
/* Label = sign of the summed response of a fixed K-tap filter slid over the input. A compact shared
 * K-wide kernel tiling the input is therefore the right answer, and the search is not told so. */
static double wstar[K];
static double Xtr[NTR][N], Xval[NVAL][N], Xte[NTE][N];
static int    ytr[NTR], yval[NVAL], yte[NTE];

static int label_of(const double *x)
{ int p,k; double s=0; for(p=0;p+K<=N;p++){ double a=0; for(k=0;k<K;k++) a+=wstar[k]*x[p+k]; s+=tanh(a);} return s>0; }
static void gen(double X[][N],int*y,int n)
{ int s,i; for(s=0;s<n;s++){ for(i=0;i<N;i++) X[s][i]=runif(); y[s]=label_of(X[s]); } }
static void new_task(uint32_t seed)
{ int k; rseed(seed); for(k=0;k<K;k++) wstar[k]=runif(); gen(Xtr,ytr,NTR); gen(Xval,yval,NVAL); gen(Xte,yte,NTE); }

/* ============================== GENOME ============================== */
/* A topology, nothing more: where each hidden unit looks, how wide, and whether weights are tied. */
typedef struct { int start[H], w[H]; int shared; } Indiv;

static int    maxwidth (const Indiv*g){ int j,m=0; for(j=0;j<H;j++) if(g->w[j]>m) m=g->w[j]; return m; }
static double meanwidth(const Indiv*g){ int j; double s=0; for(j=0;j<H;j++) s+=g->w[j]; return s/H; }
/* THE ENERGY TERM, and the reason this probe exists: it is a function of WIDTHS only. start[] does
 * not appear, so placement is free of charge and the equal-energy set is a neutral network. */
static double energy(const Indiv*g)
{ int j; if(g->shared) return (double)maxwidth(g)/(H*N);
  { long a=0; for(j=0;j<H;j++) a+=g->w[j]; return (double)a/(H*N); } }
static double coverage(const Indiv*g)
{ char c[N]; int j,t,i,cov=0; memset(c,0,sizeof c);
  for(j=0;j<H;j++) for(t=0;t<g->w[j];t++){ int i2=g->start[j]+t; if(i2>=0&&i2<N) c[i2]=1; }
  for(i=0;i<N;i++) cov+=c[i];
  return (double)cov/N; }
static void legal(Indiv*g)
{ int j; for(j=0;j<H;j++){ if(g->w[j]<1) g->w[j]=1; if(g->w[j]>N) g->w[j]=N;
    if(g->start[j]<0) g->start[j]=0;
    if(g->start[j]+g->w[j]>N) g->start[j]=N-g->w[j]; } }
/* SEED DIRECTION -- a constant no probe in this repo has ever varied, and the one that decides what
 * the question even means.
 *   dense   (default, and what every emerge_* probe does): w=N, start=0 -> coverage 1.000, max-w 12.
 *           Both are at their MAXIMUM, so the search can only spend coverage as it compacts, and any
 *           manipulation that slows the search inflates the coverage number. "Does the tiling emerge"
 *           cannot be asked here: the tiling is there at generation zero.
 *   minimal: w=1, start=0 -> coverage 0.083, max-w 1. Energy is already at its floor, so the energy
 *           term has nothing left to win and only ACCURACY can drive growth. Coverage must be earned.
 * Both seeds put every unit at start=0, so they differ in width alone and the contrast is clean. */
static int g_minimal = 0;
static void seed_individual(Indiv*g)
{ int j; for(j=0;j<H;j++){ g->start[j]=0; g->w[j]= g_minimal?1:N; } g->shared=0; }

/* THE ARM lives here and nowhere else. Width mutation is identical in all three; only placement
 * differs, so any arm difference is a placement effect by construction. */
static void mutate(Indiv*g)
{
    int j;
    for(j=0;j<H;j++){
        if(rprob()<g_pgrow) g->w[j]+=1; else if(rprob()<0.5) g->w[j]-=1;
        if(g->w[j]<1) g->w[j]=1;
        if(g->w[j]>N) g->w[j]=N;
        { int mode = (g_arm>=3) ? 2 : g_arm;   /* recombination arms inherit arm 2's placement operator,
                                                  so any arm 3..5 effect is recombination and nothing else */
          if(mode==1){ if(rprob()<g_pslide) g->start[j] += (rprob()<0.5)?-1:1; }
          else if(mode==2){ if(rprob()<g_pslide) g->start[j] = (int)(r32()%(uint32_t)(N-g->w[j]+1)); } }
    }
    if(rprob()<g_pshare) g->shared^=1;
    legal(g);
}

/* ========================== INNER LEARNER ========================== */
/* Backprop on the net a genome describes. Knows nothing about evolution. Trains for a FIXED epoch
 * budget and records accuracy every CHECK epochs (version 2 change 2); returns the final point, so
 * the value is identical to version 1's single end-of-training evaluation. */
static double g_curve[MAXEP+1]; static int g_ncurve;

static double train_eval(const Indiv*g, uint32_t seed, int on_test)
{
    static double W[H][N], wsh[N], bh[H], v[H];
    double bo=0, h[H]; int j,t,e,s, mw=maxwidth(g), ns = on_test?NTE:NVAL;
    double (*Xe)[N] = on_test?Xte:Xval; int *ye = on_test?yte:yval;

    wseed(seed); g_ncurve=0;
    for(t=0;t<N;t++) wsh[t]=0.1*wunif();
    for(j=0;j<H;j++){ for(t=0;t<N;t++) W[j][t]=0.1*wunif(); bh[j]=0; v[j]=0.1*wunif(); }

    for(e=0;e<g_epochs;e++){
        for(s=0;s<NTR;s++){                                   /* one epoch of SGD */
            const double *x=Xtr[s]; double opre=bo,o,dout, dwsh[N];
            for(j=0;j<H;j++){ double pre=bh[j];
                for(t=0;t<g->w[j];t++) pre += (g->shared?wsh[t]:W[j][t])*x[g->start[j]+t];
                h[j]=tanh(pre); opre+=v[j]*h[j]; }
            o=1.0/(1.0+exp(-opre)); dout=(o-ytr[s])*o*(1.0-o);
            if(g->shared){ for(t=0;t<mw;t++) dwsh[t]=0; }
            for(j=0;j<H;j++){ double dpre=dout*v[j]*(1.0-h[j]*h[j]); v[j]-=LR*dout*h[j]; bh[j]-=LR*dpre;
                for(t=0;t<g->w[j];t++){ double xx=x[g->start[j]+t];
                    if(g->shared) dwsh[t]+=dpre*xx; else W[j][t]-=LR*dpre*xx; } }
            bo-=LR*dout;
            if(g->shared) for(t=0;t<mw;t++) wsh[t]-=LR*dwsh[t];
        }
        if((e+1)%g_check==0 || e==g_epochs-1){                /* learning curve, no RNG consumed */
            int c=0;
            for(s=0;s<ns;s++){ const double *x=Xe[s]; double opre=bo,o;
                for(j=0;j<H;j++){ double pre=bh[j];
                    for(t=0;t<g->w[j];t++) pre+=(g->shared?wsh[t]:W[j][t])*x[g->start[j]+t];
                    opre+=v[j]*tanh(pre); }
                o=1.0/(1.0+exp(-opre)); if((o>0.5)==(ye[s]==1)) c++; }
            if(g_ncurve<=MAXEP) g_curve[g_ncurve++]=(double)c/ns;
        }
    }
    return g_curve[g_ncurve-1];
}

/* ============================== FITNESS ============================== */
/* THE CONNECTOR between the two loops: the outer fitness of a topology is the quality of the inner
 * learner it produced, minus what it cost. One function, one place to read, one place to change.
 * Continuous by design: version 1 of this line used (acc>=target ? 2-energy : acc), and above the
 * target that expression does not read placement at all, so the endgame was neutral drift. */
static double fitness(const Indiv*g, double acc)
{
    double quality = acc;
    if(g_aulc){                                   /* trajectory instead of endpoint (change 4) */
        int i; double s=0;
        for(i=0;i<g_ncurve;i++) s+=g_curve[i];
        quality = s/g_ncurve;
    }
    return quality - g_lambda*energy(g);
}

/* ========================= PROTOCOL CHECKS ========================= */
/* Item 1: does the fitness move along the axes whose outcome we intend to report? If not, nothing
 * the search does can be credited for that outcome, and the run must not produce numbers. */
static void build_axis(Indiv*g, int axis, int k)
{
    int j;
    if(axis==0){                                  /* PLACEMENT: widths pinned, energy constant */
        g->shared=1; for(j=0;j<H;j++) g->w[j]=K;
        if(k==0)      for(j=0;j<H;j++) g->start[j]=0;
        else if(k==1) for(j=0;j<H;j++) g->start[j]=N-K;
        else if(k==2) for(j=0;j<H;j++) g->start[j]=(j%2)?N-K:0;
        else          for(j=0;j<H;j++) g->start[j]=(j*(k-2))%(N-K+1);
    } else {                                      /* WIDTH: starts pinned                      */
        g->shared=1; for(j=0;j<H;j++){ g->w[j]=k+1; g->start[j]=0; }
    }
    legal(g);
}

static int protocol_checks(int seeds)
{
    static const char *axname[2] = {"placement","width"};
    int axis, k, sd, ok=1, nk[2]; double half_gap=0, converged_n=0;
    nk[0]=8; nk[1]=N;

    for(axis=0; axis<2; axis++){
        double lo=1e300, hi=-1e300, elo=1e300, ehi=-1e300;
        for(sd=1; sd<=seeds; sd++){
            new_task((uint32_t)(sd*911u+1u));
            for(k=0;k<nk[axis];k++){
                Indiv g; double f,e;
                build_axis(&g,axis,k);
                f = fitness(&g, train_eval(&g,(uint32_t)(sd*7u+k*131u+1u),0));
                e = energy(&g);
                if(f<lo) lo=f;
                if(f>hi) hi=f;
                if(e<elo) elo=e;
                if(e>ehi) ehi=e;
                if(axis==0 && g_ncurve>=2){        /* convergence, change 3 */
                    half_gap += smb_curve_gain(g_curve, g_ncurve); converged_n++;
                }
            }
        }
        if(!smb_sensitivity_report(axname[axis], lo, hi, elo, ehi)) ok=0;
    }
    half_gap /= (converged_n>0?converged_n:1);
    smb_convergence_report(half_gap);

    /* TARGET CHECK. The probe claims to be looking for a convolution: shared weights, width K, windows
     * tiling the input. Is that genome even an OPTIMUM of the objective? Score it by hand, and score
     * one deliberately worse than it (width K+1). If the search reliably beats the hand-built target,
     * the target is not what the objective wants, and no operator, budget or seed will produce it.
     * Same move as objcheck.c, applied to the goal rather than to the outcome axis. */
    { double fi=0, fw=0; int sd2, n=0;
      for(sd2=1; sd2<=seeds; sd2++){
        Indiv g; int j;
        new_task((uint32_t)(sd2*911u+1u));
        g.shared=1; for(j=0;j<H;j++){ g.w[j]=K; g.start[j]=j; } legal(&g);
        fi += fitness(&g, train_eval(&g,(uint32_t)(sd2*7u+1u),0));
        g.shared=1; for(j=0;j<H;j++){ g.w[j]=K+1; g.start[j]=j; } legal(&g);
        fw += fitness(&g, train_eval(&g,(uint32_t)(sd2*7u+1u),0));
        n++; }
      printf("PROTOCOL  target        ideal convolution (w=K=%d, tiled): fitness %.4f   w=K+1 variant: %.4f\n",
             K, fi/n, fw/n);
      printf("PROTOCOL                -> compare against the arms below; if the search beats the ideal,\n");
      printf("PROTOCOL                   the objective does not have the convolution as its optimum.\n"); }
    return ok;
}

/* ========================= RECOMBINATION ========================= */
/* Topology crossover over STRUCTURES, not over an encoding string. A genome here is a set of units,
 * each a whole little structure (where it looks, how wide), so recombination exchanges units.
 *
 * THE PERMUTATION PROBLEM, and why there are three arms rather than one. Units are interchangeable:
 * any unit may look anywhere, so two parents holding the SAME solution with their units in different
 * order are different genomes. Recombining them by index then mixes incompatible conventions and
 * produces a child worse than either parent. This is the classic competing-conventions failure, and
 * it is a live candidate for why four topology-crossover operators produced a null in paper 1.
 * CANONICALISING first -- sorting units by where they look -- makes "unit j" mean the same thing in
 * both parents and costs nothing. Arm 4 is the same operator WITHOUT that step, so the pair measures
 * whether the earlier null was operator design rather than the space. That is the question paper 1
 * says it most wants answered.
 *
 * No RNG is drawn here except by the arm that needs it, and never for arms 0..2. */
static void canon(Indiv *g)                    /* sort units by start, then width: a canonical order */
{
    int i,j;
    for(i=1;i<H;i++){
        int st=g->start[i], w=g->w[i];
        for(j=i-1; j>=0 && (g->start[j]>st || (g->start[j]==st && g->w[j]>w)); j--){
            g->start[j+1]=g->start[j]; g->w[j+1]=g->w[j];
        }
        g->start[j+1]=st; g->w[j+1]=w;
    }
}
static int ga_recombines(void){ return g_arm>=3; }

static void ga_cross(Indiv *child, const Indiv *other)
{
    Indiv o = *other; int j;
    if(g_arm!=4){ canon(child); canon(&o); }        /* arm 4 deliberately skips canonicalisation */
    if(g_arm==5){                                   /* structural: copy a contiguous BLOCK of units */
        int i0 = (int)(r32()%(uint32_t)H), len = 1 + (int)(r32()%(uint32_t)H), i;
        for(i=0;i<len;i++){ int k=(i0+i)%H; child->start[k]=o.start[k]; child->w[k]=o.w[k]; }
    } else {                                        /* uniform: each unit from either parent */
        for(j=0;j<H;j++) if(rprob()<0.5){ child->start[j]=o.start[j]; child->w[j]=o.w[j]; }
    }
    if(rprob()<0.5) child->shared = o.shared;
    legal(child);
}

/* =========================== OUTER SEARCH =========================== */
/* The GA itself lives in ga.h, included at the bottom of this file once the symbols it needs exist.
 * What stays here is the part that is specific to THIS experiment: what a finished run reports, and
 * what its RAW row carries. Neither draws a random number, so neither can move a result. */
/* Mean pairwise genome distance in the final population, normalised to [0,1]. This is the direct
 * measurement of the hypothesis that recombination has nothing to recombine: if selection has
 * collapsed the population onto one genome, crossover is copying a solution onto itself and a null
 * result says nothing about linkage. Measure it rather than infer it from arm effects. */
static double diversity(const Indiv *pop)
{
    int p,q,j; double s=0; long n=0;
    for(p=0;p<POP;p++) for(q=p+1;q<POP;q++){
        double d=0;
        for(j=0;j<H;j++) d += fabs((double)(pop[p].start[j]-pop[q].start[j]))
                            +  fabs((double)(pop[p].w[j]    -pop[q].w[j]));
        s += d/(H*2.0*N); n++;
    }
    return n? s/n : 0.0;
}
static void ga_summarize(const Indiv *pop, const double *fit, int best, double test, double *out)
{
    double sh=0,mw=0,mx=0,cv=0,en=0; int p;
    (void)fit; (void)best;
    for(p=0;p<POP;p++){ sh+=pop[p].shared; mw+=meanwidth(&pop[p]); mx+=maxwidth(&pop[p]);
                        cv+=coverage(&pop[p]); en+=energy(&pop[p]); }
    out[0]=sh/POP; out[1]=mw/POP; out[2]=mx/POP; out[3]=cv/POP;
    out[4]=test;                                   /* held-out accuracy, from ga.h */
    out[5]=en/POP;
    out[6]=diversity(pop);
}
static void ga_raw_row(const Indiv *pop, int best, int sd, double test)
{
    printf("RAW %d %d %d %.4f %.4f %.4f %.4f %.6f %.4f\n", g_arm, g_gens, sd,
           coverage(&pop[best]), meanwidth(&pop[best]), (double)maxwidth(&pop[best]),
           (double)pop[best].shared, energy(&pop[best]), test);
}

#include "ga.h"       /* the outer search: binds to the symbols defined above, at compile time */

/* =============================== MAIN =============================== */
int main(void)
{
    static const char *armname[6] = {"0 no-rewire","1 slide-1","2 rewire-rand",
                                     "3 xover-canon","4 xover-raw","5 xover-block"};
    int sd,k;

    g_seeds=envint("SEEDS",30); g_gens=envint("GENS",150);
    g_epochs=envint("EPOCHS",50); g_check=envint("CHECK",5);
    g_lambda=envdbl("LAMBDA",1.0); g_pslide=envdbl("PSLIDE",0.15);
    g_pgrow=envdbl("PGROW",0.10);  g_pshare=envdbl("PSHARE",0.05);
    g_raw=envint("RAW",0); g_elite=envint("ELITE",4); g_minimal=envis("SEED","minimal");
    if(g_elite<1) g_elite=1;
    if(g_elite>POP-1) g_elite=POP-1;
    g_aulc = envis("FITNESS","aulc");
    if(g_epochs>MAXEP) g_epochs=MAXEP;
    if(g_check<1) g_check=1;

    printf("emerge_rewire2 -- rewiring vs add/remove (paper 2, version 2)\n");
    printf("inner: backprop, %d epochs, curve every %d.   outer: GA, POP=%d ELITE=%d, %d gens x %d seeds\n",
           g_epochs, g_check, POP, g_elite, g_gens, g_seeds);
    printf("seed = %s (%s)\n", g_minimal?"minimal":"dense",
           g_minimal?"w=1, coverage must be earned":"w=N, coverage 1.0 at generation zero");
    printf("fitness = %s - %.3f*energy    PSLIDE=%.3f PGROW=%.3f PSHARE=%.3f%s\n\n",
           g_aulc?"AULC":"final accuracy", g_lambda, g_pslide, g_pgrow, g_pshare,
           g_gens==0 ? "\n(GENS=0: NO-EVOLUTION CONTROL -- any arm difference here is not selection)" : "");

    if(!protocol_checks(g_seeds>=12?12:g_seeds)) return 2;

    printf("\n  %-14s  shared-frac  mean-w  max-w  coverage  energy   test  diversity\n", "arm");
    for(g_arm=0; g_arm<6; g_arm++){
        double o[7]={0,0,0,0,0,0,0}, one[7];
        for(sd=1; sd<=g_seeds; sd++){
            new_task((uint32_t)(sd*131+1));
            run_ga((uint32_t)(sd*7+1), sd, one);
            for(k=0;k<7;k++) o[k]+=one[k];
        }
        for(k=0;k<7;k++) o[k]/=g_seeds;
        printf("  %-14s   %.3f       %.2f    %.2f   %.3f    %.4f   %.3f    %.4f\n",
               armname[g_arm], o[0], o[1], o[2], o[3], o[5], o[4], o[6]);
    }
    printf("\nacceptance test: with default knobs this must reproduce emerge_rewire.c (version 1)\n");
    printf("and the archived scratch_rewire_*.out per-seed data exactly.\n");
    return 0;
}
