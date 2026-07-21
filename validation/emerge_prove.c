/* emerge_prove.c -- does the DIRECTED energy GA reach the compact filter where RANDOM search does not?
 *
 * The grouped-mutation result (emerge_offset.c) showed the compact filter emerges under an energy budget.
 * Two questions remain to PROVE the mechanism rather than illustrate it: (1) is the emergence due to the
 * DIRECTED search, or would random sampling of offset masks find a compact filter just as well? The whole
 * project is haunted by random search being a strong baseline (the crossover study lost to it), so this is
 * the sharp test. (2) Does it hold as the problem SCALES (larger N, more offsets to prune)?
 *
 * Same setup as emerge_offset: shared-weight net, genome = offset mask, grouped per-offset mutation,
 * energy = (#taps + width*span)/(H*N), objective = minimize energy subject to accuracy >= target. Two
 * search arms at MATCHED evaluation budget:
 *   GA     : the directed energy GA (selection, elitism, grouped mutation, annealing).
 *   RANDOM : draw the same number of random offset masks (each with a random number of active offsets,
 *            so sparse and dense are both sampled), train each, keep the best by the same objective.
 * Reports taps / contiguity / on-relevant / test, mean +/- SD over many seeds, at two problem sizes.
 * The compact filter is "proven emergent" if GA reaches it (taps -> K, contiguity high, on-rel high) and
 * beats random by more than the noise; scaling holds if that survives the larger N.
 *
 * evals-to-target: alongside the quality metrics, each arm records how many evaluations it spends before
 * it first reaches a mask that is BOTH accurate (acc>=target) AND compact (taps<=--taptgt). That count is
 * the unbounded efficiency measure: the RND/GA ratio is the extra budget random needs to match the GA, and
 * it should keep growing with N even where the bounded quality gap saturates.
 *
 * Parameterization: every knob is a command-line flag (--seeds, --gens, --pop, ...). Each flag defaults to
 * its UPPERCASE environment variable, then to a compile-time default. So `--seeds 200` == `SEEDS=200 ...`.
 * Run `emerge_prove --help` for the full list. Self-contained C99.  Build: make emerge_prove
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ---- compile-time buffer capacities (memory footprint / task shape; not runtime knobs) ---- */
#define NMAX   32          /* max problem size N */
#define K      3           /* generative filter width */
#define HMAX   (NMAX - K + 1)
#define NOFFMAX (NMAX + HMAX)
#define MAXN   8           /* max number of problem sizes in a sweep */
#define NTR    64          /* train set size */
#define NVAL   300         /* validation set size (search objective) */
#define NTE    1000        /* held-out test set size (reported) */
#define TRACEG 1024        /* max generations buffered in TRACE mode */

/* ---- compile-time DEFAULTS for the runtime hyperparameters below ---- */
#define DEF_TEPOCHS 50
#define DEF_LR      0.1
#define DEF_POP     24     /* also the population array capacity: runtime --pop must be <= DEF_POP */
#define DEF_ELITE   4

static uint32_t rs=1u; static uint32_t r32(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;rs=x;return x;}
static void rseed(uint32_t s){rs=s?s:1u;} static double runif(void){return (double)r32()/4294967296.0*2.0-1.0;}
static uint32_t wr=1u; static uint32_t wr32(void){uint32_t x=wr;x^=x<<13;x^=x>>17;x^=x<<5;wr=x;return x;}
static void wseed(uint32_t s){wr=s?s:1u;} static double wunif(void){return (double)wr32()/4294967296.0*2.0-1.0;}

/* ---- active problem dimensions (set by set_n) ---- */
static int    g_n=12, g_h=10, g_noff=22, g_off0=10, g_rlo=10, g_rhi=12;

/* ================= single source of truth for every runtime parameter =================
 * To add a parameter: add ONE line to PARAMS (scalar/string) or FLAGS (boolean mode). The same
 * table generates the global variable, its env default, its --flag parsing, and its --help row.
 *   PARAMS entry:  X(KIND, var, "flag", "ENV", default, "help")   KIND in {INT, DBL, STR}
 *   FLAGS  entry:  X(var, "flag", "ENV", "help")                  (boolean; presence => true)
 */
#define PARAMS(X) \
  X(INT, g_seeds,   "seeds",   "SEEDS",   40,               "seeds per problem size") \
  X(INT, g_seed0,   "seed0",   "SEED0",   1,                "first seed / parallel-chunk offset") \
  X(STR, g_nlist,   "nlist",   "NLIST",   "12,16,20,24,28", "comma list of N (e.g. 12,16,24)") \
  X(DBL, g_target,  "target",  "TARGET",  0.90,             "accuracy gate") \
  X(DBL, g_width,   "width",   "WIDTH",   1.0,              "span weight in the energy cost") \
  X(INT, g_revals,  "revals",  "REVALS",  1,                "evals per random mask (denoise)") \
  X(INT, g_taptgt,  "taptgt",  "TAPTGT",  2*K,              "solved tap ceiling for evals->target") \
  X(INT, g_gens,    "gens",    "GENS",    150,              "GA generations") \
  X(INT, g_pop,     "pop",     "POP",     DEF_POP,          "GA population (<= capacity)") \
  X(INT, g_elite,   "elite",   "ELITE",   DEF_ELITE,        "GA elite carryover (< pop)") \
  X(DBL, g_flip,    "flip",    "FLIP",    0.06,             "per-offset mutation rate") \
  X(INT, g_tepochs, "tepochs", "TEPOCHS", DEF_TEPOCHS,      "inner training epochs") \
  X(DBL, g_lr,      "lr",      "LR",      DEF_LR,           "inner learning rate") \
  X(DBL, g_nstar,   "nstar",   "NSTAR",   0.40,             "boundary: on-rel threshold for N*") \
  X(STR, g_budgets, "budgets", "BUDGETS", "50,150,400",     "boundary: comma list of gens budgets") \
  X(INT, g_n0,      "n",       "N",       12,               "trace: which N to trace")

#define FLAGS(X) \
  X(g_raw,      "raw",      "RAW",      "one RAW line per seed (feed to --agg)") \
  X(g_agg,      "agg",      "AGG",      "read RAW lines on stdin, print aggregated table") \
  X(g_trace,    "trace",    "TRACE",    "mean per-generation offset activation (--n selects N)") \
  X(g_boundary, "boundary", "BOUNDARY", "phase boundary N* vs budget") \
  X(g_probe,    "probe",    "PROBE",    "quick sanity of the generative filter") \
  X(g_dbg,      "dbg",      "DBG",      "extra per-seed debug lines")

/* generate global declarations from the tables */
#define P_DECL_INT(v,def) static int    v = (def);
#define P_DECL_DBL(v,def) static double v = (def);
#define P_DECL_STR(v,def) static char   v[128] = def;
#define P_DECL(KIND,v,fl,ev,def,help) P_DECL_##KIND(v,def)
PARAMS(P_DECL)
#define F_DECL(v,fl,ev,help) static int v = 0;
FLAGS(F_DECL)

/* env-default loaders (used in main, before parse_args) */
#define P_LOAD_INT(v,ev,def) v = envint(ev,(def));
#define P_LOAD_DBL(v,ev,def) v = envdbl(ev,(def));
#define P_LOAD_STR(v,ev,def) do{ const char*e_=getenv(ev); if(e_&&*e_){ strncpy(v,e_,sizeof(v)-1); v[sizeof(v)-1]=0; } }while(0);
#define P_LOAD(KIND,v,fl,ev,def,help) P_LOAD_##KIND(v,ev,def)
#define F_LOAD(v,fl,ev,help) v = (getenv(ev)!=NULL);

/* argv matchers (used inside parse_args; `a` = flag name, `val` = its value string) */
#define P_TRY_INT(v,fl) else if(!strcmp(a,fl)){ v = atoi(val); }
#define P_TRY_DBL(v,fl) else if(!strcmp(a,fl)){ v = atof(val); }
#define P_TRY_STR(v,fl) else if(!strcmp(a,fl)){ strncpy(v,val,sizeof(v)-1); v[sizeof(v)-1]=0; }
#define P_TRY(KIND,v,fl,ev,def,help) P_TRY_##KIND(v,fl)
#define F_TRY(v,fl,ev,help) else if(!strcmp(a,fl)){ v = 1; }

/* --help rows (print the effective default, i.e. the variable value) */
#define P_HELP_INT(v,fl,ev,help) printf("    --%-8s %-46s (%s, default %d)\n", fl, help, ev, v);
#define P_HELP_DBL(v,fl,ev,help) printf("    --%-8s %-46s (%s, default %g)\n", fl, help, ev, v);
#define P_HELP_STR(v,fl,ev,help) printf("    --%-8s %-46s (%s, default %s)\n", fl, help, ev, v);
#define P_HELP(KIND,v,fl,ev,def,help) P_HELP_##KIND(v,fl,ev,help)
#define F_HELP(v,fl,ev,help) printf("    --%-8s %-46s (%s)\n", fl, help, ev);

static double wstar[K];
static double Xtr[NTR][NMAX], Xval[NVAL][NMAX], Xte[NTE][NMAX];
static int    ytr[NTR], yval[NVAL], yte[NTE];
static double g_tracebuf[TRACEG][NOFFMAX];   /* [gen][offset] summed activation freq across trace runs */

static int    envint(const char*k,int d){const char*e=getenv(k);return e&&*e?atoi(e):d;}
static double envdbl(const char*k,double d){const char*e=getenv(k);return e&&*e?atof(e):d;}

static void set_n(int n){ g_n=n; g_h=n-K+1; g_noff=n+g_h; g_off0=g_h; g_rlo=g_off0; g_rhi=g_off0+K-1; }

static int label_of(const double *x)
{ int p,k; double s=0; for(p=0;p+K<=g_n;p++){ double a=0; for(k=0;k<K;k++) a+=wstar[k]*x[p+k]; s+=tanh(a);} return s>0; }
static void gen(double X[][NMAX],int*y,int n){ int s,i; for(s=0;s<n;s++){ for(i=0;i<g_n;i++) X[s][i]=runif(); y[s]=label_of(X[s]); } }
static void new_task(uint32_t seed){ int k; rseed(seed); for(k=0;k<K;k++) wstar[k]=runif(); gen(Xtr,ytr,NTR); gen(Xval,yval,NVAL); gen(Xte,yte,NTE); }

static void off_stats(char off[NOFFMAX], int *taps, int *span, int *onrel)
{ int o,d=0,lo=g_noff,hi=-1,rel=0; for(o=0;o<g_noff;o++) if(off[o]){ d++; if(o>=g_rlo&&o<=g_rhi) rel++; if(o<lo)lo=o; if(o>hi)hi=o; }
  *taps=d; *span=(hi>=lo)?(hi-lo+1):0; *onrel=rel; }
static double param_energy(char off[NOFFMAX])
{ int d,sp,rel; off_stats(off,&d,&sp,&rel); return (d + g_width*sp)/(g_h*g_n); }

static double run_net(char off[NOFFMAX], uint32_t seed, int on_test)
{
    static double woff[NOFFMAX], bh[HMAX], v[HMAX];
    double bo=0, h[HMAX]; int i,j,e,s,c=0, ns=on_test?NTE:NVAL;
    double (*Xe)[NMAX]=on_test?Xte:Xval; int *ye=on_test?yte:yval;
    wseed(seed);
    for(i=0;i<g_noff;i++) woff[i]=0.1*wunif();
    for(j=0;j<g_h;j++){ bh[j]=0; v[j]=0.1*wunif(); }
    for(e=0;e<g_tepochs;e++) for(s=0;s<NTR;s++){
        const double *x=Xtr[s]; double opre=bo,o,dout; double dwoff[NOFFMAX];
        for(j=0;j<g_h;j++){ double pre=bh[j];
            for(i=0;i<g_n;i++){ int oo=i-j+g_off0; if(oo>=0&&oo<g_noff&&off[oo]) pre+=woff[oo]*x[i]; }
            h[j]=tanh(pre); opre+=v[j]*h[j]; }
        o=1.0/(1.0+exp(-opre)); dout=(o-ytr[s])*o*(1.0-o);
        for(i=0;i<g_noff;i++) dwoff[i]=0;
        for(j=0;j<g_h;j++){ double dpre=dout*v[j]*(1.0-h[j]*h[j]); v[j]-=g_lr*dout*h[j]; bh[j]-=g_lr*dpre;
            for(i=0;i<g_n;i++){ int oo=i-j+g_off0; if(oo>=0&&oo<g_noff&&off[oo]) dwoff[oo]+=dpre*x[i]; } }
        bo-=g_lr*dout;
        for(i=0;i<g_noff;i++) woff[i]-=g_lr*dwoff[i];
    }
    for(s=0;s<ns;s++){ const double *x=Xe[s]; double opre=bo,o;
        for(j=0;j<g_h;j++){ double pre=bh[j]; for(i=0;i<g_n;i++){ int oo=i-j+g_off0; if(oo>=0&&oo<g_noff&&off[oo]) pre+=woff[oo]*x[i]; } opre+=v[j]*tanh(pre); }
        o=1.0/(1.0+exp(-opre)); if((o>0.5)==(ye[s]==1)) c++; }
    return (double)c/ns;
}
static double objective(char off[NOFFMAX], double acc)
{ int d,sp,rel; off_stats(off,&d,&sp,&rel); if(d==0) return acc*0.5;
  return (acc>=g_target) ? (2.0 - param_energy(off)) : acc; }

/* collect metrics of the best offset mask into out += {taps,contig,onrel,test} */
static void tally(char best[NOFFMAX], uint32_t seed, double out[])
{ int taps,span,rel; off_stats(best,&taps,&span,&rel);
  out[0]+=taps; out[1]+= span?(double)taps/span:0.0; out[2]+= taps?(double)rel/taps:0.0;
  out[3]+=run_net(best,seed+999u,1); }

/* evals-to-target: record the (1-based) eval index ec at which a mask is first BOTH accurate and compact
 * (acc>=g_target AND taps<=g_taptgt) -- i.e. reaching the compact accurate filter. *ett stays -1 if the
 * budget is exhausted without reaching it (right-censored). */
static void ett_check(char off[NOFFMAX], double acc, long ec, double *ett)
{ if(*ett<0.0 && acc>=g_target){ int t,sp,rl; off_stats(off,&t,&sp,&rl); if(t<=g_taptgt) *ett=(double)ec; } }

/* GA arm: directed energy GA with grouped mutation. out[0..3]=metrics, out[4]=evals-to-target (-1 if none). */
static void run_ga(uint32_t seed, double out[])
{
    static char pop[DEF_POP][NOFFMAX], nxt[DEF_POP][NOFFMAX], best[NOFFMAX];
    double fit[DEF_POP], facc[DEF_POP]; int idx[DEF_POP]; int g,p,q,o; long ec=0;
    out[4]=-1.0; rseed(seed);
    for(p=0;p<g_pop;p++) for(o=0;o<g_noff;o++) pop[p][o]=(o!=0);   /* offset 0 is a phantom (no edge maps to it); never activate it, else off_stats pins span's lower bound to 0 and kills the contiguity gradient */
    for(p=0;p<g_pop;p++){ facc[p]=run_net(pop[p],(uint32_t)(seed+p*2654435761u+1u),0); fit[p]=objective(pop[p],facc[p]);
        ec++; ett_check(pop[p],facc[p],ec,&out[4]); }
    for(g=0;g<g_gens;g++){
        if(g_trace && g<TRACEG){ int oo,pp;   /* accumulate population offset-activation freq entering gen g (g=0 = dense seed) */
            for(oo=0;oo<g_noff;oo++){ int cnt=0; for(pp=0;pp<g_pop;pp++) cnt+=pop[pp][oo]; g_tracebuf[g][oo]+=(double)cnt/g_pop; } }
        for(p=0;p<g_pop;p++) idx[p]=p;
        for(p=0;p<g_pop;p++) for(q=p+1;q<g_pop;q++) if(fit[idx[q]]>fit[idx[p]]){int t=idx[p];idx[p]=idx[q];idx[q]=t;}
        for(p=0;p<g_elite;p++) memcpy(nxt[p],pop[idx[p]],sizeof nxt[p]);
        for(p=g_elite;p<g_pop;p++){ int a=idx[(int)(r32()%(uint32_t)g_elite)]; memcpy(nxt[p],pop[a],sizeof nxt[p]);
            for(o=1;o<g_noff;o++) if((double)r32()/4294967296.0<g_flip) nxt[p][o]^=1; }
        memcpy(pop,nxt,sizeof pop);
        for(p=0;p<g_pop;p++){ facc[p]=run_net(pop[p],(uint32_t)(seed+(uint32_t)(g*g_pop+p)+7u),0); fit[p]=objective(pop[p],facc[p]);
            ec++; ett_check(pop[p],facc[p],ec,&out[4]); }
    }
    { int bi=0; double bf=-1; for(p=0;p<g_pop;p++) if(fit[p]>bf){bf=fit[p];bi=p;} memcpy(best,pop[bi],sizeof best); }
    tally(best,seed,out);
}
/* RANDOM arm: draw random masks at matched total budget, keep the best by the same objective.
   revals>1 gives each mask r independent evaluations (best-of-r objective) so it gets the same
   denoising against the hard accuracy gate that the GA gets by re-evaluating survivors each
   generation; total evaluations stay = evals, so the budget is still matched (evals/r distinct masks). */
static void run_random(uint32_t seed, int evals, int revals, double out[])
{
    static char cand[NOFFMAX], best[NOFFMAX]; double bestfit=-1; int e,o,rr; long ec=0; double bacc;
    int nmask = (revals>1) ? evals/revals : evals;
    out[4]=-1.0; rseed(seed^0x5bd1e995u);
    for(e=0;e<nmask;e++){
        int k = 1 + (int)(r32()%(uint32_t)(g_noff-1));       /* random density: sparse..dense over the real offsets */
        double f=-1;
        for(o=0;o<g_noff;o++) cand[o]=0;                     /* offset 0 phantom, left off (matches the GA) */
        { int placed=0; while(placed<k){ int oo=1+(int)(r32()%(uint32_t)(g_noff-1)); if(!cand[oo]){cand[oo]=1;placed++;} } }
        bacc=-1.0;
        for(rr=0;rr<revals;rr++){                            /* best-of-r: did this mask ever clear the gate / score well */
            double acc=run_net(cand,(uint32_t)(seed+((uint32_t)e*7u+(uint32_t)rr)*2654435761u+3u),0), fo=objective(cand,acc);
            ec++; if(fo>f) f=fo; if(acc>bacc) bacc=acc;
        }
        ett_check(cand,bacc,ec,&out[4]);
        if(f>bestfit){ bestfit=f; memcpy(best,cand,sizeof best); }
    }
    tally(best,seed,out);
}

static void usage(const char*prog)
{
    printf("usage: %s [options]   (every option also settable via its UPPERCASE env var)\n", prog);
    printf("  parameters:\n");
    PARAMS(P_HELP)
    printf("  modes (choose one; default = scaling table):\n");
    FLAGS(F_HELP)
    printf("  compile-time capacities (rebuild to change): NMAX=%d K=%d NTR=%d NVAL=%d NTE=%d POPcap=%d\n",
           NMAX,K,NTR,NVAL,NTE,DEF_POP);
}

/* argv overrides env-loaded defaults. Accepts --flag, --key value, and --key=value. */
static void parse_args(int argc, char**argv)
{
    int i;
    for(i=1;i<argc;i++){
        char *a=argv[i], *eq; const char *val=NULL;
        if(strncmp(a,"--",2)!=0){ fprintf(stderr,"ignoring non-option arg: %s\n", a); continue; }
        a+=2; eq=strchr(a,'='); if(eq){ *eq=0; val=eq+1; }
        if(!strcmp(a,"help")||!strcmp(a,"h")){ usage(argv[0]); exit(0); }
        else if(0){ /* anchor for FLAGS chain */ }
        FLAGS(F_TRY)
        else {
            if(!val){ if(i+1<argc) val=argv[++i]; else { fprintf(stderr,"missing value for --%s\n", a); exit(2); } }
            if(0){ /* anchor for PARAMS chain */ }
            PARAMS(P_TRY)
            else { fprintf(stderr,"unknown option: --%s\n", a); exit(2); }
        }
    }
    if(g_pop>DEF_POP) g_pop=DEF_POP;
    if(g_pop<2) g_pop=2;
    if(g_elite>=g_pop) g_elite=g_pop-1;
    if(g_elite<1) g_elite=1;
}

int main(int argc, char**argv)
{
    int sd, ni, k, evals;
    int Ns[MAXN], nN=0;

    /* defaults <- environment (argv overrides in parse_args) */
    PARAMS(P_LOAD)
    FLAGS(F_LOAD)

    parse_args(argc, argv);   /* argv overrides env; may exit on --help/error; clamps pop/elite */

    evals = g_pop*(g_gens+1);

    if(g_probe){ char t[NOFFMAX]; int q; set_n(12); new_task(132u);
        for(q=0;q<NOFFMAX;q++) t[q]=0;
        t[10]=1; t[11]=1; t[12]=1;
        printf("PROBE prove: acc(off 10,11,12)=%.4f  acc2=%.4f\n", run_net(t,42u,0), run_net(t,7u,0)); return 0; }

    if(g_trace){   /* GA runs over SEEDS seeds; emit the MEAN population offset-activation per generation */
        int ts, gg, oo, ng=(g_gens<TRACEG?g_gens:TRACEG);
        set_n(g_n0);
        for(gg=0;gg<TRACEG;gg++) for(oo=0;oo<NOFFMAX;oo++) g_tracebuf[gg][oo]=0;
        for(ts=g_seed0;ts<g_seed0+g_seeds;ts++){ double out[5]={0,0,0,0,0};
            new_task((uint32_t)(ts*131+1)); run_ga((uint32_t)(ts*7+1), out); }
        printf("# TRACE N=%d noff=%d off0=%d gen_offsets=%d..%d gens=%d seeds=%d (mean activation)\n",
               g_n,g_noff,g_off0,g_rlo,g_rhi,ng,g_seeds);
        for(gg=0;gg<ng;gg++){ printf("TRACE %d", gg);
            for(oo=0;oo<g_noff;oo++) printf(" %.4f", g_tracebuf[gg][oo]/g_seeds);
            printf("\n"); }
        return 0; }

    if(g_boundary){   /* phase boundary: N* (largest N whose GA still lands on the generative offsets) vs budget */
        int Bl[8], nb=0, Nl[MAXN], nNl=0, bi2, sd2; double thr=g_nstar;
        { char buf[128]; char*p; strncpy(buf,g_budgets,127); buf[127]=0;
          p=strtok(buf,","); while(p&&nb<8){ int val=atoi(p); if(val>0) Bl[nb++]=val; p=strtok(NULL,","); } }
        { char buf[128]; char*p; strncpy(buf,g_nlist,127); buf[127]=0;
          p=strtok(buf,","); while(p&&nNl<MAXN){ int val=atoi(p); if(val>=K+2&&val<=NMAX) Nl[nNl++]=val; p=strtok(NULL,","); } }
        printf("PHASE BOUNDARY: does emergence (GA landing on the K=%d generative offsets) survive to larger N\n", K);
        printf("as the search budget grows?  order parameter = GA on-relevance; N* = largest N with on-rel >= %.2f.\n", thr);
        printf("%d seeds/cell, width=%.1f, target=%.2f.\n\n", g_seeds, g_width, g_target);
        printf("  budget(gens,evals) |");
        for(ni=0;ni<nNl;ni++) printf("  N=%d", Nl[ni]);
        printf("   | N*\n");
        for(bi2=0;bi2<nb;bi2++){ int nstar=-1; g_gens=Bl[bi2]; evals=g_pop*(g_gens+1);
            printf("  gens=%-4d ev=%-6d |", g_gens, evals);
            for(ni=0;ni<nNl;ni++){ set_n(Nl[ni]); double onrel=0;
                for(sd2=g_seed0;sd2<g_seed0+g_seeds;sd2++){ double a[5]={0,0,0,0,0}; new_task((uint32_t)(sd2*131+1));
                    run_ga((uint32_t)(sd2*7+1),a); onrel+=a[2]; }
                onrel/=g_seeds; printf("  %.2f", onrel); if(onrel>=thr) nstar=Nl[ni]; }
            if(nstar>0) printf("   | %d\n", nstar); else printf("   | -\n"); }
        printf("\nphase boundary confirmed if N* GROWS with budget: directed emergence has an upper critical size\n");
        printf("that a larger search budget pushes outward (more evaluations reach the generative offsets at larger N).\n");
        return 0; }

    if(g_agg){   /* read RAW lines from stdin (from parallel chunks), print the aggregated scaling table -- pure C, no external tools */
        static double GA[NMAX+1][4], GQ[NMAX+1][4], RN[NMAX+1][4], RQ[NMAX+1][4];
        static double DC[NMAX+1], DC2[NMAX+1], DO[NMAX+1], DO2[NMAX+1], DT[NMAX+1], DT2[NMAX+1];
        static double EGA[NMAX+1], ERN[NMAX+1]; static int NEG[NMAX+1], NER[NMAX+1];
        static int CNT[NMAX+1], WC[NMAX+1], TC[NMAX+1], LC[NMAX+1], WA[NMAX+1], TA[NMAX+1], LA[NMAX+1];
        int nn, ss, n; double a[4], b[4], ega, ern;
        while(scanf(" RAW %d %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                    &nn,&ss,&a[0],&a[1],&a[2],&a[3],&b[0],&b[1],&b[2],&b[3],&ega,&ern)==12){
            if(nn<0||nn>NMAX) continue;
            CNT[nn]++;
            for(k=0;k<4;k++){ GA[nn][k]+=a[k]; GQ[nn][k]+=a[k]*a[k]; RN[nn][k]+=b[k]; RQ[nn][k]+=b[k]*b[k]; }
            { double d=a[1]-b[1]; DC[nn]+=d; DC2[nn]+=d*d; if(d>0.02) WC[nn]++; else if(d<-0.02) LC[nn]++; else TC[nn]++;
              d=a[2]-b[2]; DO[nn]+=d; DO2[nn]+=d*d;
              d=a[3]-b[3]; DT[nn]+=d; DT2[nn]+=d*d; if(d>0.005) WA[nn]++; else if(d<-0.005) LA[nn]++; else TA[nn]++; }
            if(ega>=0){ EGA[nn]+=ega; NEG[nn]++; }
            if(ern>=0){ ERN[nn]+=ern; NER[nn]++; }
        }
        printf("PROVE (scaling, aggregated): directed energy GA vs RANDOM at matched evals, grouped mutation.\n");
        printf("compact = taps->K(=%d), contig->1, on-rel->1. paired GA-RANDOM per seed (same task).\n", K);
        printf("evals->target = # evals until a mask is first accurate (>=%.2f) AND compact (taps<=%d).\n\n", g_target, g_taptgt);
        printf("  N   arm     taps          contig        on-rel        test        | paired GA-RANDOM (mean±SEM, win/tie/loss)\n");
        for(n=0;n<=NMAX;n++){ int c=CNT[n]; if(c<1) continue;
            double gm[4],gs[4],rm[4],rsd[4]; for(k=0;k<4;k++){ gm[k]=GA[n][k]/c; rm[k]=RN[n][k]/c;
                gs[k]=c>1?sqrt((GQ[n][k]-GA[n][k]*GA[n][k]/c)/(c-1)):0; rsd[k]=c>1?sqrt((RQ[n][k]-RN[n][k]*RN[n][k]/c)/(c-1)):0; }
            double mdc=DC[n]/c, sdc=c>1?sqrt((DC2[n]-DC[n]*DC[n]/c)/(c-1)):0, semc=sdc/sqrt((double)c);
            double mdo=DO[n]/c, sdo=c>1?sqrt((DO2[n]-DO[n]*DO[n]/c)/(c-1)):0, semo=sdo/sqrt((double)c);
            double mdt=DT[n]/c, sdt=c>1?sqrt((DT2[n]-DT[n]*DT[n]/c)/(c-1)):0, semt=sdt/sqrt((double)c);
            printf("  %-3d GA      %.1f±%.1f     %.2f±%.2f    %.2f±%.2f    %.3f±%.3f  | contig %+.2f±%.2f  onrel %+.2f±%.2f  test %+.4f±%.4f  (contig %d/%d/%d, acc %d/%d/%d, n=%d)\n",
                   n, gm[0],gs[0], gm[1],gs[1], gm[2],gs[2], gm[3],gs[3], mdc,semc, mdo,semo, mdt,semt, WC[n],TC[n],LC[n], WA[n],TA[n],LA[n], c);
            printf("  %-3d RANDOM  %.1f±%.1f     %.2f±%.2f    %.2f±%.2f    %.3f±%.3f  |\n",
                   n, rm[0],rsd[0], rm[1],rsd[1], rm[2],rsd[2], rm[3],rsd[3]);
            { double eg=NEG[n]?EGA[n]/NEG[n]:-1.0, er=NER[n]?ERN[n]/NER[n]:-1.0;
              printf("  %-3d evals->target: GA %.0f (%d/%d solved)  RANDOM %.0f (%d/%d solved)  ratio RND/GA %.1fx%s\n",
                     n, eg, NEG[n], c, er, NER[n], c, (eg>0&&er>0)?er/eg:0.0, (NER[n]<NEG[n]?"  (random censored: lower bound)":"")); }
        }
        return 0; }

    /* default mode: scaling table (raw per-seed lines, or aggregated over seeds) */
    { char buf[128]; char *p; strncpy(buf,g_nlist,127); buf[127]=0;
      p=strtok(buf,","); while(p&&nN<MAXN){ int v=atoi(p); if(v>=K+2&&v<=NMAX) Ns[nN++]=v; p=strtok(NULL,","); }
      if(nN==0){ int def[]={12,16,20,24,28}; for(k=0;k<5;k++) Ns[nN++]=def[k]; } }
    if(!g_raw){
        printf("PROVE (scaling): directed energy GA vs RANDOM search at matched evals (%d), grouped mutation, width=%.1f\n", evals, g_width);
        printf("local K=%d generative filter, %d seeds, %d gens, pop %d. paired per seed. compact = taps->K(=%d), contig->1, on-rel->1\n", K, g_seeds, g_gens, g_pop, K);
        printf("evals->target = # evals until a mask is first accurate (>=%.2f) AND compact (taps<=%d).\n\n", g_target, g_taptgt);
        printf("  N   arm     taps          contig        on-rel        test        | paired GA-RANDOM (mean±SEM, win/tie/loss)\n"); }
    for(ni=0;ni<nN;ni++){ set_n(Ns[ni]);
        double ga[4]={0,0,0,0}, gq[4]={0,0,0,0}, rn[4]={0,0,0,0}, rq[4]={0,0,0,0};
        double dc=0,dc2=0,doo=0,doo2=0,dt=0,dt2=0; int wc=0,tc=0,lc=0,wa=0,ta=0,la=0;
        double sega=0,sern=0; int nsg=0,nsr=0;      /* evals-to-target: sum + solved-count per arm */
        for(sd=g_seed0;sd<g_seed0+g_seeds;sd++){ new_task((uint32_t)(sd*131+1));
            double a[5]={0,0,0,0,0}, b[5]={0,0,0,0,0};
            run_ga((uint32_t)(sd*7+1),a);
            run_random((uint32_t)(sd*7+1),evals,g_revals,b);
            if(g_raw){ printf("RAW %d %d %.0f %.4f %.4f %.4f %.0f %.4f %.4f %.4f %.0f %.0f\n",
                            Ns[ni], sd, a[0],a[1],a[2],a[3], b[0],b[1],b[2],b[3], a[4], b[4]); continue; }
            for(k=0;k<4;k++){ ga[k]+=a[k]; gq[k]+=a[k]*a[k]; }
            for(k=0;k<4;k++){ rn[k]+=b[k]; rq[k]+=b[k]*b[k]; }
            { double d=a[1]-b[1]; dc+=d; dc2+=d*d; if(d>0.02) wc++; else if(d<-0.02) lc++; else tc++;
              d=a[2]-b[2]; doo+=d; doo2+=d*d;
              d=a[3]-b[3]; dt+=d; dt2+=d*d; if(d>0.005) wa++; else if(d<-0.005) la++; else ta++; }
            if(a[4]>=0){ sega+=a[4]; nsg++; }  if(b[4]>=0){ sern+=b[4]; nsr++; }
            if(g_dbg) printf("  [dbg N=%d seed=%d GA c=%.2f o=%.2f | RND c=%.2f o=%.2f]\n", Ns[ni], sd, a[1],a[2], b[1],b[2]);
        }
        if(g_raw) continue;
        { double gm[4],gs[4],rm[4],rsd[4]; for(k=0;k<4;k++){ gm[k]=ga[k]/g_seeds; rm[k]=rn[k]/g_seeds;
            gs[k]=g_seeds>1?sqrt((gq[k]-ga[k]*ga[k]/g_seeds)/(g_seeds-1)):0; rsd[k]=g_seeds>1?sqrt((rq[k]-rn[k]*rn[k]/g_seeds)/(g_seeds-1)):0; }
          double mdc=dc/g_seeds, sdc=g_seeds>1?sqrt((dc2-dc*dc/g_seeds)/(g_seeds-1)):0, semc=sdc/sqrt((double)g_seeds);
          double mdo=doo/g_seeds, sdo=g_seeds>1?sqrt((doo2-doo*doo/g_seeds)/(g_seeds-1)):0, semo=sdo/sqrt((double)g_seeds);
          double mdt=dt/g_seeds, sdt=g_seeds>1?sqrt((dt2-dt*dt/g_seeds)/(g_seeds-1)):0, semt=sdt/sqrt((double)g_seeds);
          printf("  %-3d GA      %.1f±%.1f     %.2f±%.2f    %.2f±%.2f    %.3f±%.3f  | contig %+.2f±%.2f  onrel %+.2f±%.2f  test %+.4f±%.4f  (contig %d/%d/%d, acc %d/%d/%d)\n",
                 Ns[ni], gm[0],gs[0], gm[1],gs[1], gm[2],gs[2], gm[3],gs[3], mdc,semc, mdo,semo, mdt,semt, wc,tc,lc, wa,ta,la);
          printf("  %-3d RANDOM  %.1f±%.1f     %.2f±%.2f    %.2f±%.2f    %.3f±%.3f  |\n",
                 Ns[ni], rm[0],rsd[0], rm[1],rsd[1], rm[2],rsd[2], rm[3],rsd[3]);
          { double eg=nsg?sega/nsg:-1.0, er=nsr?sern/nsr:-1.0;
            printf("  %-3d evals->target: GA %.0f (%d/%d solved)  RANDOM %.0f (%d/%d solved)  ratio RND/GA %.1fx%s\n",
                   Ns[ni], eg, nsg, g_seeds, er, nsr, g_seeds, (eg>0&&er>0)?er/eg:0.0, (nsr<nsg?"  (random censored: lower bound)":"")); }
        }
    }
    if(!g_raw){
        printf("\ndirected search does real work if the paired GA-RANDOM contig/on-rel gap is positive by several SEM,\n");
        printf("and the scaling claim holds if that gap GROWS with N (tie at small N, widening advantage as the space grows).\n");
        printf("evals->target makes the efficiency explicit: RND/GA is the (unbounded) budget random needs to match the GA;\n");
        printf("it should keep GROWING with N even where the bounded quality gap saturates.\n"); }
    return 0;
}
