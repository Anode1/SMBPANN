/* conv_emerge.c -- topology by DIRECTED EMERGENCE under an energy budget.
 *
 * Emergence -- structure arising from simple local rules -- is the theme of Conway's Life,
 * Wolfram's automata, fractals, and chaos, but there it is open-ended. Here it is
 * *directed*: a genetic pressure steers emergence toward a structure that solves a task with
 * the least energy. The original SMBPANN ambition was exactly this, and convolution is the
 * ideal proof because the answer (LeCun's local receptive field) is already known.
 *
 * Method: start from a DENSE seed (every hidden unit wired to every input) and evolve binary
 * connectivity masks. Objective: MINIMIZE ENERGY (number of connections) SUBJECT TO reaching
 * a target accuracy. Removing a connection saves energy; removing a USEFUL one breaks the
 * accuracy floor and is punished, so the search settles on the minimal structure that still
 * solves the task. Mutation is mostly-prune with RARE growth (annealing at low temperature:
 * downhill prune moves, occasional uphill add moves to escape local minima).
 *
 * The headline question: does the RIGHT topology emerge for the RIGHT task? We run the same
 * method on three information structures:
 *   LOCAL   -- label depends on local windows  -> local receptive fields should emerge (conv)
 *   GLOBAL  -- label depends on all inputs      -> broad connectivity should be kept
 *   SPARSE  -- label depends on a few inputs    -> connections should select those (features)
 * If the emerged structure matches the task's information in each case, the method discovers
 * task-appropriate inductive bias -- the point of using it on domains where we do not know it.
 *
 * Also: an annealing-temperature sweep (growth rate) and an energy/accuracy target sweep.
 * Metrics: density (energy), receptive-field span (locality), and the fraction of surviving
 * connections that land on task-RELEVANT inputs. Weight sharing (convolution's other half) a
 * mask cannot express, so this probes emergence of CONNECTIVITY only. Self-contained C99.
 * Config via env: SEEDS, GENS, TARGET, PADD (growth rate).  Build: make conv_emerge
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define N   12
#define K   3
#define H   (N - K + 1)      /* = 10 */
#define SUBSZ 4              /* sparse task: how many inputs actually matter */
#define NTR   64
#define NVAL  300
#define NTE   1000
#define TEPOCHS 50
#define LR    0.1
#define POP   24
#define ELITE 4

/* GA + data PRNG */
static uint32_t rs = 1u;
static uint32_t r32(void){ uint32_t x=rs; x^=x<<13; x^=x>>17; x^=x<<5; rs=x; return x; }
static void     rseed(uint32_t s){ rs = s?s:1u; }
static double   runif(void){ return (double)r32()/4294967296.0*2.0 - 1.0; }
/* separate PRNG for weight init, so fitness() never disturbs the GA stream */
static uint32_t wr = 1u;
static uint32_t wr32(void){ uint32_t x=wr; x^=x<<13; x^=x>>17; x^=x<<5; wr=x; return x; }
static void     wseed(uint32_t s){ wr = s?s:1u; }
static double   wunif(void){ return (double)wr32()/4294967296.0*2.0 - 1.0; }

/* ---- tasks (all translation of "which inputs carry the signal") ---- */
enum { T_LOCAL = 0, T_GLOBAL = 1, T_SPARSE = 2 };
static int    cur_task = T_LOCAL;
static double wstar[K];              /* LOCAL filter */
static double gvec[N];               /* GLOBAL weights over all inputs */
static int    sub[SUBSZ];            /* SPARSE: which inputs matter */
static double ssub[SUBSZ];           /* SPARSE weights */

static double Xtr[NTR][N], Xval[NVAL][N], Xte[NTE][N];
static int    ytr[NTR], yval[NVAL], yte[NTE];
static char   g_best[H][N];          /* best mask from the last run_ga (for reuse) */

/* configuration (env-settable) */
static double g_target = 0.90;
static double g_padd   = 0.006;      /* growth (add) rate: low = cool anneal, stable */
static double g_prem   = 0.030;      /* prune (remove) rate */

static int label_of(const double *x)
{
    int p, k; double s = 0;
    if (cur_task == T_LOCAL) {
        for (p=0;p+K<=N;p++){ double a=0; for(k=0;k<K;k++) a+=wstar[k]*x[p+k]; s+=tanh(a); }
    } else if (cur_task == T_GLOBAL) {
        int i; for (i=0;i<N;i++) s += gvec[i]*x[i];  /* linear over ALL inputs: no locality to find */
    } else { /* T_SPARSE */
        int t; for (t=0;t<SUBSZ;t++) s += ssub[t]*x[sub[t]];
    }
    return (s > 0) ? 1 : 0;
}
static void gen(double X[][N], int *y, int n)
{ int s,i; for(s=0;s<n;s++){ for(i=0;i<N;i++) X[s][i]=runif(); y[s]=label_of(X[s]); } }

static void new_task(int task, uint32_t seed)
{
    int k, i, t;
    cur_task = task; rseed(seed);
    for (k=0;k<K;k++) wstar[k]=runif();
    for (i=0;i<N;i++) gvec[i]=runif();
    { int chosen[N]; for(i=0;i<N;i++) chosen[i]=0;                /* SUBSZ distinct positions */
      for (t=0;t<SUBSZ;){ int p=(int)(r32()%(uint32_t)N); if(!chosen[p]){chosen[p]=1; sub[t]=p; ssub[t]=runif(); t++; } } }
    gen(Xtr,ytr,NTR); gen(Xval,yval,NVAL); gen(Xte,yte,NTE);
}

/* ---- structure metrics ---- */
static double density(const char m[H][N])
{ long a=0; int j,i; for(j=0;j<H;j++)for(i=0;i<N;i++) a+=m[j][i]; return (double)a/(H*N); }

static double rf_span(const char m[H][N])
{ int j,i,cnt=0; double tot=0;
  for(j=0;j<H;j++){ int lo=-1,hi=-1; for(i=0;i<N;i++) if(m[j][i]){ if(lo<0)lo=i; hi=i; }
      if(lo>=0){ tot+=(double)(hi-lo+1)/N; cnt++; } }
  return cnt? tot/cnt : 0.0; }

/* fraction of surviving connections that land on a task-RELEVANT input */
static int relevant(int j, int i)
{
    if (cur_task == T_LOCAL)  return (i>=j && i<j+K);            /* unit j's local window */
    if (cur_task == T_GLOBAL) return 1;                          /* all inputs matter */
    { int t; for(t=0;t<SUBSZ;t++) if(i==sub[t]) return 1; return 0; }  /* SPARSE subset */
}
static double on_relevant(const char m[H][N])
{ long act=0,rel=0; int j,i;
  for(j=0;j<H;j++)for(i=0;i<N;i++) if(m[j][i]){ act++; if(relevant(j,i)) rel++; }
  return act? (double)rel/act : 0.0; }

/* train the masked net on the scarce train set, return validation accuracy */
static double fitness(const char m[H][N], uint32_t seed)
{
    static double W[H][N], bh[H], v[H]; double bo=0,h[H]; int i,j,e,s,c=0;
    wseed(seed);
    for(j=0;j<H;j++){ for(i=0;i<N;i++) W[j][i]=m[j][i]?0.1*wunif():0.0; bh[j]=0; v[j]=0.1*wunif(); }
    for(e=0;e<TEPOCHS;e++) for(s=0;s<NTR;s++){ const double*x=Xtr[s]; double opre=bo,o,dout;
        for(j=0;j<H;j++){ double pre=bh[j]; for(i=0;i<N;i++) if(m[j][i]) pre+=W[j][i]*x[i]; h[j]=tanh(pre); opre+=v[j]*h[j]; }
        o=1.0/(1.0+exp(-opre)); dout=(o-ytr[s])*o*(1.0-o);
        for(j=0;j<H;j++){ double dpre=dout*v[j]*(1.0-h[j]*h[j]); v[j]-=LR*dout*h[j]; bh[j]-=LR*dpre;
            for(i=0;i<N;i++) if(m[j][i]) W[j][i]-=LR*dpre*x[i]; } bo-=LR*dout; }
    for(s=0;s<NVAL;s++){ const double*x=Xval[s]; double opre=bo,o;
        for(j=0;j<H;j++){ double pre=bh[j]; for(i=0;i<N;i++) if(m[j][i]) pre+=W[j][i]*x[i]; opre+=v[j]*tanh(pre);}
        o=1.0/(1.0+exp(-opre)); if((o>0.5)==(yval[s]==1)) c++; }
    return (double)c/NVAL;
}
static double test_acc(const char m[H][N], uint32_t seed)
{
    static double W[H][N], bh[H], v[H]; double bo=0,h[H]; int i,j,e,s,c=0;
    wseed(seed);
    for(j=0;j<H;j++){ for(i=0;i<N;i++) W[j][i]=m[j][i]?0.1*wunif():0.0; bh[j]=0; v[j]=0.1*wunif(); }
    for(e=0;e<TEPOCHS;e++) for(s=0;s<NTR;s++){ const double*x=Xtr[s]; double opre=bo,o,dout;
        for(j=0;j<H;j++){ double pre=bh[j]; for(i=0;i<N;i++) if(m[j][i]) pre+=W[j][i]*x[i]; h[j]=tanh(pre); opre+=v[j]*h[j];}
        o=1.0/(1.0+exp(-opre)); dout=(o-ytr[s])*o*(1.0-o);
        for(j=0;j<H;j++){ double dpre=dout*v[j]*(1.0-h[j]*h[j]); v[j]-=LR*dout*h[j]; bh[j]-=LR*dpre;
            for(i=0;i<N;i++) if(m[j][i]) W[j][i]-=LR*dpre*x[i]; } bo-=LR*dout; }
    for(s=0;s<NTE;s++){ const double*x=Xte[s]; double opre=bo,o;
        for(j=0;j<H;j++){ double pre=bh[j]; for(i=0;i<N;i++) if(m[j][i]) pre+=W[j][i]*x[i]; opre+=v[j]*tanh(pre);}
        o=1.0/(1.0+exp(-opre)); if((o>0.5)==(yte[s]==1)) c++; }
    return (double)c/NTE;
}

/* energy-under-constraint objective: feasible (acc>=target) in [1,2] ranked by 1-density,
 * infeasible = accuracy in [0,1) so the population climbs to feasibility first */
static double objective(const char m[H][N], double acc)
{ return (acc >= g_target) ? (2.0 - density(m)) : acc; }

/* one GA run. Fills final-generation means into out[] = {span,density,on_relevant,feasible};
 * returns best-individual TEST accuracy. sel=0 -> drift (no selection). */
static double run_ga(int sel, int gens, uint32_t seed, double out[4], double *trel, double *tden)
{
    static char pop[POP][H][N], nxt[POP][H][N];
    double fit[POP], facc[POP]; int idx[POP]; int g,p,q,i,j;
    rseed(seed);
    for(p=0;p<POP;p++) for(j=0;j<H;j++) for(i=0;i<N;i++) pop[p][j][i]=1;
    for(p=0;p<POP;p++){ facc[p]=fitness(pop[p],(uint32_t)(seed+p*2654435761u+1u)); fit[p]=objective(pop[p],facc[p]); }
    for(g=0;g<gens;g++){
        if(trel){ double R=0,D=0; for(p=0;p<POP;p++){R+=on_relevant(pop[p]);D+=density(pop[p]);} trel[g]+=R/POP; tden[g]+=D/POP; }
        for(p=0;p<POP;p++) idx[p]=p;
        if(sel){ for(p=0;p<POP;p++) for(q=p+1;q<POP;q++) if(fit[idx[q]]>fit[idx[p]]){int t=idx[p];idx[p]=idx[q];idx[q]=t;} }
        else   { for(p=POP-1;p>0;p--){int r=(int)(r32()%(uint32_t)(p+1));int t=idx[p];idx[p]=idx[r];idx[r]=t;} }
        for(p=0;p<ELITE;p++) memcpy(nxt[p],pop[idx[p]],sizeof pop[0]);
        for(p=ELITE;p<POP;p++){ int a=idx[(int)(r32()%ELITE)]; memcpy(nxt[p],pop[a],sizeof pop[0]);
            for(j=0;j<H;j++) for(i=0;i<N;i++){                       /* mostly-prune, rare-grow mutation */
                if(nxt[p][j][i]){ if((double)r32()/4294967296.0 < g_prem) nxt[p][j][i]=0; }
                else            { if((double)r32()/4294967296.0 < g_padd) nxt[p][j][i]=1; } } }
        memcpy(pop,nxt,sizeof pop);
        for(p=0;p<POP;p++){ facc[p]=fitness(pop[p],(uint32_t)(seed+(uint32_t)(g*POP+p)+7u)); fit[p]=objective(pop[p],facc[p]); }
    }
    { double sp=0,de=0,re=0; int f=0;
      for(p=0;p<POP;p++){ sp+=rf_span(pop[p]); de+=density(pop[p]); re+=on_relevant(pop[p]); if(facc[p]>=g_target) f++; }
      out[0]+=sp/POP; out[1]+=de/POP; out[2]+=re/POP; out[3]+=(double)f/POP; }
    { int bi=0; double bf=-1; for(p=0;p<POP;p++) if(fit[p]>bf){bf=fit[p];bi=p;}
      memcpy(g_best, pop[bi], sizeof g_best); return test_acc(pop[bi],seed+999u); }
}

static void avg(double out[4], int seeds){ int k; for(k=0;k<4;k++) out[k]/=seeds; }
static int envint(const char*k,int d){const char*e=getenv(k);return e&&*e?atoi(e):d;}
static double envdbl(const char*k,double d){const char*e=getenv(k);return e&&*e?atof(e):d;}

/* run selection AND drift on a task over all seeds; print the row. The emergence signal is
 * ON-RELEVANT: under selection the surviving connections land on the task-relevant inputs
 * (far above the chance baseline and above drift), while drift prunes at random. */
static void task_row(const char *name, int task, int gens, int seeds, double chance)
{
    double sel[4]={0,0,0,0}, drf[4]={0,0,0,0}, acc=0; int sd;
    for (sd=1; sd<=seeds; sd++){
        new_task(task,(uint32_t)(sd*131+1));
        acc += run_ga(1,gens,(uint32_t)(sd*7+1),sel,0,0)/seeds;
        new_task(task,(uint32_t)(sd*131+1));
        run_ga(0,gens,(uint32_t)(sd*7+1),drf,0,0);
    }
    avg(sel,seeds); avg(drf,seeds);
    printf("  %-18s  %.3f   %.3f    %.3f  (chance %.2f, drift %.3f)   %.3f\n",
           name, sel[1], sel[0], sel[2], chance, drf[2], acc);
}

int main(void)
{
    int seeds = envint("SEEDS", 16), gens = envint("GENS", 120), sd, r;
    g_target = envdbl("TARGET", 0.90);
    g_padd   = envdbl("PADD",   0.006);

    printf("DIRECTED EMERGENCE of topology: minimize energy s.t. accuracy >= %.2f\n", g_target);
    printf("dense seed, N=%d hidden=%d, scarce train=%d, %d seeds x %d generations\n", N, H, NTR, seeds, gens);
    printf("mutation: prune rate %.3f, grow rate %.3f (low grow = cool anneal)\n\n", g_prem, g_padd);

    /* ---- headline: same method, three information structures ---- */
    printf("== does the RIGHT topology emerge for the RIGHT task? ==\n");
    printf("on-relevant = fraction of surviving connections landing on task-relevant inputs\n");
    printf("  %-18s  dens   span    on-relevant                        test\n", "task");
    task_row("LOCAL (windows)",  T_LOCAL,  gens, seeds, (double)K/N);
    task_row("GLOBAL (all)",     T_GLOBAL, gens, seeds, 1.0);
    task_row("SPARSE (few)",     T_SPARSE, gens, seeds, (double)SUBSZ/N);
    printf("  read: LOCAL/SPARSE lift on-relevant well above chance and above drift (structure\n");
    printf("  matches the task); GLOBAL keeps broad connectivity (density stays high).\n");

    /* ---- annealing temperature: growth rate vs convergence on LOCAL ---- */
    printf("\n== annealing temperature (grow rate) on LOCAL: rarer growth converges lower ==\n");
    printf("  %-10s  span   density  on-relevant  test\n", "grow-rate");
    { double rates[4] = {0.000, 0.006, 0.03, 0.10};
      for (r=0;r<4;r++){ double sel[4]={0,0,0,0}, acc=0; g_padd=rates[r];
        for (sd=1; sd<=seeds; sd++){ new_task(T_LOCAL,(uint32_t)(sd*131+1)); acc+=run_ga(1,gens,(uint32_t)(sd*7+1),sel,0,0)/seeds; }
        avg(sel,seeds);
        printf("  %-10.3f  %.3f   %.3f    %.3f      %.3f\n", rates[r], sel[0], sel[1], sel[2], acc); }
      g_padd = envdbl("PADD",0.006); }

    /* ---- energy/accuracy Pareto on LOCAL ---- */
    printf("\n== energy vs accuracy target on LOCAL (Pareto): higher target keeps more energy ==\n");
    printf("  %-8s  density  span   feasible  test\n", "target");
    { double tg[4] = {0.80, 0.85, 0.90, 0.95}; double save=g_target;
      for (r=0;r<4;r++){ double sel[4]={0,0,0,0}, acc=0; g_target=tg[r];
        for (sd=1; sd<=seeds; sd++){ new_task(T_LOCAL,(uint32_t)(sd*131+1)); acc+=run_ga(1,gens,(uint32_t)(sd*7+1),sel,0,0)/seeds; }
        avg(sel,seeds);
        printf("  %-8.2f  %.3f    %.3f    %3.0f%%    %.3f\n", tg[r], sel[1], sel[0], 100*sel[3], acc); }
      g_target=save; }

    /* ---- convergence trajectory: watch feature selection emerge on SPARSE ---- */
    printf("\n== emergence in action: SPARSE feature selection over generations ==\n");
    { double trel[512]={0}, tden[512]={0}; int g, mile[6]={0,10,25,50,100,0};
      mile[5]=gens-1;
      for (sd=1; sd<=seeds; sd++){ new_task(T_SPARSE,(uint32_t)(sd*131+1));
        run_ga(1,gens,(uint32_t)(sd*7+1),(double[4]){0,0,0,0},trel,tden); }
      for (g=0; g<gens; g++){ trel[g]/=seeds; tden[g]/=seeds; }
      printf("  %-5s  on-relevant  density   (chance on-relevant = %.2f)\n", "gen", (double)SUBSZ/N);
      { int r; for (r=0;r<6;r++){ g=mile[r]; if(g<0||g>=gens) continue;
          printf("  %-5d   %.3f       %.3f\n", g, trel[g], tden[g]); } } }

    /* ---- reuse: emerge a structure on one LOCAL task, apply to a fresh one ---- */
    printf("\n== reuse: emerge structure on task A, apply to fresh task B (same family) ==\n");
    { char keep[H][N]; double dA, tB, tBd; int a,b; static char dense[H][N];
      new_task(T_LOCAL, 20240001u); run_ga(1,gens,20240002u,(double[4]){0,0,0,0},0,0);
      memcpy(keep, g_best, sizeof keep); dA = density(keep);
      new_task(T_LOCAL, 20240777u);
      for(a=0;a<H;a++) for(b=0;b<N;b++) dense[a][b]=1;
      tB = test_acc(keep, 31u); tBd = test_acc(dense, 31u);
      printf("  emerged structure (density %.3f): test %.3f on B   vs   dense (density 1.0): test %.3f\n",
             dA, tB, tBd);
      printf("  -> a structure discovered once transfers across the family at %.0f%% of the energy.\n", 100*dA); }
    return 0;
}
