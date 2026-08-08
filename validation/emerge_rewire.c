/* emerge_rewire.c -- does REWIRING (placement change at constant energy) find structure that
 * add/remove cannot? smbpann2 probe. Paper 1 (../articles/smbpann) is not affected by anything here.
 *
 * WHY THIS PROBE EXISTS. emerge_local.c gives each hidden unit a contiguous window [start, start+w)
 * and charges energy for free parameters: shared => #distinct tied weights (= max width); unshared =>
 * total connections (= sum of widths). NEITHER READS `start`. So moving a window at fixed width is
 * exactly energy-neutral: the equal-energy genomes form a connected NEUTRAL NETWORK, and an operator
 * that slides windows travels it at zero energy cost. That is the biological picture (synaptic
 * rewiring is frequent; adding a unit is rare), and it is also the one axis a count-based penalty
 * provably cannot select on, which is the mechanism paper 1 identified on the width axis.
 *
 * THE DEFECT THIS PROBE FIXES. emerge_local.c's objective is
 *     (acc >= target) ? (2 - energy) : acc
 * a STEP. Above target it is `2 - energy`, which does not read `start`, so it is exactly constant in
 * placement: measured over 288 hand-built genomes at fixed width, all 15 that cleared target scored
 * 1.975000, identical to six decimals, while accuracy over the same genomes ranged 0.559..0.931.
 * Since any genome clearing target outscores every genome below it, the endgame -- precisely when
 * structure is supposed to refine, and when the final population is measured -- is neutral DRIFT.
 * An operator evaluated there is a random walk, which is the artifact class PROTOCOL.md was written
 * about. Here the objective is CONTINUOUS, `acc - lambda*energy`, so accuracy keeps selecting on
 * placement after the threshold would have fired.
 *
 * ARMS (ARM=0,1,2). Same budget, same seeds, same everything else:
 *   0  no-rewire     start never mutates; only width grows/shrinks (the add/remove-only control)
 *   1  slide-1       start += +-1 at rate PSLIDE          (neighbour-only; diffusive, O(d^2) to move d)
 *   2  rewire-rand   start := uniform valid at rate PSLIDE (any target; O(1) to move d)
 * Arm 1 IMPOSES a spatial neighbourhood, which is the prior under test, so arm 2 is the primary
 * result and arm 1 measures what the prior buys. GENS=0 is the no-evolution control for every arm.
 *
 * PROTOCOL.md COMPLIANCE. Item 1: sensitivity_check() runs by default and ABORTS (rc=2) if the
 * objective is flat over placement or over width. Item 2: coverage is the free variable of the old
 * objective, so every run emits a RAW line and the arm effect must be read as the increment over the
 * pooled acc(coverage) curve, not as a raw difference. Item 3: every constant below is an env knob.
 *
 * Reports per arm: shared-frac, mean/max width, coverage, energy, test acc, plus RAW rows.
 * Self-contained C99.  Build: make emerge_rewire
 *   env: SEEDS GENS LAMBDA PSLIDE PGROW PSHARE RAW   (all three arms run in one invocation)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define N   12
#define K   3
#define H   (N - K + 1)     /* = 10 */
#define NTR   64
#define NVAL  300
#define NTE   1000
#define TEPOCHS 50
#define LR    0.1
#define POP   24
#define ELITE 4

static uint32_t rs=1u; static uint32_t r32(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;rs=x;return x;}
static void rseed(uint32_t s){rs=s?s:1u;} static double runif(void){return (double)r32()/4294967296.0*2.0-1.0;}
static double rprob(void){return (double)r32()/4294967296.0;}
static uint32_t wr=1u; static uint32_t wr32(void){uint32_t x=wr;x^=x<<13;x^=x>>17;x^=x<<5;wr=x;return x;}
static void wseed(uint32_t s){wr=s?s:1u;} static double wunif(void){return (double)wr32()/4294967296.0*2.0-1.0;}

static double wstar[K];
static double Xtr[NTR][N], Xval[NVAL][N], Xte[NTE][N];
static int    ytr[NTR], yval[NVAL], yte[NTE];
static double g_lambda = 1.0, g_pgrow = 0.10, g_pshare = 0.05, g_pslide = 0.15;
static int    g_gens = 150, g_arm = 2, g_raw = 0;
static int    envint(const char*k,int d){const char*e=getenv(k);return e&&*e?atoi(e):d;}
static double envdbl(const char*k,double d){const char*e=getenv(k);return e&&*e?atof(e):d;}

static int label_of(const double *x)
{ int p,k; double s=0; for(p=0;p+K<=N;p++){ double a=0; for(k=0;k<K;k++) a+=wstar[k]*x[p+k]; s+=tanh(a);} return s>0; }
static void gen(double X[][N],int*y,int n){ int s,i; for(s=0;s<n;s++){ for(i=0;i<N;i++) X[s][i]=runif(); y[s]=label_of(X[s]); } }
static void new_task(uint32_t seed){ int k; rseed(seed); for(k=0;k<K;k++) wstar[k]=runif(); gen(Xtr,ytr,NTR); gen(Xval,yval,NVAL); gen(Xte,yte,NTE); }

typedef struct { int start[H], w[H]; int shared; } Indiv;

static int maxwidth(const Indiv*g){ int j,m=0; for(j=0;j<H;j++) if(g->w[j]>m) m=g->w[j]; return m; }
/* free-parameter energy, normalized by H*N. shared => #distinct tied weights (=max width);
 * unshared => total connections (sum of widths).  NOTE: independent of start[] -- that is the point. */
static double param_energy(const Indiv*g)
{ int j; if(g->shared) return (double)maxwidth(g)/(H*N);
  { long a=0; for(j=0;j<H;j++) a+=g->w[j]; return (double)a/(H*N); } }
static double meanwidth(const Indiv*g){ int j; double s=0; for(j=0;j<H;j++) s+=g->w[j]; return s/H; }
static double coverage(const Indiv*g)
{ char c[N]; int j,t,i,cov=0; memset(c,0,sizeof c);
  for(j=0;j<H;j++) for(t=0;t<g->w[j];t++){ int i2=g->start[j]+t; if(i2>=0&&i2<N) c[i2]=1; }
  for(i=0;i<N;i++) cov+=c[i];
  return (double)cov/N; }

static double run_net(const Indiv*g, uint32_t seed, int on_test)
{
    static double W[H][N], wsh[N], bh[H], v[H];
    double bo=0, h[H]; int j,t,e,s,c=0, ns = on_test?NTE:NVAL, mw=maxwidth(g);
    double (*Xe)[N] = on_test?Xte:Xval; int *ye = on_test?yte:yval;
    wseed(seed);
    for(t=0;t<N;t++) wsh[t]=0.1*wunif();
    for(j=0;j<H;j++){ for(t=0;t<N;t++) W[j][t]=0.1*wunif(); bh[j]=0; v[j]=0.1*wunif(); }
    for(e=0;e<TEPOCHS;e++) for(s=0;s<NTR;s++){
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
    for(s=0;s<ns;s++){ const double *x=Xe[s]; double opre=bo,o;
        for(j=0;j<H;j++){ double pre=bh[j];
            for(t=0;t<g->w[j];t++) pre+=(g->shared?wsh[t]:W[j][t])*x[g->start[j]+t];
            opre+=v[j]*tanh(pre); }
        o=1.0/(1.0+exp(-opre)); if((o>0.5)==(ye[s]==1)) c++; }
    return (double)c/ns;
}

/* CONTINUOUS objective: no step, so accuracy keeps selecting on placement at every accuracy level.
 * This is the one substantive change from emerge_local.c. */
static double objective(const Indiv*g, double acc){ return acc - g_lambda*param_energy(g); }

static void clampw(Indiv*g){ int j; for(j=0;j<H;j++){ if(g->w[j]<1) g->w[j]=1; if(g->w[j]>N) g->w[j]=N;
    if(g->start[j]<0) g->start[j]=0;
    if(g->start[j]+g->w[j]>N) g->start[j]=N-g->w[j]; } }

/* PROTOCOL item 1. Hand-build genomes spanning each outcome axis and confirm the objective MOVES.
 * Placement axis: widths pinned at K, sharing on, only start varies -> energy identical by
 * construction, so any variation is accuracy seeing placement. Width axis: starts pinned, w varies. */
static int sensitivity_check(int seeds)
{
    int sd,j,c; double pmin=1e9,pmax=-1e9,wmin=1e9,wmax=-1e9, emin=1e9,emax=-1e9;
    for(sd=1; sd<=seeds; sd++){
        new_task((uint32_t)(sd*911u+1u));
        for(c=0;c<8;c++){                                  /* placement axis, energy fixed */
            Indiv g; g.shared=1; for(j=0;j<H;j++) g.w[j]=K;
            if(c==0)      for(j=0;j<H;j++) g.start[j]=0;
            else if(c==1) for(j=0;j<H;j++) g.start[j]=(j%2)?N-K:0;
            else if(c==2) for(j=0;j<H;j++) g.start[j]=j;
            else          for(j=0;j<H;j++) g.start[j]=(int)(r32()%(uint32_t)(N-K+1));
            clampw(&g);
            { double o=objective(&g, run_net(&g,(uint32_t)(sd*7u+c*131u+1u),0)), e=param_energy(&g);
              if(o<pmin)pmin=o; if(o>pmax)pmax=o; if(e<emin)emin=e; if(e>emax)emax=e; } }
        for(c=1;c<=N;c++){                                 /* width axis */
            Indiv g; g.shared=1; for(j=0;j<H;j++){ g.w[j]=c; g.start[j]=0; }
            clampw(&g);
            { double o=objective(&g, run_net(&g,(uint32_t)(sd*7u+c*17u+3u),0));
              if(o<wmin)wmin=o; if(o>wmax)wmax=o; } }
    }
    printf("SENSITIVITY  placement: objective %.4f..%.4f (span %.4f) at energy %.6f..%.6f\n",
           pmin,pmax,pmax-pmin,emin,emax);
    printf("SENSITIVITY  width:     objective %.4f..%.4f (span %.4f)\n", wmin,wmax,wmax-wmin);
    if(pmax-pmin < 1e-6){ printf("SENSITIVITY FAIL: objective is flat over PLACEMENT. Nothing a rewire\n"
                                 "operator does can be credited for a placement outcome. Stop.\n"); return 0; }
    if(wmax-wmin < 1e-6){ printf("SENSITIVITY FAIL: objective is flat over WIDTH.\n"); return 0; }
    return 1;
}

/* out += {shared_frac, mean_width, max_width, coverage, test, energy} */
static void run_ga(uint32_t seed, double out[6], int sd)
{
    static Indiv pop[POP], nxt[POP]; double fit[POP], facc[POP]; int idx[POP]; int g,p,q,j;
    rseed(seed);
    for(p=0;p<POP;p++){ for(j=0;j<H;j++){ pop[p].start[j]=0; pop[p].w[j]=N; } pop[p].shared=0; }
    for(p=0;p<POP;p++){ facc[p]=run_net(&pop[p],(uint32_t)(seed+p*2654435761u+1u),0);
                        fit[p]=objective(&pop[p],facc[p]); }
    for(g=0;g<g_gens;g++){
        for(p=0;p<POP;p++) idx[p]=p;
        for(p=0;p<POP;p++) for(q=p+1;q<POP;q++) if(fit[idx[q]]>fit[idx[p]]){int t=idx[p];idx[p]=idx[q];idx[q]=t;}
        for(p=0;p<ELITE;p++) nxt[p]=pop[idx[p]];
        for(p=ELITE;p<POP;p++){ int a=idx[(int)(r32()%ELITE)]; nxt[p]=pop[a];
            for(j=0;j<H;j++){
                if(rprob()<g_pgrow) nxt[p].w[j]+=1; else if(rprob()<0.5) nxt[p].w[j]-=1;
                if(nxt[p].w[j]<1) nxt[p].w[j]=1;
                if(nxt[p].w[j]>N) nxt[p].w[j]=N;
                /* THE ARM. Width mutation above is identical everywhere; only placement differs. */
                if(g_arm==1){ if(rprob()<g_pslide) nxt[p].start[j] += (rprob()<0.5)?-1:1; }
                else if(g_arm==2){ if(rprob()<g_pslide) nxt[p].start[j] = (int)(r32()%(uint32_t)(N-nxt[p].w[j]+1)); }
            }
            if(rprob()<g_pshare) nxt[p].shared^=1;
            clampw(&nxt[p]); }
        memcpy(pop,nxt,sizeof pop);
        for(p=0;p<POP;p++){ facc[p]=run_net(&pop[p],(uint32_t)(seed+(uint32_t)(g*POP+p)+7u),0);
                            fit[p]=objective(&pop[p],facc[p]); }
    }
    { double sh=0,mw=0,mx=0,cv=0,en=0; int bi=0; double bf=-1e9, te;
      for(p=0;p<POP;p++){ sh+=pop[p].shared; mw+=meanwidth(&pop[p]); mx+=maxwidth(&pop[p]);
                          cv+=coverage(&pop[p]); en+=param_energy(&pop[p]);
                          if(fit[p]>bf){bf=fit[p];bi=p;} }
      te=run_net(&pop[bi],seed+999u,1);
      out[0]+=sh/POP; out[1]+=mw/POP; out[2]+=mx/POP; out[3]+=cv/POP; out[4]+=te; out[5]+=en/POP;
      /* PROTOCOL item 2: coverage is the free variable of the OLD objective. Emit it per run so the
       * arm effect is read as the increment over the pooled acc(coverage) curve, never raw. */
      if(g_raw) printf("RAW %d %d %d %.4f %.4f %.4f %.4f %.6f %.4f\n",
                       g_arm, g_gens, sd, coverage(&pop[bi]), meanwidth(&pop[bi]),
                       (double)maxwidth(&pop[bi]), (double)pop[bi].shared, param_energy(&pop[bi]), te); }
}

int main(void)
{
    int seeds=envint("SEEDS",16), sd, k;
    const char *nm[3]={"0 no-rewire","1 slide-1","2 rewire-rand"};
    g_gens=envint("GENS",150); g_lambda=envdbl("LAMBDA",1.0);
    g_pslide=envdbl("PSLIDE",0.15); g_pgrow=envdbl("PGROW",0.10); g_pshare=envdbl("PSHARE",0.05);
    g_raw=envint("RAW",0);
    printf("REWIRING vs ADD/REMOVE under a CONTINUOUS objective (acc - %.3f*energy)\n", g_lambda);
    printf("N=%d hidden=%d K=%d, %d seeds x %d gens, PSLIDE=%.3f\n", N, H, K, seeds, g_gens, g_pslide);
    printf("compact conv = width -> K, coverage -> 1, shared-frac -> 1\n\n");
    if(!sensitivity_check(seeds >= 12 ? 12 : seeds)) return 2;
    printf("\n  %-14s  shared-frac  mean-w  max-w  coverage  energy   test\n", "arm");
    for(g_arm=0; g_arm<3; g_arm++){
        double o[6]={0,0,0,0,0,0};
        for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*131+1)); run_ga((uint32_t)(sd*7+1), o, sd); }
        for(k=0;k<6;k++) o[k]/=seeds;
        printf("  %-14s   %.3f       %.2f    %.2f   %.3f    %.4f   %.3f\n",
               nm[g_arm], o[0], o[1], o[2], o[3], o[5], o[4]);
    }
    printf("\nGENS=0 reruns this as the no-evolution control: any arm difference that survives there\n");
    printf("is an artifact of the seed and the operator, not of selection.\n");
    return 0;
}
