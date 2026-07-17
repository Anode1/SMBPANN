/* emerge_baseline.c -- FAIR-BASELINE TEST: does weight-sharing still win against a fully-trained,
 *                      oracle-supervised locally-connected baseline? (reviewers flagged the weak baseline)
 *
 * The scale/translate/develop experiments compare SHARED (one filter tiled = a convolution) against
 * INDEP (a filter per position). But INDEP was trained through a max-pool, so only the pool-WINNER
 * position updates each step; positions that rarely win stay at random init and inject noise at test.
 * Two independent reviewers noted this makes INDEP a weak "search-from-scratch" strawman, and that a
 * FAIR fully-trained baseline might narrow the gap. This tests exactly that.
 *
 * Three arms, same translation-invariant motif task (M vs M' planted at a random position), across N:
 *   develop      = SHARED, trained through max-pool (the paper's arm).
 *   scratch      = INDEP,  trained through max-pool -- only the winner updates (the paper's weak arm).
 *   scratch_fair = INDEP,  ORACLE-trained: for every example we KNOW where the motif is, so every
 *                  position's filter is trained fully every step toward its correct per-window target
 *                  (fire + iff its window is M, - for M' or noise). This is the strongest possible
 *                  locally-connected baseline -- full gradient, every position, plus the motif location
 *                  handed to it (which the task does not provide). Test is honest for all arms (max-pool,
 *                  no oracle).
 *
 * If develop still beats scratch_fair, and the gap still grows with N, the effect is real weight-sharing
 * DATA EFFICIENCY (the shared filter learns the motif from all positions; each per-position filter sees
 * only ~ntr/P motif-examples, however it is trained) -- not the winner-only training rule. If the fair
 * baseline closes the gap, the headline was a training artifact and we say so.
 *
 * FINDING (60 seeds, mean +/- std): the effect SURVIVES. The oracle-supervised, fully-trained baseline
 * (scratch_FAIR) does help INDEP a little (N=16: 0.71 -> 0.81), but at scale it still collapses toward
 * chance (0.55 at N=128) and weight-sharing still beats it by +0.14 (N=16) up to +0.38 (N=128). So the
 * effect is REAL weight-sharing DATA EFFICIENCY, not the winner-only training artifact. IMPORTANT
 * REFRAME: this also confirms the reviewers -- the phenomenon is weight-SHARING (tying) vs untied, i.e.
 * LeCun's data-efficiency argument, NOT "reuse vs architecture search". No search happens in these arms.
 * Self-contained C99. Build: make emerge_baseline
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define K     3
#define NMAX  128
#define PMAX  (NMAX-K+1)
#define NTRMAX 64
#define NTE   2000
#define TEPOCHS 500
#define LR     0.05
#define LRF    0.02          /* fair arm does P updates/example -> smaller step */
#define AMP    2.6
#define TGT    0.8           /* per-window target magnitude for oracle training */
#define RESTARTS 4

static uint32_t rs=1u; static uint32_t r32(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;rs=x;return x;}
static void rseed(uint32_t s){rs=s?s:1u;} static double runif(void){return (double)r32()/4294967296.0*2.0-1.0;}
static uint32_t urand(uint32_t n){ return r32()%n; }
static uint32_t wr=1u; static uint32_t wr32(void){uint32_t x=wr;x^=x<<13;x^=x>>17;x^=x<<5;wr=x;return x;}
static void wseed(uint32_t s){wr=s?s:1u;} static double wunif(void){return (double)wr32()/4294967296.0*2.0-1.0;}

static int envint(const char*k,int d){const char*e=getenv(k);return e&&*e?atoi(e):d;}
static int    g_n=16, g_p=14, g_ntr=32;

static double Mpos[K], Mneg[K];
static double Xtr[NTRMAX][NMAX], Xte[NTE][NMAX];
static int    ytr[NTRMAX], yte[NTE], ppos[NTRMAX];   /* ppos = planted motif position (oracle) */

static void gen(double X[][NMAX],int*y,int*pp,int n)
{ int s,i,p,k; for(s=0;s<n;s++){ for(i=0;i<g_n;i++) X[s][i]=runif(); y[s]=(int)(r32()&1u); p=(int)urand((uint32_t)g_p);
    if(pp) pp[s]=p;
    for(k=0;k<K;k++) X[s][p+k]+=AMP*(y[s]?Mpos[k]:Mneg[k]); } }
static void new_task(uint32_t seed){ int k; rseed(seed);
    for(k=0;k<K;k++){ Mpos[k]=runif()>0?1:-1; Mneg[k]=runif()>0?1:-1; }
    if(Mpos[0]==Mneg[0]&&Mpos[1]==Mneg[1]&&Mpos[2]==Mneg[2]) Mneg[0]=-Mpos[0];
    gen(Xtr,ytr,ppos,g_ntr); gen(Xte,yte,NULL,NTE); }

/* SHARED through max-pool (the paper's develop arm). */
static double train_shared(uint32_t seed)
{
    double w[K], b=0, rsc, rb=0; int e,s,k,p, c=0;
    wseed(seed); for(k=0;k<K;k++) w[k]=0.3*wunif(); rsc=0.3*wunif();
    for(e=0;e<TEPOCHS;e++) for(s=0;s<g_ntr;s++){ double best=-1e9; int pm=0;
        for(p=0;p<g_p;p++){ double z=b; for(k=0;k<K;k++) z+=w[k]*Xtr[s][p+k]; double v=tanh(z); if(v>best){best=v;pm=p;} }
        double o=1.0/(1.0+exp(-(rsc*best+rb))), dout=(o-ytr[s])*o*(1.0-o), dpre=dout*rsc*(1.0-best*best);
        rsc-=LR*dout*best; rb-=LR*dout; b-=LR*dpre; for(k=0;k<K;k++) w[k]-=LR*dpre*Xtr[s][pm+k]; }
    for(s=0;s<NTE;s++){ double best=-1e9;
        for(p=0;p<g_p;p++){ double z=b; for(k=0;k<K;k++) z+=w[k]*Xte[s][p+k]; double v=tanh(z); if(v>best)best=v; }
        double o=1.0/(1.0+exp(-(rsc*best+rb))); if((o>0.5)==(yte[s]==1)) c++; }
    return (double)c/NTE;
}
/* INDEP through max-pool -- only the winner trains (the paper's weak scratch arm). */
static double train_indep(uint32_t seed)
{
    static double w[PMAX][K]; double b[PMAX], rsc, rb=0; int e,s,k,p, c=0;
    wseed(seed); for(p=0;p<g_p;p++){ for(k=0;k<K;k++) w[p][k]=0.3*wunif(); b[p]=0; } rsc=0.3*wunif();
    for(e=0;e<TEPOCHS;e++) for(s=0;s<g_ntr;s++){ double best=-1e9; int pm=0;
        for(p=0;p<g_p;p++){ double z=b[p]; for(k=0;k<K;k++) z+=w[p][k]*Xtr[s][p+k]; double v=tanh(z); if(v>best){best=v;pm=p;} }
        double o=1.0/(1.0+exp(-(rsc*best+rb))), dout=(o-ytr[s])*o*(1.0-o), dpre=dout*rsc*(1.0-best*best);
        rsc-=LR*dout*best; rb-=LR*dout; b[pm]-=LR*dpre; for(k=0;k<K;k++) w[pm][k]-=LR*dpre*Xtr[s][pm+k]; }
    for(s=0;s<NTE;s++){ double best=-1e9;
        for(p=0;p<g_p;p++){ double z=b[p]; for(k=0;k<K;k++) z+=w[p][k]*Xte[s][p+k]; double v=tanh(z); if(v>best)best=v; }
        double o=1.0/(1.0+exp(-(rsc*best+rb))); if((o>0.5)==(yte[s]==1)) c++; }
    return (double)c/NTE;
}
/* INDEP oracle-trained: EVERY position's filter trained fully every step toward its per-window target
 * (fire + iff its window is the M motif). The strongest possible locally-connected baseline. */
static double train_indep_fair(uint32_t seed)
{
    static double w[PMAX][K]; double b[PMAX]; int e,s,k,p, c=0;
    wseed(seed); for(p=0;p<g_p;p++){ for(k=0;k<K;k++) w[p][k]=0.3*wunif(); b[p]=0; }
    for(e=0;e<TEPOCHS;e++) for(s=0;s<g_ntr;s++){
        for(p=0;p<g_p;p++){ int isM = (p==ppos[s] && ytr[s]==1); double target = isM? TGT : -TGT;
            double z=b[p]; for(k=0;k<K;k++) z+=w[p][k]*Xtr[s][p+k]; double v=tanh(z);
            double dpre=(v-target)*(1.0-v*v);                 /* MSE on the tanh output */
            b[p]-=LRF*dpre; for(k=0;k<K;k++) w[p][k]-=LRF*dpre*Xtr[s][p+k]; }
    }
    for(s=0;s<NTE;s++){ double best=-1e9;
        for(p=0;p<g_p;p++){ double z=b[p]; for(k=0;k<K;k++) z+=w[p][k]*Xte[s][p+k]; double v=tanh(z); if(v>best)best=v; }
        if((best>0)==(yte[s]==1)) c++; }                       /* filter fires + for M -> honest max-pool */
    return (double)c/NTE;
}
static double mean_arm(int arm, uint32_t seed)
{ int r; double s=0; for(r=0;r<RESTARTS;r++){ uint32_t sd=seed*131u+(uint32_t)r*97u+1u;
    double a = arm==0? train_shared(sd) : arm==1? train_indep(sd) : train_indep_fair(sd); s+=a; } return s/RESTARTS; }

int main(void)
{
    int seeds=envint("SEEDS",60), sd, t;
    int Ns[4]={16,48,96,128};
    g_ntr=envint("NTR",32);
    printf("FAIR-BASELINE TEST: does weight-sharing beat a FULLY-TRAINED, oracle-supervised INDEP baseline?\n");
    printf("translation-invariant motif task, %d train examples, %d seeds x %d restarts (mean, +/- std).\n\n", g_ntr, seeds, RESTARTS);
    printf("  N   | develop (shared) | scratch (indep, winner-only) | scratch_FAIR (indep, oracle full) | develop - FAIR\n");
    for(t=0;t<4;t++){ g_n=Ns[t]; g_p=g_n-K+1;
        double a0=0,a1=0,a2=0, s0=0,s1=0,s2=0;
        for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*911u+(unsigned)g_n*17u+1u));
            double v0=mean_arm(0,(uint32_t)(sd*7u+1u)), v1=mean_arm(1,(uint32_t)(sd*7u+1u)), v2=mean_arm(2,(uint32_t)(sd*7u+1u));
            a0+=v0; s0+=v0*v0; a1+=v1; s1+=v1*v1; a2+=v2; s2+=v2*v2; }
        { double m0=a0/seeds,m1=a1/seeds,m2=a2/seeds;
          double d0=seeds>1?sqrt((s0-a0*a0/seeds)/(seeds-1)):0, d1=seeds>1?sqrt((s1-a1*a1/seeds)/(seeds-1)):0, d2=seeds>1?sqrt((s2-a2*a2/seeds)/(seeds-1)):0;
          printf("  %-3d | %.3f +/- %.3f  | %.3f +/- %.3f              | %.3f +/- %.3f                     | %+.3f\n",
                 g_n, m0,d0, m1,d1, m2,d2, m0-m2); }
    }
    printf("\nverdict: if develop still beats scratch_FAIR and the gap grows with N, the effect is REAL weight-\n");
    printf("sharing data efficiency; if scratch_FAIR closes the gap, the headline was a weak-baseline artifact.\n");
    return 0;
}
