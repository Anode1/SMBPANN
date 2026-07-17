/* emerge_2d_grow.c -- does the emergent CHANNEL count track the number of operations? (2-D)
 *
 * Section 13 showed a two-operation 2-D task needs >=2 feature channels. This asks the general
 * question and makes it a staged/growth searcher (the channel-axis analogue of Section 11's
 * clone-then-recombine, and the width analogue of Section 8's emergent-depth staircase): start at one
 * channel and GROW only when the task is not solved, and see whether the selected channel count C*
 * lands on the number of DISTINCT operations the task requires.
 *
 * K-operation task: K distinct oriented bars (K=1 uses horizontal; K=2 adds vertical; K=3 adds a main
 * diagonal). Positive = all K present; negative = all K but one (a random one dropped) -- balanced
 * 50/50. To classify, a detector must confirm EVERY one of the K is present, so it must cover all K
 * orientations; with fewer than K channels it is blind to at least one and cannot catch the negative
 * that drops that one. So the minimal solving width is C = K. Growth searcher: raise C from 1 until the
 * target is met; report the selected C* per K, plus accuracy vs C. Expectation: C* tracks K (1,2,3).
 *
 * FINDING (24 seeds, fair mean over restarts) -- PARTIAL, honest: the clean "C* = K" staircase does NOT
 * hold. Direction is right (more ops -> more channels: C* ~ 1.2, 3.2, 5.0 for K=1,2,3) but the equality
 * fails and is worse than a best-of run suggested: K=2 already overshoots (C*~3, not 2), and K=3 BREAKS
 * -- sub-target at every channel count (C=5 = 0.79), only 2/24 seeds solve, and an independent heavy-
 * compute rerun (2.5x epochs, 2x amplitude) leaves it sub-target too, so it is STRUCTURAL not under-
 * training. Reasons: gradient descent does not cleanly assign one channel per operation (redundancy is
 * needed to cover all K detectors) and a K-way conjunction compounds per-channel error (0.95^3 ~ 0.86,
 * at the edge). Section 13's core (one channel is not enough for two operations) stands; this clean
 * generalization does not.
 * Self-contained C99 (multi-channel engine from emerge_2d_chan). Build: make emerge_2d_grow
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define G     12
#define KK    3
#define SP    (G-KK+1)
#define CMAX  5
#define NTR   256
#define NTE   1200
#define LR    0.05

static uint32_t rs=1u; static uint32_t r32(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;rs=x;return x;}
static void rseed(uint32_t s){rs=s?s:1u;} static double runif(void){return (double)r32()/4294967296.0*2.0-1.0;}
static uint32_t urand(uint32_t n){ return r32()%n; }
static uint32_t wr=1u; static uint32_t wr32(void){uint32_t x=wr;x^=x<<13;x^=x>>17;x^=x<<5;wr=x;return x;}
static void wseed(uint32_t s){wr=s?s:1u;} static double wunif(void){return (double)wr32()/4294967296.0*2.0-1.0;}

static int envint(const char*k,int d){const char*e=getenv(k);return e&&*e?atoi(e):d;}
static double envdbl(const char*k,double d){const char*e=getenv(k);return e&&*e?atof(e):d;}
static int    TEPOCHS = 200;
static double AMP     = 3.0;
static int    RESTARTS = 3;
static double g_target = 0.85;
static int    g_nops = 1;              /* K: number of distinct orientations required */

static double Xtr[NTR][G][G], Xte[NTE][G][G];
static int    ytr[NTR], yte[NTE];

/* plant orientation o (0=H,1=V,2=D) as a 3-cell bar at a random location */
static void put_orient(double g[G][G],int o)
{ int r=(int)urand(G-2), c=(int)urand(G-2);
  if(o==0){ g[r][c]+=AMP; g[r][c+1]+=AMP; g[r][c+2]+=AMP; }          /* horizontal */
  else if(o==1){ g[r][c]+=AMP; g[r+1][c]+=AMP; g[r+2][c]+=AMP; }     /* vertical */
  else { g[r][c]+=AMP; g[r+1][c+1]+=AMP; g[r+2][c+2]+=AMP; } }       /* main diagonal */

static void gen(double X[][G][G],int*y,int n)
{
    int s,i,j,o,drop;
    for(s=0;s<n;s++){
        for(i=0;i<G;i++) for(j=0;j<G;j++) X[s][i][j]=runif();
        y[s]=(int)(r32()&1u);
        drop = y[s] ? -1 : (int)urand((uint32_t)g_nops);   /* negative drops exactly one of the K */
        for(o=0;o<g_nops;o++) if(o!=drop) put_orient(X[s],o);
    }
}
static void new_task(uint32_t seed){ rseed(seed); gen(Xtr,ytr,NTR); gen(Xte,yte,NTE); }

/* single 2-D conv layer 1 -> C channels, per-channel global max-pool, linear readout. */
static double train_eval(int C, uint32_t seed)
{
    static double w[CMAX][KK][KK], bc[CMAX], rd[CMAX], rb;
    static double out[CMAX][SP][SP];
    int e,s,i,j,co,r,c, cc=0;
    wseed(seed);
    for(co=0;co<C;co++){ for(i=0;i<KK;i++) for(j=0;j<KK;j++) w[co][i][j]=0.3*wunif(); bc[co]=0; rd[co]=0.3*wunif(); }
    rb=0;
    for(e=0;e<TEPOCHS;e++) for(s=0;s<NTR;s++){
        double p[CMAX]; int rm[CMAX],cm[CMAX]; double pre_o=rb, o, dout, dp[CMAX];
        for(co=0;co<C;co++){ double best=-1e9; int br=0,bc2=0;
            for(r=0;r<SP;r++) for(c=0;c<SP;c++){ double pre=bc[co];
                for(i=0;i<KK;i++) for(j=0;j<KK;j++) pre+=w[co][i][j]*Xtr[s][r+i][c+j];
                double v=tanh(pre); out[co][r][c]=v; if(v>best){best=v;br=r;bc2=c;} }
            p[co]=best; rm[co]=br; cm[co]=bc2; pre_o+=rd[co]*best; }
        o=1.0/(1.0+exp(-pre_o)); dout=(o-ytr[s])*o*(1.0-o);
        for(co=0;co<C;co++) dp[co]=dout*rd[co];
        for(co=0;co<C;co++) rd[co]-=LR*dout*p[co];
        rb-=LR*dout;
        for(co=0;co<C;co++){ double v=out[co][rm[co]][cm[co]]; double dpre=dp[co]*(1.0-v*v);
            bc[co]-=LR*dpre;
            for(i=0;i<KK;i++) for(j=0;j<KK;j++) w[co][i][j]-=LR*dpre*Xtr[s][rm[co]+i][cm[co]+j]; }
    }
    for(s=0;s<NTE;s++){ double pre_o=rb;
        for(co=0;co<C;co++){ double best=-1e9;
            for(r=0;r<SP;r++) for(c=0;c<SP;c++){ double pre=bc[co];
                for(i=0;i<KK;i++) for(j=0;j<KK;j++) pre+=w[co][i][j]*Xte[s][r+i][c+j];
                double v=tanh(pre); if(v>best)best=v; }
            pre_o+=rd[co]*best; }
        { double o=1.0/(1.0+exp(-pre_o)); if((o>0.5)==(yte[s]==1)) cc++; }
    }
    return (double)cc/NTE;
}
/* mean over restarts (fair -- no optimistic best-of). */
static double best_eval(int C, uint32_t seed)
{ int r; double s=0; for(r=0;r<RESTARTS;r++){ double a=train_eval(C,seed*131u+(uint32_t)r*97u+1u); s+=a; } return s/RESTARTS; }

int main(void)
{
    int seeds=envint("SEEDS",24), sd, K, C;
    g_target=envdbl("TARGET",0.85);
    TEPOCHS=envint("TEPOCHS",TEPOCHS); AMP=envdbl("AMP",AMP); RESTARTS=envint("RESTARTS",RESTARTS);
    printf("2-D EMERGENT WIDTH: does the grown channel count C* track the number of operations K?\n");
    printf("G=%dx%d k=%d, single conv 1->C + per-channel max-pool, %d seeds x %d restarts, AMP=%.1f\n", G, G, KK, seeds, RESTARTS, AMP);
    printf("K-op task: all K oriented bars present (pos) vs one dropped (neg), balanced. minimal width C=K.\n\n");
    printf("  K ops | accuracy at C=1..%d           | selected C* (mean, first C>=target)\n", CMAX);
    for(K=1;K<=3;K++){ g_nops=K;
        double acc[CMAX+1]; double cstar_sum=0; int cstar_n=0;
        for(C=1;C<=CMAX;C++){ double sum=0; int nv=0;
            for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*911u+(unsigned)K*53u+1u));
                sum+=best_eval(C,(uint32_t)(sd*7u+1u)); nv++; } acc[C]=sum/nv; }
        for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*911u+(unsigned)K*53u+1u));
            int cs=-1; for(C=1;C<=CMAX;C++){ if(best_eval(C,(uint32_t)(sd*7u+1u))>=g_target){ cs=C; break; } }
            if(cs>0){ cstar_sum+=cs; cstar_n++; } }
        printf("  K=%d   |", K);
        for(C=1;C<=CMAX;C++) printf(" %.3f", acc[C]);
        printf("   | C*=%.2f (%d/%d solved)\n", cstar_n? cstar_sum/cstar_n : 0.0, cstar_n, seeds);
    }
    printf("\nemergent width tracks operation count if C* rises 1,2,3 with K -- the network discovers how\n");
    printf("many distinct feature detectors the task needs, by growing channels only when one is not enough.\n");
    return 0;
}
