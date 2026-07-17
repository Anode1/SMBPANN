/* emerge_2d_chan.c -- 2-D reuse vs recombination on the CHANNEL axis: do two ops need two channels?
 *
 * The 1-D two-op result (Section 10) lived on the depth/dilation axis. In 2-D "two different operations"
 * means two different FEATURE DETECTORS -- a horizontal filter and a vertical filter -- and a single
 * conv channel can only be one orientation at a time. So the 2-D analogue of reuse-vs-recombination is
 * CHANNEL WIDTH: C=1 is "one feature type reused", C>=2 is "diverse features". This builds a
 * multi-channel 2-D conv engine (1 -> C channels, per-channel global max-pool, linear readout) and asks,
 * on the stable orientation foundation (emerge_2d_orient), whether a task needing BOTH orientations
 * forces C>=2.
 *
 *   ONE-OP task: horizontal vs vertical bar (orientation discrimination). One filter suffices, so C=1
 *     should already solve and extra channels add nothing.
 *   TWO-OP task: positive = a horizontal bar AND a vertical bar both present (different locations);
 *     negative = exactly one of them. A single channel tuned to one orientation cannot tell "both" from
 *     "only my orientation" -> it caps near 0.75; two channels (H-detector + V-detector, AND'd by the
 *     readout) should separate them. So C=1 fails and C=2 solves -- the crossover, in genuine 2-D.
 *
 * Reports accuracy vs channel count C for both tasks, over many seeds with spread.
 *
 * Finding (8 seeds, balanced classes): confirmed. ONE-OP is flat-high (C=1 0.98 -> C=4 1.00): one
 * channel already suffices, extra channels add nothing. TWO-OP shows the crossover: C=1 caps at 0.73
 * (~ the 0.75 ceiling a single orientation detector can reach), C=2 jumps to 0.89 (crosses target),
 * C=4 robust at 0.97. So two different operations require >=2 feature channels, and the requirement is
 * specific to the two-op task. NOTE (a caught confound): classes MUST be balanced 50/50 -- an earlier
 * 1/3-positive / 2/3-negative split made C=1 and C=2 both sit at ~0.67, the majority-class baseline,
 * masking the effect entirely. Balanced classes reveal the real 0.75 cap and the C=2 jump.
 * Self-contained C99. Build: make emerge_2d_chan
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define G     12
#define KK    3
#define SP    (G-KK+1)      /* = 10, spatial side after one valid conv */
#define CMAX  6
#define NTR   192
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
static int    g_task = 0;              /* 0 = ONE-OP (orientation), 1 = TWO-OP (both orientations) */

static double Xtr[NTR][G][G], Xte[NTE][G][G];
static int    ytr[NTR], yte[NTE];

static void put_h(double g[G][G],int r,int c){ g[r][c]+=AMP; g[r][c+1]+=AMP; g[r][c+2]+=AMP; }
static void put_v(double g[G][G],int r,int c){ g[r][c]+=AMP; g[r+1][c]+=AMP; g[r+2][c]+=AMP; }

static void gen(double X[][G][G],int*y,int n)
{
    int s,i,j;
    for(s=0;s<n;s++){
        for(i=0;i<G;i++) for(j=0;j<G;j++) X[s][i][j]=runif();
        if(g_task==0){                                  /* ONE-OP: orientation of a single bar */
            y[s]=(int)(r32()&1u);
            { int r=(int)urand(G-2), c=(int)urand(G-2);
              if(y[s]) put_h(X[s],r,c); else put_v(X[s],r,c); }
        } else {                                        /* TWO-OP: both orientations vs one (BALANCED) */
            y[s]=(int)(r32()&1u);                         /* 50% positive (both), 50% negative (one) */
            { int rh=(int)urand(G-2), ch=(int)urand(G-2), rv=(int)urand(G-2), cv=(int)urand(G-2);
              if(y[s]){ put_h(X[s],rh,ch); put_v(X[s],rv,cv); }        /* positive: both orientations */
              else if(r32()&1u){ put_h(X[s],rh,ch); }                  /* negative: H only ... */
              else { put_v(X[s],rv,cv); } }                            /* ... or V only */
        }
    }
}
static void new_task(uint32_t seed){ rseed(seed); gen(Xtr,ytr,NTR); gen(Xte,yte,NTE); }

/* single 2-D conv layer 1 -> C channels, per-channel global max-pool, linear readout. train, return acc. */
static double train_eval(int C, uint32_t seed)
{
    static double w[CMAX][KK][KK], bc[CMAX], rd[CMAX], rb;
    static double out[CMAX][SP][SP];
    int e,s,i,j,co,r,c, cc=0;
    wseed(seed);
    for(co=0;co<C;co++){ for(i=0;i<KK;i++) for(j=0;j<KK;j++) w[co][i][j]=0.3*wunif(); bc[co]=0; rd[co]=0.3*wunif(); }
    rb=0;
    for(e=0;e<TEPOCHS;e++) for(s=0;s<NTR;s++){
        double p[CMAX]; int rm[CMAX],cm[CMAX];
        double pre_o=rb, o, dout, dp[CMAX];
        for(co=0;co<C;co++){ double best=-1e9; int br=0,bc2=0;
            for(r=0;r<SP;r++) for(c=0;c<SP;c++){ double pre=bc[co];
                for(i=0;i<KK;i++) for(j=0;j<KK;j++) pre+=w[co][i][j]*Xtr[s][r+i][c+j];
                double v=tanh(pre); out[co][r][c]=v; if(v>best){best=v;br=r;bc2=c;} }
            p[co]=best; rm[co]=br; cm[co]=bc2; pre_o+=rd[co]*best; }
        o=1.0/(1.0+exp(-pre_o)); dout=(o-ytr[s])*o*(1.0-o);
        for(co=0;co<C;co++) dp[co]=dout*rd[co];        /* backprop to pooled value (pre-update rd) */
        for(co=0;co<C;co++){ rd[co]-=LR*dout*p[co]; }
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
    int seeds=envint("SEEDS",24), sd, t, C;
    int chans[4]={1,2,3,4};
    const char *tn[2]={"ONE-OP (orientation of one bar)","TWO-OP (both orientations present)"};
    g_target=envdbl("TARGET",0.85);
    TEPOCHS=envint("TEPOCHS",TEPOCHS); AMP=envdbl("AMP",AMP); RESTARTS=envint("RESTARTS",RESTARTS);
    printf("2-D CHANNELS: do two different operations (orientations) require two feature CHANNELS?\n");
    printf("G=%dx%d k=%d, single conv layer 1->C, per-channel max-pool, %d seeds x %d restarts, AMP=%.1f\n\n",
           G, G, KK, seeds, RESTARTS, AMP);
    for(t=0;t<2;t++){ g_task=t;
        printf("  %-34s | acc at C=1,2,3,4        | spread(SD)\n", tn[t]);
        double acc[4], sdv[4];
        for(C=0;C<4;C++){ double sum=0,sm2=0; int nv=0;
            for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*911u+(unsigned)t*53u+1u));
                double v=best_eval(chans[C],(uint32_t)(sd*7u+1u)); sum+=v; sm2+=v*v; nv++; }
            acc[C]=sum/nv; sdv[C]= nv>1? sqrt((sm2-sum*sum/nv)/(nv-1)) : 0; }
        printf("  %-34s |", "");
        for(C=0;C<4;C++) printf(" %.3f", acc[C]);
        printf("   |");
        for(C=0;C<4;C++) printf(" %.02f", sdv[C]);
        printf("\n\n");
    }
    printf("crossover if ONE-OP is flat-high across C (one channel enough) but TWO-OP jumps C=1->C=2\n");
    printf("(one orientation cannot see both; two channels can) -- reuse vs recombination on the channel axis.\n");
    return 0;
}
