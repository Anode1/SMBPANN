/* emerge_2d_deep.c -- does a COMBINING LAYER recover the K=3 failure of emerge_2d_grow?
 *
 * emerge_2d_grow found the "channels = operations" staircase breaks at K=3: with a single conv layer
 * (1->C channels, per-channel max-pool) and a LINEAR readout, a 3-operation task is not solved at C=3
 * (0.75), needing 4-5 redundant channels. Two suspects: (a) the C channels do not cleanly specialize to
 * the K orientations, and (b) a linear readout over imperfect detections compounds error. A hidden
 * COMBINING layer on the pooled channel vector addresses both -- it can pool redundant/partial channels
 * into robust per-orientation evidence and compute a robust K-way conjunction. So the clean test of
 * whether the failure is "the head" or "the channels" is: add a small MLP head and see if K=3 solves at
 * C=3.
 *
 * Architecture: conv 1->C (per-channel global max-pool -> p[C]); HEAD = linear (p -> out) OR an MLP
 * (p -> NH tanh hidden -> out). Same K-operation task as emerge_2d_grow (K oriented bars all present vs
 * one dropped, balanced). We compare LINEAR vs MLP heads at K=2,3 across C, asking specifically whether
 * the MLP head solves K=3 at C=3 where the linear head could not.
 *
 * FINDING (8 seeds): the combining layer recovers K=2 but NOT K=3 -- it splits the blame. K=2 at C=2
 * goes 0.845 (linear) -> 0.951 (MLP): the K=2 marginality WAS the linear head. But K=3 at C=3 goes only
 * 0.759 -> 0.794, still under target: the MLP helps yet cannot rescue it, and K=3 needs C>=5 even with
 * the MLP (0.870). So the K=3 failure is the CONV CHANNELS, not the head -- three conv channels cannot
 * pack three clean orientation detectors, and a nonlinear combiner cannot compensate. Blame: head for
 * K=2, channels for K=3. Section 14's structural finding stands. Self-contained C99. make emerge_2d_deep
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
#define NH    8              /* hidden units in the MLP head */
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
static int    TEPOCHS = 250;
static double AMP     = 3.0;
static int    RESTARTS = 3;
static double g_target = 0.85;
static int    g_nops = 1;

static double Xtr[NTR][G][G], Xte[NTE][G][G];
static int    ytr[NTR], yte[NTE];

static void put_orient(double g[G][G],int o)
{ int r=(int)urand(G-2), c=(int)urand(G-2);
  if(o==0){ g[r][c]+=AMP; g[r][c+1]+=AMP; g[r][c+2]+=AMP; }
  else if(o==1){ g[r][c]+=AMP; g[r+1][c]+=AMP; g[r+2][c]+=AMP; }
  else { g[r][c]+=AMP; g[r+1][c+1]+=AMP; g[r+2][c+2]+=AMP; } }
static void gen(double X[][G][G],int*y,int n)
{
    int s,i,j,o,drop;
    for(s=0;s<n;s++){
        for(i=0;i<G;i++) for(j=0;j<G;j++) X[s][i][j]=runif();
        y[s]=(int)(r32()&1u);
        drop = y[s] ? -1 : (int)urand((uint32_t)g_nops);
        for(o=0;o<g_nops;o++) if(o!=drop) put_orient(X[s],o);
    }
}
static void new_task(uint32_t seed){ rseed(seed); gen(Xtr,ytr,NTR); gen(Xte,yte,NTE); }

/* conv 1->C + per-channel max-pool -> p[C]; head: mlp!=0 uses NH tanh hidden, else linear. */
static double train_eval(int C, int mlp, uint32_t seed)
{
    static double w[CMAX][KK][KK], bc[CMAX];
    static double rd[CMAX], rb;                 /* linear head */
    static double V[NH][CMAX], bh[NH], wo[NH], bo;  /* mlp head */
    static double out[CMAX][SP][SP];
    int e,s,i,j,co,r,c,h, cc=0;
    wseed(seed);
    for(co=0;co<C;co++){ for(i=0;i<KK;i++) for(j=0;j<KK;j++) w[co][i][j]=0.3*wunif(); bc[co]=0; rd[co]=0.3*wunif(); }
    rb=0; bo=0;
    for(h=0;h<NH;h++){ for(co=0;co<CMAX;co++) V[h][co]=0.3*wunif(); bh[h]=0; wo[h]=0.3*wunif(); }
    for(e=0;e<TEPOCHS;e++) for(s=0;s<NTR;s++){
        double p[CMAX]; int rm[CMAX],cm[CMAX]; double dp[CMAX];
        for(co=0;co<C;co++){ double best=-1e9; int br=0,bc2=0;
            for(r=0;r<SP;r++) for(c=0;c<SP;c++){ double pre=bc[co];
                for(i=0;i<KK;i++) for(j=0;j<KK;j++) pre+=w[co][i][j]*Xtr[s][r+i][c+j];
                double v=tanh(pre); out[co][r][c]=v; if(v>best){best=v;br=r;bc2=c;} }
            p[co]=best; rm[co]=br; cm[co]=bc2; }
        for(co=0;co<C;co++) dp[co]=0;
        { double o,dout;
          if(!mlp){ double zo=rb; for(co=0;co<C;co++) zo+=rd[co]*p[co];
              o=1.0/(1.0+exp(-zo)); dout=(o-ytr[s])*o*(1.0-o);
              for(co=0;co<C;co++) dp[co]=dout*rd[co];
              for(co=0;co<C;co++) rd[co]-=LR*dout*p[co];
              rb-=LR*dout;
          } else { double hv[NH], zo=bo;
              for(h=0;h<NH;h++){ double z=bh[h]; for(co=0;co<C;co++) z+=V[h][co]*p[co]; hv[h]=tanh(z); zo+=wo[h]*hv[h]; }
              o=1.0/(1.0+exp(-zo)); dout=(o-ytr[s])*o*(1.0-o);
              for(h=0;h<NH;h++){ double dh=dout*wo[h]; wo[h]-=LR*dout*hv[h];
                  double dz=dh*(1.0-hv[h]*hv[h]); bh[h]-=LR*dz;
                  for(co=0;co<C;co++){ dp[co]+=dz*V[h][co]; V[h][co]-=LR*dz*p[co]; } }
              bo-=LR*dout;
          }
        }
        for(co=0;co<C;co++){ double v=out[co][rm[co]][cm[co]]; double dpre=dp[co]*(1.0-v*v);
            bc[co]-=LR*dpre;
            for(i=0;i<KK;i++) for(j=0;j<KK;j++) w[co][i][j]-=LR*dpre*Xtr[s][rm[co]+i][cm[co]+j]; }
    }
    for(s=0;s<NTE;s++){ double p[CMAX];
        for(co=0;co<C;co++){ double best=-1e9;
            for(r=0;r<SP;r++) for(c=0;c<SP;c++){ double pre=bc[co];
                for(i=0;i<KK;i++) for(j=0;j<KK;j++) pre+=w[co][i][j]*Xte[s][r+i][c+j];
                double v=tanh(pre); if(v>best)best=v; }
            p[co]=best; }
        { double o;
          if(!mlp){ double zo=rb; for(co=0;co<C;co++) zo+=rd[co]*p[co]; o=1.0/(1.0+exp(-zo)); }
          else { double zo=bo; for(h=0;h<NH;h++){ double z=bh[h]; for(co=0;co<C;co++) z+=V[h][co]*p[co]; zo+=wo[h]*tanh(z); } o=1.0/(1.0+exp(-zo)); }
          if((o>0.5)==(yte[s]==1)) cc++; }
    }
    return (double)cc/NTE;
}
static double best_eval(int C, int mlp, uint32_t seed)
{ int r; double best=0; for(r=0;r<RESTARTS;r++){ double a=train_eval(C,mlp,seed*131u+(uint32_t)r*97u+1u); if(a>best) best=a; } return best; }

int main(void)
{
    int seeds=envint("SEEDS",8), sd, K, C, mlp;
    g_target=envdbl("TARGET",0.85);
    TEPOCHS=envint("TEPOCHS",TEPOCHS); AMP=envdbl("AMP",AMP); RESTARTS=envint("RESTARTS",RESTARTS);
    printf("2-D DEPTH-IN-HEAD: does a combining layer recover the K=3 failure? (conv 1->C, then head)\n");
    printf("G=%dx%d k=%d, %d seeds x %d restarts, AMP=%.1f, MLP head has %d hidden units.\n", G, G, KK, seeds, RESTARTS, AMP, NH);
    printf("K oriented bars all present vs one dropped, balanced. does MLP head solve K=3 at C=3?\n\n");
    printf("  head    | K | accuracy at C=1..%d\n", CMAX);
    for(mlp=0;mlp<2;mlp++) for(K=2;K<=3;K++){ g_nops=K;
        printf("  %-6s  | %d |", mlp?"MLP":"linear", K);
        for(C=1;C<=CMAX;C++){ double sum=0; int nv=0;
            for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*911u+(unsigned)K*53u+1u));
                sum+=best_eval(C,mlp,(uint32_t)(sd*7u+1u)); nv++; }
            printf(" %.3f", sum/nv); }
        printf("\n");
    }
    printf("\nthe combining layer recovers K=3 if the MLP row for K=3 crosses target at C=3, where the\n");
    printf("linear row did not -- i.e. the failure was the linear head, not the conv channels themselves.\n");
    return 0;
}
