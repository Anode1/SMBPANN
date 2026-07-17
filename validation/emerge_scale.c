/* emerge_scale.c -- does the developmental advantage (reuse beats re-search) GROW with problem size?
 *
 * Sec 17 and 19 showed, on small tasks, that reusing one working block (translate/tile it) beats
 * searching independent structure from scratch. The honest worry is that this is a small-task artifact.
 * This scales it: sweep the input size N (so the number of positions to cover grows), holding the data
 * scarce, and compare across scale:
 *   DEVELOPMENTAL (SHARED) : one block tiled at every position -- K weights, INDEPENDENT of N; learns
 *                            the motif from all positions at once.
 *   SEARCH-FROM-SCRATCH (INDEP) : a separate detector per position -- K*P weights, growing with N;
 *                            each starves on the fraction of examples that land on it.
 * Prediction: SHARED stays ~flat as N grows (one block, N-invariant); INDEP degrades (more positions,
 * same data -> harder starvation); the GAP widens with scale. If so, "reuse beats re-search" is a
 * property that strengthens with size, not a toy effect. Translation-invariant motif task, scarce data.
 *
 * FINDING (16 seeds, 600 epochs): confirmed and strong. As input N grows 16 -> 128, DEVELOPMENTAL (one
 * tiled block, 3 weights) stays flat at ~0.98, while SEARCH-FROM-SCRATCH (independent, up to 378 weights)
 * collapses toward chance 0.77 -> 0.54, and the gap MORE THAN DOUBLES (+0.21 -> +0.43). One tiled block
 * is size-independent; independent detectors starve harder as positions multiply. The developmental
 * advantage grows with size -- and it is decisive: 3 weights at 0.98 beats 378 weights at 0.54.
 * Self-contained C99. Build: make emerge_scale
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
#define TEPOCHS 600
#define LR    0.05
#define AMP   2.6
#define RESTARTS 4

static uint32_t rs=1u; static uint32_t r32(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;rs=x;return x;}
static void rseed(uint32_t s){rs=s?s:1u;} static double runif(void){return (double)r32()/4294967296.0*2.0-1.0;}
static uint32_t urand(uint32_t n){ return r32()%n; }
static uint32_t wr=1u; static uint32_t wr32(void){uint32_t x=wr;x^=x<<13;x^=x>>17;x^=x<<5;wr=x;return x;}
static void wseed(uint32_t s){wr=s?s:1u;} static double wunif(void){return (double)wr32()/4294967296.0*2.0-1.0;}

static int envint(const char*k,int d){const char*e=getenv(k);return e&&*e?atoi(e):d;}
static int    g_n = 16, g_p = 14, g_ntr = 32;

static double Mpos[K], Mneg[K];
static double Xtr[NTRMAX][NMAX], Xte[NTE][NMAX];
static int    ytr[NTRMAX], yte[NTE];

static void gen(double X[][NMAX],int*y,int n)
{ int s,i,p,k; for(s=0;s<n;s++){ for(i=0;i<g_n;i++) X[s][i]=runif(); y[s]=(int)(r32()&1u); p=(int)urand((uint32_t)g_p);
    for(k=0;k<K;k++) X[s][p+k]+=AMP*(y[s]?Mpos[k]:Mneg[k]); } }
static void new_task(uint32_t seed){ int k; rseed(seed);
    for(k=0;k<K;k++){ Mpos[k]=runif()>0?1:-1; Mneg[k]=runif()>0?1:-1; }
    if(Mpos[0]==Mneg[0]&&Mpos[1]==Mneg[1]&&Mpos[2]==Mneg[2]) Mneg[0]=-Mpos[0];
    gen(Xtr,ytr,g_ntr); gen(Xte,yte,NTE); }

/* SHARED: one filter tiled at every position (a convolution). K weights, independent of N. */
static double train_shared(uint32_t seed)
{
    double w[K], b=0, rsc, rb=0; int e,s,k,p, c=0;
    wseed(seed); for(k=0;k<K;k++) w[k]=0.3*wunif(); rsc=0.3*wunif();
    for(e=0;e<TEPOCHS;e++) for(s=0;s<g_ntr;s++){ double best=-1e9; int pm=0;
        for(p=0;p<g_p;p++){ double z=b; for(k=0;k<K;k++) z+=w[k]*Xtr[s][p+k]; double v=tanh(z); if(v>best){best=v;pm=p;} }
        double o=1.0/(1.0+exp(-(rsc*best+rb))), dout=(o-ytr[s])*o*(1.0-o), dpre=dout*rsc*(1.0-best*best);
        rsc-=LR*dout*best; rb-=LR*dout; b-=LR*dpre;
        for(k=0;k<K;k++) w[k]-=LR*dpre*Xtr[s][pm+k]; }
    for(s=0;s<NTE;s++){ double best=-1e9;
        for(p=0;p<g_p;p++){ double z=b; for(k=0;k<K;k++) z+=w[k]*Xte[s][p+k]; double v=tanh(z); if(v>best)best=v; }
        double o=1.0/(1.0+exp(-(rsc*best+rb))); if((o>0.5)==(yte[s]==1)) c++; }
    return (double)c/NTE;
}
/* INDEP: a separate filter per position (locally connected). K*P weights, growing with N. */
static double train_indep(uint32_t seed)
{
    static double w[PMAX][K]; double b[PMAX], rsc, rb=0; int e,s,k,p, c=0;
    wseed(seed); for(p=0;p<g_p;p++){ for(k=0;k<K;k++) w[p][k]=0.3*wunif(); b[p]=0; } rsc=0.3*wunif();
    for(e=0;e<TEPOCHS;e++) for(s=0;s<g_ntr;s++){ double best=-1e9; int pm=0;
        for(p=0;p<g_p;p++){ double z=b[p]; for(k=0;k<K;k++) z+=w[p][k]*Xtr[s][p+k]; double v=tanh(z); if(v>best){best=v;pm=p;} }
        double o=1.0/(1.0+exp(-(rsc*best+rb))), dout=(o-ytr[s])*o*(1.0-o), dpre=dout*rsc*(1.0-best*best);
        rsc-=LR*dout*best; rb-=LR*dout; b[pm]-=LR*dpre;
        for(k=0;k<K;k++) w[pm][k]-=LR*dpre*Xtr[s][pm+k]; }
    for(s=0;s<NTE;s++){ double best=-1e9;
        for(p=0;p<g_p;p++){ double z=b[p]; for(k=0;k<K;k++) z+=w[p][k]*Xte[s][p+k]; double v=tanh(z); if(v>best)best=v; }
        double o=1.0/(1.0+exp(-(rsc*best+rb))); if((o>0.5)==(yte[s]==1)) c++; }
    return (double)c/NTE;
}
/* MEAN over restarts (fair expected performance -- no optimistic best-of selection). */
static double mean_of(int shared, uint32_t seed)
{ int r; double s=0; for(r=0;r<RESTARTS;r++){ double a= shared? train_shared(seed*131u+(uint32_t)r*97u+1u) : train_indep(seed*131u+(uint32_t)r*97u+1u); s+=a; } return s/RESTARTS; }

int main(void)
{
    int seeds=envint("SEEDS",16), sd, t;
    int Ns[6]={16,32,48,64,96,128};
    g_ntr=envint("NTR",32);
    printf("SCALE: does 'reuse beats re-search' GROW with problem size? (%d train examples, %d seeds x %d restarts)\n", g_ntr, seeds, RESTARTS);
    printf("translation-invariant motif task. DEVELOPMENTAL = one tiled block (K=%d weights, N-invariant).\n", K);
    printf("SEARCH-FROM-SCRATCH = one filter per position (K*P weights, grows with N).\n\n");
    printf("fair statistics: MEAN over restarts (no best-of selection), reported as mean +/- std over seeds.\n");
    printf("  input N | positions | develop (shared) mean+/-sd | scratch (indep) mean+/-sd | gap   | scratch weights\n");
    for(t=0;t<6;t++){ g_n=Ns[t]; g_p=g_n-K+1;
        double ss=0,ss2=0, si=0,si2=0;
        for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*911u+(unsigned)g_n*17u+1u));
            double vs=mean_of(1,(uint32_t)(sd*7u+1u)), vi=mean_of(0,(uint32_t)(sd*7u+1u));
            ss+=vs; ss2+=vs*vs; si+=vi; si2+=vi*vi; }
        { double ms=ss/seeds, mi=si/seeds;
          double sds = seeds>1? sqrt((ss2-ss*ss/seeds)/(seeds-1)) : 0.0;
          double sdi = seeds>1? sqrt((si2-si*si/seeds)/(seeds-1)) : 0.0;
          printf("  %-4d    | %-4d      | %.3f +/- %.3f            | %.3f +/- %.3f           | %+.3f | %d\n",
                 g_n, g_p, ms, sds, mi, sdi, ms-mi, K*g_p); }
    }
    printf("\nreuse-beats-re-search STRENGTHENS with scale if 'develop' stays flat while 'scratch' falls and\n");
    printf("the gap widens -- one tiled block is size-independent; independent detectors starve harder as N grows.\n");
    return 0;
}
