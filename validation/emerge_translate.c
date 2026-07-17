/* emerge_translate.c -- the third operator: does REPLICATING a working detector (translation) pay?
 *
 * Two operators are established: clone/reuse (copy a block, stack it -- Sec 9) and recombine (mix
 * different blocks -- Sec 10). This adds the third, the one with the strongest biological grounding:
 * TRANSLATE -- copy a working detector to a shifted position (transposable elements / "jumping genes",
 * serial homology, topographic map tiling). Reconnecting one module across positions with SHARED
 * weights is exactly convolution's weight-shared tiling, reached by replication rather than by design.
 *
 * Is it cheating? A translate operator imposes only that the domain is TRANSLATABLE (a real symmetry of
 * a signal, like locality is a real prior) -- not whether to use it. It pays only when the task is
 * translation-invariant. The killer reason it pays is DATA EFFICIENCY: a shared filter learns the motif
 * from every position's examples, while independent per-position detectors each starve on a fraction of
 * the data (and positions never seen in training stay random). So we compare, on a translation-invariant
 * task, over training-set size:
 *   SHARED   : one filter applied at every position (weight-shared = translated) -> max-pool -> readout.
 *              K filter weights, learns from ALL positions' data. This is a convolution.
 *   INDEP    : a separate filter PER position (locally connected, no sharing) -> max-pool -> readout.
 *              K*P weights, each trained only when the motif lands on it (via the max) -> data-starved.
 * Prediction: SHARED matches/beats INDEP at far fewer parameters, and the gap GROWS as data shrinks.
 *
 * FINDING (10 seeds) -- a clean, strong POSITIVE: SHARED (translation, 3 weights) beats INDEP
 * (per-position, 42 weights) at every train size, and the gap GROWS as data shrinks -- +0.16 at 128
 * examples up to +0.22 at 16 (0.87 vs 0.65). Translation = data efficiency: one tiled working module
 * learns from all positions at once; independent per-position detectors starve on their fraction and
 * leave unseen positions random. This is the antidote to the "adding independent structure worsens
 * things" problem: nature does not grow a fresh detector per location, it replicates one that works.
 * Self-contained C99. Build: make emerge_translate
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define N     16
#define K     3
#define P     (N-K+1)        /* = 14 positions */
#define NTRMAX 160
#define NTE   1500
#define TEPOCHS 300
#define LR    0.05
#define AMP   1.6
#define RESTARTS 3

static uint32_t rs=1u; static uint32_t r32(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;rs=x;return x;}
static void rseed(uint32_t s){rs=s?s:1u;} static double runif(void){return (double)r32()/4294967296.0*2.0-1.0;}
static uint32_t urand(uint32_t n){ return r32()%n; }
static uint32_t wr=1u; static uint32_t wr32(void){uint32_t x=wr;x^=x<<13;x^=x>>17;x^=x<<5;wr=x;return x;}
static void wseed(uint32_t s){wr=s?s:1u;} static double wunif(void){return (double)wr32()/4294967296.0*2.0-1.0;}

static int envint(const char*k,int d){const char*e=getenv(k);return e&&*e?atoi(e):d;}
static double envdbl(const char*k,double d){const char*e=getenv(k);return e&&*e?atof(e):d;}
static double g_target = 0.85;
static int    g_ntr = 32;

static double Mpos[K], Mneg[K];        /* the two motifs for the current task */
static double Xtr[NTRMAX][N], Xte[NTE][N];
static int    ytr[NTRMAX], yte[NTE];

/* plant motif M (positive) or the other motif (negative) at a RANDOM position -> translation-invariant */
static void gen(double X[][N],int*y,int n)
{
    int s,i,p,k;
    for(s=0;s<n;s++){
        for(i=0;i<N;i++) X[s][i]=runif();
        y[s]=(int)(r32()&1u);
        p=(int)urand(P);
        for(k=0;k<K;k++) X[s][p+k]+=AMP*(y[s]?Mpos[k]:Mneg[k]);
    }
}
static void new_task(uint32_t seed){ int k; rseed(seed);
    for(k=0;k<K;k++){ Mpos[k]=runif()>0?1:-1; Mneg[k]=runif()>0?1:-1; }
    /* ensure the two motifs differ */
    if(Mpos[0]==Mneg[0]&&Mpos[1]==Mneg[1]&&Mpos[2]==Mneg[2]) Mneg[0]=-Mpos[0];
    gen(Xtr,ytr,g_ntr); gen(Xte,yte,NTE); }

/* SHARED: one filter shared across all positions (a convolution). max-pool, linear readout. */
static double train_shared(uint32_t seed)
{
    double w[K], b=0, rsc, rb=0; int e,s,k,p, c=0;
    wseed(seed); for(k=0;k<K;k++) w[k]=0.3*wunif(); rsc=0.3*wunif();
    for(e=0;e<TEPOCHS;e++) for(s=0;s<g_ntr;s++){
        double best=-1e9; int pm=0;
        for(p=0;p<P;p++){ double z=b; for(k=0;k<K;k++) z+=w[k]*Xtr[s][p+k]; double v=tanh(z); if(v>best){best=v;pm=p;} }
        double o=1.0/(1.0+exp(-(rsc*best+rb))), dout=(o-ytr[s])*o*(1.0-o);
        double dpre=dout*rsc*(1.0-best*best);
        rsc-=LR*dout*best; rb-=LR*dout; b-=LR*dpre;
        for(k=0;k<K;k++) w[k]-=LR*dpre*Xtr[s][pm+k];
    }
    for(s=0;s<NTE;s++){ double best=-1e9;
        for(p=0;p<P;p++){ double z=b; for(k=0;k<K;k++) z+=w[k]*Xte[s][p+k]; double v=tanh(z); if(v>best)best=v; }
        double o=1.0/(1.0+exp(-(rsc*best+rb))); if((o>0.5)==(yte[s]==1)) c++; }
    return (double)c/NTE;
}
/* INDEP: a separate filter per position (locally connected, no weight sharing). max-pool, readout. */
static double train_indep(uint32_t seed)
{
    static double w[P][K]; double b[P], rsc, rb=0; int e,s,k,p, c=0;
    wseed(seed); for(p=0;p<P;p++){ for(k=0;k<K;k++) w[p][k]=0.3*wunif(); b[p]=0; } rsc=0.3*wunif();
    for(e=0;e<TEPOCHS;e++) for(s=0;s<g_ntr;s++){
        double best=-1e9; int pm=0;
        for(p=0;p<P;p++){ double z=b[p]; for(k=0;k<K;k++) z+=w[p][k]*Xtr[s][p+k]; double v=tanh(z); if(v>best){best=v;pm=p;} }
        double o=1.0/(1.0+exp(-(rsc*best+rb))), dout=(o-ytr[s])*o*(1.0-o);
        double dpre=dout*rsc*(1.0-best*best);
        rsc-=LR*dout*best; rb-=LR*dout; b[pm]-=LR*dpre;
        for(k=0;k<K;k++) w[pm][k]-=LR*dpre*Xtr[s][pm+k];   /* only the winning position's filter learns */
    }
    for(s=0;s<NTE;s++){ double best=-1e9;
        for(p=0;p<P;p++){ double z=b[p]; for(k=0;k<K;k++) z+=w[p][k]*Xte[s][p+k]; double v=tanh(z); if(v>best)best=v; }
        double o=1.0/(1.0+exp(-(rsc*best+rb))); if((o>0.5)==(yte[s]==1)) c++; }
    return (double)c/NTE;
}
/* MEAN over restarts (fair -- no optimistic best-of selection). */
static double mean_of(int shared, uint32_t seed)
{ int r; double s=0; for(r=0;r<RESTARTS;r++){ double a= shared? train_shared(seed*131u+(uint32_t)r*97u+1u) : train_indep(seed*131u+(uint32_t)r*97u+1u); s+=a; } return s/RESTARTS; }

int main(void)
{
    int seeds=envint("SEEDS",10), sd, t;
    int sizes[4]={16,32,64,128};
    g_target=envdbl("TARGET",0.85);
    printf("TRANSLATE (the 3rd operator): does replicating a working detector beat learning each position?\n");
    printf("N=%d K=%d, %d positions, %d seeds x %d restarts. translation-invariant motif task.\n", N, K, P, seeds, RESTARTS);
    printf("SHARED = one filter tiled at every position (a convolution), %d filter weights.\n", K);
    printf("INDEP  = a separate filter per position (locally connected), %d filter weights.\n\n", K*P);
    printf("fair stats: mean over restarts (no best-of), mean +/- std over seeds.\n");
    printf("  train size | SHARED (weight-shared) | INDEP (per-position)  | gap\n");
    for(t=0;t<4;t++){ g_ntr=sizes[t];
        double ss=0,ss2=0, si=0,si2=0;
        for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*911u+(unsigned)g_ntr*17u+1u));
            double vs=mean_of(1,(uint32_t)(sd*7u+1u)), vi=mean_of(0,(uint32_t)(sd*7u+1u));
            ss+=vs; ss2+=vs*vs; si+=vi; si2+=vi*vi; }
        { double ms=ss/seeds, mi=si/seeds;
          double ds=seeds>1?sqrt((ss2-ss*ss/seeds)/(seeds-1)):0, di=seeds>1?sqrt((si2-si*si/seeds)/(seeds-1)):0;
          printf("  %-4d       | %.3f +/- %.3f       | %.3f +/- %.3f      | %+.3f\n", sizes[t], ms,ds, mi,di, ms-mi); }
    }
    printf("\ntranslation pays if SHARED matches/beats INDEP at %dx fewer weights, and the gap grows as data\n", P);
    printf("shrinks -- one working module tiled across positions learns from all of them (data efficiency).\n");
    return 0;
}
