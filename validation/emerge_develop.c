/* emerge_develop.c -- the whole sequence, chained: search-from-scratch vs a developmental trajectory.
 *
 * Every operator was validated alone (prune 1-7, clone 8-9, recombine 10, translate 17, staged 11).
 * This chains them into ONE run and asks the developmental question: once you have a stable working
 * block, is it better to REUSE and REFINE it than to search structure from scratch -- and do you still
 * need jitter (annealing) to escape local minima, or does reusing the block land you close enough that
 * only light refinement is needed?
 *
 * Task: translation-invariant motif (M vs M' at a random position), scarce data (where it matters).
 * Trajectory, logged phase by phase:
 *   (1) SEARCH FROM SCRATCH  -- P independent per-position detectors trained from random. This is the
 *       "search every structure" baseline; it STARVES (each position sees few examples) -- the worsening.
 *   (2) FIND A STABLE BLOCK  -- train ONE shared detector; data-efficient, learns from all positions.
 *   (3) CLONE + TRANSLATE    -- the block is the same filter tiled at every position (a convolution):
 *       we ASSEMBLE the structure by replication, not by re-searching each position.
 *   (4) REFINE IN PLACE      -- let the tiled copies fine-tune (minor per-position mutation), WITHOUT
 *       and WITH jitter (annealed weight noise), to test whether escaping local minima still helps.
 *
 * Reports test accuracy at each phase, and refine with vs without jitter, so the developmental payoff
 * and the need (or not) for jitter are both visible.
 *
 * FINDING (40 seeds, scarce data; paper Sec 3.5): find-block+translate reaches 0.86 vs 0.60 for P
 * independent detectors searched from random (+0.25). That +0.25 is the SAME
 * weight-sharing DATA-EFFICIENCY effect as the translate/scale results (one tiled block learns from every
 * position while the independent per-position detectors starve), shown inside a chained pipeline -- NOT a
 * separate "reuse beats re-search" mechanism, and a fair shared baseline trained end-to-end captures most
 * of it. What the chain adds is two side findings: refining the tiled copies in place HURTS (0.84 < 0.86),
 * because independent fine-tuning re-introduces per-position starvation and breaks the sharing; and
 * annealing jitter hurts more (0.76), because reuse lands AT the optimum so perturbation only kicks it off.
 * Lesson: once reuse gives a good shared structure keep the sharing coordinated, and add no jitter you do
 * not need. (Caveat: on a genuinely multimodal task jitter could help -- here the assembly finds the
 * optimum directly. Whether the FOURTH operator, recombination of a second block, EMERGES when the search
 * must discover the decomposition itself is the negative of emerge_discover.c / Sec 3.6.)
 * Self-contained C99. Build: make emerge_develop
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define N     16
#define K     3
#define P     (N-K+1)
#define NTRMAX 64
#define NTE   1500
#define BLOCK_EPOCHS  300
#define REFINE_EPOCHS 150
#define LR    0.05
#define AMP   1.6
#define RESTARTS 3

static uint32_t rs=1u; static uint32_t r32(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;rs=x;return x;}
static void rseed(uint32_t s){rs=s?s:1u;} static double runif(void){return (double)r32()/4294967296.0*2.0-1.0;}
static uint32_t urand(uint32_t n){ return r32()%n; }
static uint32_t wr=1u; static uint32_t wr32(void){uint32_t x=wr;x^=x<<13;x^=x>>17;x^=x<<5;wr=x;return x;}
static void wseed(uint32_t s){wr=s?s:1u;} static double wunif(void){return (double)wr32()/4294967296.0*2.0-1.0;}

static int envint(const char*k,int d){const char*e=getenv(k);return e&&*e?atoi(e):d;}
static int    g_ntr = 24;

static double Mpos[K], Mneg[K];
static double Xtr[NTRMAX][N], Xte[NTE][N];
static int    ytr[NTRMAX], yte[NTE];

static void gen(double X[][N],int*y,int n)
{ int s,i,p,k; for(s=0;s<n;s++){ for(i=0;i<N;i++) X[s][i]=runif(); y[s]=(int)(r32()&1u); p=(int)urand(P);
    for(k=0;k<K;k++) X[s][p+k]+=AMP*(y[s]?Mpos[k]:Mneg[k]); } }
static void new_task(uint32_t seed){ int k; rseed(seed);
    for(k=0;k<K;k++){ Mpos[k]=runif()>0?1:-1; Mneg[k]=runif()>0?1:-1; }
    if(Mpos[0]==Mneg[0]&&Mpos[1]==Mneg[1]&&Mpos[2]==Mneg[2]) Mneg[0]=-Mpos[0];
    gen(Xtr,ytr,g_ntr); gen(Xte,yte,NTE); }

typedef struct { double w[P][K], b[P], rsc, rb; } Net;   /* per-position filters (shared => all equal) */

static double test_acc(const Net*net)
{ int s,k,p,c=0; for(s=0;s<NTE;s++){ double best=-1e9;
    for(p=0;p<P;p++){ double z=net->b[p]; for(k=0;k<K;k++) z+=net->w[p][k]*Xte[s][p+k]; double v=tanh(z); if(v>best)best=v; }
    double o=1.0/(1.0+exp(-(net->rsc*best+net->rb))); if((o>0.5)==(yte[s]==1)) c++; } return (double)c/NTE; }

/* (1) SEARCH FROM SCRATCH: P independent filters from random; only the winning position learns. */
static double search_scratch(uint32_t seed)
{
    Net net; int e,s,k,p; wseed(seed);
    for(p=0;p<P;p++){ for(k=0;k<K;k++) net.w[p][k]=0.3*wunif(); net.b[p]=0; } net.rsc=0.3*wunif(); net.rb=0;
    for(e=0;e<BLOCK_EPOCHS;e++) for(s=0;s<g_ntr;s++){ double best=-1e9; int pm=0;
        for(p=0;p<P;p++){ double z=net.b[p]; for(k=0;k<K;k++) z+=net.w[p][k]*Xtr[s][p+k]; double v=tanh(z); if(v>best){best=v;pm=p;} }
        double o=1.0/(1.0+exp(-(net.rsc*best+net.rb))), dout=(o-ytr[s])*o*(1.0-o), dpre=dout*net.rsc*(1.0-best*best);
        net.rsc-=LR*dout*best; net.rb-=LR*dout; net.b[pm]-=LR*dpre;
        for(k=0;k<K;k++) net.w[pm][k]-=LR*dpre*Xtr[s][pm+k]; }
    return test_acc(&net);
}
/* (2)+(3) FIND A STABLE BLOCK (one shared filter) and TRANSLATE it to every position -> Net. */
static Net find_block_and_translate(uint32_t seed)
{
    double w[K], b=0, rsc, rb=0; int e,s,k,p; wseed(seed);
    for(k=0;k<K;k++) w[k]=0.3*wunif();
    rsc=0.3*wunif();
    for(e=0;e<BLOCK_EPOCHS;e++) for(s=0;s<g_ntr;s++){ double best=-1e9; int pm=0;
        for(p=0;p<P;p++){ double z=b; for(k=0;k<K;k++) z+=w[k]*Xtr[s][p+k]; double v=tanh(z); if(v>best){best=v;pm=p;} }
        double o=1.0/(1.0+exp(-(rsc*best+rb))), dout=(o-ytr[s])*o*(1.0-o), dpre=dout*rsc*(1.0-best*best);
        rsc-=LR*dout*best; rb-=LR*dout; b-=LR*dpre;
        for(k=0;k<K;k++) w[k]-=LR*dpre*Xtr[s][pm+k]; }
    { Net net; for(p=0;p<P;p++){ for(k=0;k<K;k++) net.w[p][k]=w[k]; net.b[p]=b; } net.rsc=rsc; net.rb=rb; return net; }
}
/* (4) REFINE IN PLACE: fine-tune per-position from the translated structure; jit0>0 adds annealed noise. */
static double refine(Net net, double jit0, uint32_t seed)
{
    int e,s,k,p; wseed(seed*2654435761u+7u);
    for(e=0;e<REFINE_EPOCHS;e++){
        double temp = jit0 * (1.0 - (double)e/REFINE_EPOCHS);          /* annealed jitter */
        for(s=0;s<g_ntr;s++){ double best=-1e9; int pm=0;
            for(p=0;p<P;p++){ double z=net.b[p]; for(k=0;k<K;k++) z+=net.w[p][k]*Xtr[s][p+k]; double v=tanh(z); if(v>best){best=v;pm=p;} }
            double o=1.0/(1.0+exp(-(net.rsc*best+net.rb))), dout=(o-ytr[s])*o*(1.0-o), dpre=dout*net.rsc*(1.0-best*best);
            net.rsc-=LR*dout*best; net.rb-=LR*dout; net.b[pm]-=LR*dpre;
            for(k=0;k<K;k++) net.w[pm][k]-=LR*dpre*Xtr[s][pm+k]; }
        if(temp>0.0) for(p=0;p<P;p++) for(k=0;k<K;k++) net.w[p][k]+=temp*wunif();   /* jitter */
    }
    return test_acc(&net);
}

int main(void)
{
    int seeds=envint("SEEDS",30), sd;
    g_ntr=envint("NTR",24);
    double a1=0,a2=0,a3=0,a4=0, q1=0,q2=0,q3=0,q4=0;   /* means and sum-of-squares over seeds */
    for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*911u+1u));
        double s1=0,s2=0,s3=0,s4=0; int r;
        for(r=0;r<RESTARTS;r++){ uint32_t sd2=(uint32_t)(sd*7u+1u)*131u+(uint32_t)r*97u+1u;
            s1+=search_scratch(sd2);
            { Net blk=find_block_and_translate(sd2); s2+=test_acc(&blk);
              s3+=refine(blk,0.0,sd2); s4+=refine(blk,0.15,sd2); } }
        s1/=RESTARTS; s2/=RESTARTS; s3/=RESTARTS; s4/=RESTARTS;   /* mean over restarts (no best-of) */
        a1+=s1; q1+=s1*s1; a2+=s2; q2+=s2*s2; a3+=s3; q3+=s3*s3; a4+=s4; q4+=s4*s4; }
    { double m1=a1/seeds,m2=a2/seeds,m3=a3/seeds,m4=a4/seeds;
      double d1=seeds>1?sqrt((q1-a1*a1/seeds)/(seeds-1)):0, d2=seeds>1?sqrt((q2-a2*a2/seeds)/(seeds-1)):0,
             d3=seeds>1?sqrt((q3-a3*a3/seeds)/(seeds-1)):0, d4=seeds>1?sqrt((q4-a4*a4/seeds)/(seeds-1)):0;
      printf("THE WHOLE SEQUENCE, chained: search-from-scratch vs developmental (block -> translate -> refine)\n");
      printf("translation-invariant motif task, %d train examples (scarce), %d seeds x %d restarts (fair mean +/- std).\n\n", g_ntr, seeds, RESTARTS);
      printf("  phase                                   | test acc (mean +/- std)\n");
      printf("  (1) search from scratch (P independent) | %.3f +/- %.3f   <- the worsening: starves\n", m1,d1);
      printf("  (2) find a stable block + (3) translate | %.3f +/- %.3f   <- weight-shared block, reused\n", m2,d2);
      printf("  (4) refine in place, no jitter          | %.3f +/- %.3f\n", m3,d3);
      printf("  (4) refine in place, + jitter (anneal)  | %.3f +/- %.3f\n", m4,d4);
      printf("\npayoff = (2) over (1); jitter is a large effect (4+jitter vs 2); (3) vs (2) refine diff may be within noise.\n"); }
    return 0;
}
