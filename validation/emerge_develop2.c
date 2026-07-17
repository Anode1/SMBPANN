/* emerge_develop2.c -- the developmental chain extended to TWO discovered blocks + RECOMBINATION,
 * on a task that genuinely needs two DIFFERENT operations. Sibling of emerge_develop.c (one block).
 *
 * emerge_develop.c chained the whole sequence for ONE operation: search-from-scratch STARVES, while
 * finding one stable shared block and TRANSLATING it (a convolution) reuses data across positions and
 * wins (+0.25). But one reused block only buys ONE operation. emerge_twoop.c showed the regime where
 * that breaks: a task needing two DIFFERENT feature extractors, where no single reused block serves
 * both roles. This file asks the developmental question there: if you DISCOVER two blocks (one per
 * operation), tile each (translate), and RECOMBINE them into one read-out, do you SOLVE the two-op
 * task that search-from-scratch and either single block CANNOT -- and at what parameter energy?
 *
 * Task (translation-invariant, two operations, AND-composition):
 *   Two distinct 3-tap sign motifs A and B (drawn per seed, A != B, A != -B) are each planted at a
 *   random non-overlapping position in a noisy length-16 signal.
 *     POSITIVE : BOTH A and B are present.
 *     NEGATIVE : exactly ONE is present (half A-only, half B-only).
 *   So presence of A alone, or B alone, never separates the classes: you must detect A AND B. A single
 *   max-pool channel (one filter, tiled) can fire on at most one motif, so ANY single-block model caps
 *   at "predict positive iff that motif present" = 0.75 by construction (pos .50 all have it, plus the
 *   .25 of negatives that lack it). Only a TWO-channel read-out that ANDs the two pools can reach ~1.0.
 *
 * The four arms (fair mean over restarts, NOT best-of; same NTR training data for every arm):
 *   (1) SEARCH FROM SCRATCH -- equal-capacity control: TWO channels, each with P independent
 *       per-position 3-tap detectors (no weight sharing, the "search every structure" baseline),
 *       trained JOINTLY from random on the full labels. Has the capacity to AND, but each position
 *       sees few examples -> starves (the emerge_develop worsening, now in 2 channels). Gets as many
 *       total training epochs as the whole developmental chain below, so it is not starved of compute.
 *   (2) BLOCK A ONLY -- discover one shared block on operation 1 (aux label "A present"), tile it, fit
 *       a single-channel read-out on the full task. Caps ~0.75: one op is not enough.
 *   (3) BLOCK B ONLY -- symmetric, operation 2. Also ~0.75.
 *   (4) DEVELOPMENTAL TWO-BLOCK RECOMBINED -- discover block A on op 1 and block B on op 2 (each a
 *       shared filter trained via curriculum aux labels, then CLONED+TRANSLATED to every position),
 *       FREEZE both blocks, and train ONLY the recombination read-out (rA,rB,bias) that ANDs the two
 *       tiled channels on the full task. This is "assemble the structure by recombining reused parts."
 *
 * Honesty / confounds controlled:
 *   - Capacity: the scratch baseline has the SAME two-channel AND capacity as the recombined arm (in
 *     fact FAR more parameters: independent per-position filters), so a recombined win is not "more
 *     compute for one arm" -- it wins with ~10x FEWER parameters. Parameter energy is reported.
 *   - Training compute: scratch gets 2*BLOCK+READOUT epochs = the sum the developmental chain spends
 *     across its three training phases, so scratch is not disadvantaged on gradient steps.
 *   - Data: every arm trains on the SAME NTR examples. The developmental arm's ONLY extra ingredient
 *     is auxiliary per-operation labels ("which op is present") -- the curriculum signal that names the
 *     two operations. That is the developmental assumption, stated openly; the result is conditional on
 *     it. It is NOT extra data.
 *   - Task validity: single-block arms (2),(3) must cap near 0.75 (neither op alone solves it); if a
 *     single block already solved the task the recombination would look falsely necessary -- checked.
 *
 * FINDING (SEEDS=__ x RESTARTS=__, NTR=__): __FILL_AFTER_RUN__
 *
 * Self-contained C99. Build: make emerge_develop2 . Env overrides: SEEDS, NTR, RESTARTS.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define N      16
#define K      3
#define P      (N-K+1)
#define NTRMAX 256
#define NTE    1500
#define BLOCK_EPOCHS   300
#define READOUT_EPOCHS 200
#define SCRATCH_EPOCHS (2*BLOCK_EPOCHS+READOUT_EPOCHS)   /* = total epochs the dev chain spends */
#define LR    0.05
#define AMP_DEFAULT 3.0    /* motif-vs-noise margin; large so a matched filter clears noise max-pool */

/* task PRNG (data) and weight PRNG (init) kept separate, exactly as the sibling files */
static uint32_t rs=1u; static uint32_t r32(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;rs=x;return x;}
static void rseed(uint32_t s){rs=s?s:1u;} static double runif(void){return (double)r32()/4294967296.0*2.0-1.0;}
static uint32_t urand(uint32_t n){ return r32()%n; }
static uint32_t wr=1u; static uint32_t wr32(void){uint32_t x=wr;x^=x<<13;x^=x>>17;x^=x<<5;wr=x;return x;}
static void wseed(uint32_t s){wr=s?s:1u;} static double wunif(void){return (double)wr32()/4294967296.0*2.0-1.0;}

static int envint(const char*k,int d){const char*e=getenv(k);return e&&*e?atoi(e):d;}
static double envdbl(const char*k,double d){const char*e=getenv(k);return e&&*e?atof(e):d;}
static int    g_ntr = 64;
static double g_amp = AMP_DEFAULT;

static double MA[K], MB[K];                 /* the two motifs for the current task */
static double Xtr[NTRMAX][N], Xte[NTE][N];
static int    ytr[NTRMAX], yte[NTE];        /* full-task label: positive iff A AND B present */
static int    yAtr[NTRMAX], yBtr[NTRMAX];   /* auxiliary curriculum labels: A present / B present */

/* plant a 3-tap motif M at position q (q..q+2), adding AMP*sign on top of the noise */
static void plant(double*x,int q,const double*M){ int k; for(k=0;k<K;k++) x[q+k]+=g_amp*M[k]; }

/* pick two non-overlapping motif positions qa,qb in [0,P-1] with |qa-qb|>=K */
static void two_pos(int*qa,int*qb){ int a,b; a=(int)urand(P);
    do { b=(int)urand(P); } while(abs(a-b)<K); *qa=a; *qb=b; }

/* generate the full two-operation task and the auxiliary per-op labels.
 * cls 0 = positive (A and B), cls 1 = neg A-only, cls 2 = neg B-only (negatives split evenly). */
static void gen(double X[][N],int*y,int*yA,int*yB,int n)
{
    int s,i,qa,qb,cls;
    for(s=0;s<n;s++){
        for(i=0;i<N;i++) X[s][i]=runif();
        cls=(int)urand(2) ? (int)(1+urand(2)) : 0;     /* 50% positive, 25% A-only, 25% B-only */
        two_pos(&qa,&qb);
        if(cls==0){ plant(X[s],qa,MA); plant(X[s],qb,MB); y[s]=1; if(yA)yA[s]=1; if(yB)yB[s]=1; }
        else if(cls==1){ plant(X[s],qa,MA); y[s]=0; if(yA)yA[s]=1; if(yB)yB[s]=0; }   /* A only */
        else            { plant(X[s],qb,MB); y[s]=0; if(yA)yA[s]=0; if(yB)yB[s]=1; }  /* B only */
    }
}
static void rand_motif(double*M){ int k; for(k=0;k<K;k++) M[k]=runif()>0?1.0:-1.0; }
static double dot3(const double*u,const double*v){ return u[0]*v[0]+u[1]*v[1]+u[2]*v[2]; }
/* draw A,B anti-correlated (dot == -1: share one tap, differ in two). Then motif B actively DRIVES
 * filter A negative (and vice versa), so neither motif false-triggers the other's detector -- the two
 * operations are cleanly separable, and any residual error is honest noise, not motif cross-talk. */
static void new_task(uint32_t seed){ rseed(seed);
    do { rand_motif(MA); rand_motif(MB); } while(dot3(MA,MB)!=-1.0);
    gen(Xtr,ytr,yAtr,yBtr,g_ntr); gen(Xte,yte,NULL,NULL,NTE); }

/* ---- shared-channel forward: one filter w[K],b tiled at every position, max-pool over positions ---- */
static double chan_pool(const double*w,double b,const double*x,int*argmax)
{ int k,p,am=0; double best=-1e9;
  for(p=0;p<P;p++){ double z=b; for(k=0;k<K;k++) z+=w[k]*x[p+k]; { double v=tanh(z); if(v>best){best=v;am=p;} } }
  if(argmax)*argmax=am;
  return best; }

/* ================= (2)+(4) DISCOVER ONE SHARED BLOCK on a given (auxiliary) label ================= */
/* trains a shared filter + throwaway scalar read-out to detect the op named by lab[]; returns the block
 * weights in outw,*outb. This is emerge_develop's find_block, driven by the curriculum label. */
static void find_block(const int*lab, uint32_t seed, double outw[K], double*outb)
{
    double w[K], b=0, rsc, rb=0; int e,s,k; wseed(seed);
    for(k=0;k<K;k++) w[k]=0.3*wunif();
    rsc=0.3*wunif();
    for(e=0;e<BLOCK_EPOCHS;e++) for(s=0;s<g_ntr;s++){
        int am; double pool=chan_pool(w,b,Xtr[s],&am);
        double o=1.0/(1.0+exp(-(rsc*pool+rb))), dout=(o-lab[s])*o*(1.0-o), dz=dout*rsc*(1.0-pool*pool);
        rsc-=LR*dout*pool; rb-=LR*dout; b-=LR*dz;
        for(k=0;k<K;k++) w[k]-=LR*dz*Xtr[s][am+k]; }
    for(k=0;k<K;k++) outw[k]=w[k];
    *outb=b;
}

/* ============ (2)/(3) BLOCK-ONLY ARM: freeze one block, fit a 1-channel read-out on full task ======= */
static double readout1_acc(const double w[K], double b, uint32_t seed)
{
    double rsc, rb=0; int e,s,k,c=0; wseed(seed*2654435761u+7u);
    rsc=0.3*wunif();
    for(e=0;e<READOUT_EPOCHS;e++) for(s=0;s<g_ntr;s++){
        double pool=chan_pool(w,b,Xtr[s],NULL);
        double o=1.0/(1.0+exp(-(rsc*pool+rb))), dout=(o-ytr[s])*o*(1.0-o);
        rsc-=LR*dout*pool; rb-=LR*dout; }
    for(s=0;s<NTE;s++){ double pool=chan_pool(w,b,Xte[s],NULL);
        double o=1.0/(1.0+exp(-(rsc*pool+rb))); if((o>0.5)==(yte[s]==1)) c++; }
    (void)k; return (double)c/NTE;
}

/* ============ (4) RECOMBINE: freeze block A and block B, train ONLY the ANDing read-out ============ */
static double recombine_acc(const double wA[K],double bA,const double wB[K],double bB,uint32_t seed)
{
    double rA, rB, rb=0; int e,s,c=0; wseed(seed*40503u+13u);
    rA=0.3*wunif(); rB=0.3*wunif();
    for(e=0;e<READOUT_EPOCHS;e++) for(s=0;s<g_ntr;s++){
        double pa=chan_pool(wA,bA,Xtr[s],NULL), pb=chan_pool(wB,bB,Xtr[s],NULL);
        double o=1.0/(1.0+exp(-(rA*pa+rB*pb+rb))), dout=(o-ytr[s])*o*(1.0-o);
        rA-=LR*dout*pa; rB-=LR*dout*pb; rb-=LR*dout; }
    for(s=0;s<NTE;s++){ double pa=chan_pool(wA,bA,Xte[s],NULL), pb=chan_pool(wB,bB,Xte[s],NULL);
        double o=1.0/(1.0+exp(-(rA*pa+rB*pb+rb))); if((o>0.5)==(yte[s]==1)) c++; }
    return (double)c/NTE;
}

/* ================= (1) SEARCH FROM SCRATCH: two channels, P independent per-position filters ========= */
/* equal-capacity control -- can represent the AND, but per-position filters starve on scarce data.
 * trained JOINTLY from random on the full label, with SCRATCH_EPOCHS = the dev chain's total epochs. */
static double search_scratch(uint32_t seed)
{
    static double wA[P][K], bA[P], wB[P][K], bB[P];
    double rA, rB, rb=0; int e,s,k,p,c=0; wseed(seed);
    for(p=0;p<P;p++){ for(k=0;k<K;k++){ wA[p][k]=0.3*wunif(); } bA[p]=0; }
    for(p=0;p<P;p++){ for(k=0;k<K;k++){ wB[p][k]=0.3*wunif(); } bB[p]=0; }  /* B drawn after A => asymmetric init */
    rA=0.3*wunif(); rB=0.3*wunif();
    for(e=0;e<SCRATCH_EPOCHS;e++) for(s=0;s<g_ntr;s++){
        int amA=0,amB=0; double bestA=-1e9,bestB=-1e9;
        for(p=0;p<P;p++){ double z=bA[p]; for(k=0;k<K;k++) z+=wA[p][k]*Xtr[s][p+k];
            { double v=tanh(z); if(v>bestA){bestA=v;amA=p;} } }
        for(p=0;p<P;p++){ double z=bB[p]; for(k=0;k<K;k++) z+=wB[p][k]*Xtr[s][p+k];
            { double v=tanh(z); if(v>bestB){bestB=v;amB=p;} } }
        { double o=1.0/(1.0+exp(-(rA*bestA+rB*bestB+rb))), dout=(o-ytr[s])*o*(1.0-o);
          double dzA=dout*rA*(1.0-bestA*bestA), dzB=dout*rB*(1.0-bestB*bestB);
          rA-=LR*dout*bestA; rB-=LR*dout*bestB; rb-=LR*dout;
          bA[amA]-=LR*dzA; for(k=0;k<K;k++) wA[amA][k]-=LR*dzA*Xtr[s][amA+k];
          bB[amB]-=LR*dzB; for(k=0;k<K;k++) wB[amB][k]-=LR*dzB*Xtr[s][amB+k]; }
    }
    for(s=0;s<NTE;s++){
        double bestA=-1e9,bestB=-1e9;
        for(p=0;p<P;p++){ double z=bA[p]; for(k=0;k<K;k++) z+=wA[p][k]*Xte[s][p+k]; { double v=tanh(z); if(v>bestA)bestA=v; } }
        for(p=0;p<P;p++){ double z=bB[p]; for(k=0;k<K;k++) z+=wB[p][k]*Xte[s][p+k]; { double v=tanh(z); if(v>bestB)bestB=v; } }
        { double o=1.0/(1.0+exp(-(rA*bestA+rB*bestB+rb))); if((o>0.5)==(yte[s]==1)) c++; }
    }
    return (double)c/NTE;
}

int main(void)
{
    int seeds=envint("SEEDS",30), restarts=envint("RESTARTS",3), sd, r;
    g_ntr=envint("NTR",64); if(g_ntr>NTRMAX) g_ntr=NTRMAX;
    g_amp=envdbl("AMP",AMP_DEFAULT);
    double a1=0,a2=0,a3=0,a4=0, q1=0,q2=0,q3=0,q4=0;
    for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*911u+1u));
        double s1=0,s2=0,s3=0,s4=0;
        for(r=0;r<restarts;r++){ uint32_t w=(uint32_t)(sd*7u+1u)*131u+(uint32_t)r*97u+1u;
            double wA[K],bA,wB[K],bB;
            s1+=search_scratch(w);
            find_block(yAtr,w,wA,&bA);            /* discover block A on operation 1 */
            find_block(yBtr,w,wB,&bB);            /* discover block B on operation 2 */
            s2+=readout1_acc(wA,bA,w);            /* block A alone */
            s3+=readout1_acc(wB,bB,w);            /* block B alone */
            s4+=recombine_acc(wA,bA,wB,bB,w);     /* recombine both */
        }
        s1/=restarts; s2/=restarts; s3/=restarts; s4/=restarts;   /* mean over restarts, not best-of */
        a1+=s1; q1+=s1*s1; a2+=s2; q2+=s2*s2; a3+=s3; q3+=s3*s3; a4+=s4; q4+=s4*s4;
    }
    { double m1=a1/seeds,m2=a2/seeds,m3=a3/seeds,m4=a4/seeds;
      double d1=seeds>1?sqrt((q1-a1*a1/seeds)/(seeds-1)):0, d2=seeds>1?sqrt((q2-a2*a2/seeds)/(seeds-1)):0,
             d3=seeds>1?sqrt((q3-a3*a3/seeds)/(seeds-1)):0, d4=seeds>1?sqrt((q4-a4*a4/seeds)/(seeds-1)):0;
      int e_scratch = 2*P*K + 2*P + 3;      /* two channels of P independent per-pos filters + read-out */
      int e_block   = K + 1 + 2;            /* one shared block + 1-channel read-out */
      int e_recomb  = 2*(K+1) + 3;          /* two shared blocks + ANDing read-out */
      printf("DEVELOPMENTAL TWO-BLOCK RECOMBINATION on a two-operation (A AND B) task\n");
      printf("N=%d K=%d P=%d, %d train examples, %d seeds x %d restarts (fair mean +/- SD, not best-of)\n",
             N,K,P,g_ntr,seeds,restarts);
      printf("positive = motif A AND motif B present; negative = exactly one present. Single-op cap = 0.75.\n\n");
      printf("  arm                                         | test acc (mean +/- SD) | params(energy)\n");
      printf("  (1) search from scratch (2ch, P indep/pos)  | %.3f +/- %.3f        | %d\n", m1,d1,e_scratch);
      printf("  (2) block A only  (op 1, tiled + read-out)  | %.3f +/- %.3f        | %d\n", m2,d2,e_block);
      printf("  (3) block B only  (op 2, tiled + read-out)  | %.3f +/- %.3f        | %d\n", m3,d3,e_block);
      printf("  (4) developmental TWO-BLOCK recombined      | %.3f +/- %.3f        | %d\n", m4,d4,e_recomb);
      printf("\ntraining epochs: scratch=%d (= dev chain total); each block=%d; each read-out=%d.\n",
             SCRATCH_EPOCHS, BLOCK_EPOCHS, READOUT_EPOCHS);
      printf("does the two-block chain close the task? compare (4) vs (1) and vs max((2),(3)); note (4)'s energy.\n");
    }
    return 0;
}
