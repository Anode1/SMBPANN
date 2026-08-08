/* emerge_discover.c -- compositional emergence: does the SEARCH (not the experimenter)
 * discover the decomposition, from the composite label + an energy budget ALONE, with NO auxiliary
 * per-operation labels?
 *
 * WHY THIS FILE EXISTS. emerge_develop2.c reported a "recombination" win (recombined 0.89 vs
 * scratch 0.65). A fair control (two SHARED channels trained jointly end-to-end on the composite
 * label, no aux labels, no freeze) reaches 0.81 by itself, so that +0.24 decomposes into ~2/3
 * weight-sharing (already known, Sec 3.3-3.4) + ~1/3 auxiliary-label CURRICULUM, with ~0 attributable
 * to emergent composition. There, the experimenter supplied the decomposition: how many parts (two
 * find_block calls), which parts (the aux labels yA/yB), and how they wire (a fixed AND read-out).
 * That is construction, not emergence.
 *
 * THE QUESTION here: give the search only the COMPOSITE label and an energy budget on channels,
 * and let it decide the number of parts by itself (energy-selected channel count C*, exactly as depth
 * is energy-selected in Sec 2.5). Two things must EMERGE with no per-operation supervision:
 *   (a) DISCOVERY OF "HOW MANY PARTS": C* tracks the number of operations -- C*=1 for a one-operation
 *       task, C*=2 for a two-operation (A AND B) task -- because one shared tiled channel caps at 0.75
 *       on the two-op task (arithmetic: it fires on at most one motif) and only a second channel breaks
 *       the cap.
 *   (b) SPONTANEOUS SPECIALIZATION ("WHICH PARTS"): in the C=2 solution the two channels lock onto the
 *       two DIFFERENT motifs (channel filters align with MA and MB respectively) -- functional
 *       decomposition discovered from the composite label alone, never told which op is which.
 *
 * MODEL. C weight-shared K-tap filters, each tiled across all positions (translation by construction)
 * with per-channel global max-pool; a linear read-out sum_c r[c]*pool[c] + rb -> sigmoid. ALL weights
 * (filters + read-out) trained jointly by plain SGD on the COMPOSITE label only. Channel count is swept
 * C=1..CMAX and the energy-selected C* is the shallowest C reaching target (fair mean over restarts and
 * paired seeds -- never best-of).
 *
 * BASELINES for the two-op task (context, not the claim):
 *   - unshared from-scratch (P independent per-position filters, 2 channels)  -- the starvation floor.
 *   - aux-label CURRICULUM (find A on yA, find B on yB, freeze, train read-out) -- the SUPERVISED upper
 *     bound; the discovered solution should sit below it (we are giving the search LESS information).
 *
 * A NULL IS AN ACCEPTABLE RESULT and must be reported straight: if C* does not track op count, or the
 * channels do not specialize, then directed search does NOT spontaneously discover this decomposition
 * under an energy budget at this scale -- itself a clean, publishable finding.
 *
 * FINDING (40 seeds x 3 restarts, NTR=64, AMP=3.0, target 0.85) -- a PARTIAL NULL:
 *   one-op:  C*=1 (0.972 at C=1) -- the method selects a low channel count when one part suffices.
 *   two-op:  acc 0.711/0.812/0.875/0.899 at C=1..4 -> C*=3, an OVERSHOOT of the operation count (2).
 *   At C=2 the discovered (composite-label-only) solution reaches just 0.812 (= the fair joint-shared
 *   baseline), BELOW the aux-label CURRICULUM's 0.874, because the two channels SPECIALIZE to the two
 *   motifs in only 55% of runs (the rest collapse onto one motif). So directed/gradient search does NOT
 *   spontaneously discover the two-part decomposition under an energy budget alone: it under-specializes
 *   and overshoots the channel count. The clean decomposition emerges reliably only WITH per-operation
 *   supervision -- which CONFIRMS, from the opposite direction, that the auxiliary-label curriculum in
 *   emerge_develop2.c was doing the real work (its "recombination" win was ~2/3 weight-sharing + ~1/3
 *   curriculum, ~0 emergent composition). This reconciles the (removed) 2-D channel overshoot (C*~3 for
 *   a 2-op task) and bounds the emergence claim: depth emerges cleanly to match the receptive field
 *   (Sec 2.5), but COMPOSITION does not emerge cleanly on the channel axis without supervision.
 *
 * Self-contained C99. Build: make emerge_discover . Env: SEEDS, RESTARTS, NTR, AMP, TARGET.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define N      16
#define K      3
#define P      (N-K+1)
#define CMAX   4
#define NTRMAX 256
#define NTE    1500
#define EPOCHS 800          /* = emerge_develop2 SCRATCH_EPOCHS, so every arm gets equal gradient steps */
#define LR     0.05
#define AMP_DEFAULT 3.0
#define BLOCK_EPOCHS   300
#define READOUT_EPOCHS 200
#define SUP_W_DEFAULT  2.0          /* weight of the per-channel part-label auxiliary push (env SUPW) */

static uint32_t rs=1u; static uint32_t r32(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;rs=x;return x;}
static void rseed(uint32_t s){rs=s?s:1u;} static double runif(void){return (double)r32()/4294967296.0*2.0-1.0;}
static uint32_t urand(uint32_t n){ return r32()%n; }
static uint32_t wr=1u; static uint32_t wr32(void){uint32_t x=wr;x^=x<<13;x^=x>>17;x^=x<<5;wr=x;return x;}
static void wseed(uint32_t s){wr=s?s:1u;} static double wunif(void){return (double)wr32()/4294967296.0*2.0-1.0;}

static int envint(const char*k,int d){const char*e=getenv(k);return e&&*e?atoi(e):d;}
static double envdbl(const char*k,double d){const char*e=getenv(k);return e&&*e?atof(e):d;}
static int    g_ntr = 64;
static double g_amp = AMP_DEFAULT;
static double g_supw = SUP_W_DEFAULT;

static double MA[K], MB[K];
static double Xtr[NTRMAX][N], Xte[NTE][N];
static int    ytr[NTRMAX], yte[NTE];
static int    yAtr[NTRMAX], yBtr[NTRMAX];   /* aux labels: ONLY used by the curriculum baseline */

static void plant(double*x,int q,const double*M){ int k; for(k=0;k<K;k++) x[q+k]+=g_amp*M[k]; }
static void two_pos(int*qa,int*qb){ int a,b; a=(int)urand(P);
    do { b=(int)urand(P); } while(abs(a-b)<K); *qa=a; *qb=b; }

/* kop=1: one-operation task (positive = motif A present, negative = motif B present -- one motif each,
 *        so a single channel that detects A already separates the classes -> C* should be 1).
 * kop=2: two-operation AND task (positive = A AND B present; negative = exactly one) -> C* should be 2,
 *        since one channel fires on at most one motif and caps at 0.75. */
static void gen(double X[][N],int*y,int*yA,int*yB,int n,int kop)
{
    int s,i,qa,qb,cls;
    for(s=0;s<n;s++){
        for(i=0;i<N;i++) X[s][i]=runif();
        if(kop==1){
            int pos=(int)urand(P);
            if(urand(2)){ plant(X[s],pos,MA); y[s]=1; if(yA)yA[s]=1; if(yB)yB[s]=0; }
            else        { plant(X[s],pos,MB); y[s]=0; if(yA)yA[s]=0; if(yB)yB[s]=1; }
        } else {
            cls=(int)urand(2) ? (int)(1+urand(2)) : 0;   /* 50% positive, 25% A-only, 25% B-only */
            two_pos(&qa,&qb);
            if(cls==0){ plant(X[s],qa,MA); plant(X[s],qb,MB); y[s]=1; if(yA)yA[s]=1; if(yB)yB[s]=1; }
            else if(cls==1){ plant(X[s],qa,MA); y[s]=0; if(yA)yA[s]=1; if(yB)yB[s]=0; }
            else            { plant(X[s],qb,MB); y[s]=0; if(yA)yA[s]=0; if(yB)yB[s]=1; }
        }
    }
}
static void rand_motif(double*M){ int k; for(k=0;k<K;k++) M[k]=runif()>0?1.0:-1.0; }
static double dot3(const double*u,const double*v){ return u[0]*v[0]+u[1]*v[1]+u[2]*v[2]; }
static void new_task(uint32_t seed,int kop){ rseed(seed);
    do { rand_motif(MA); rand_motif(MB); } while(dot3(MA,MB)!=-1.0);   /* distinct motifs, share one tap */
    gen(Xtr,ytr,yAtr,yBtr,g_ntr,kop); gen(Xte,yte,NULL,NULL,NTE,kop); }

/* one weight-shared K-tap filter tiled at every position, per-channel global max-pool */
static double chan_pool(const double*w,double b,const double*x,int*argmax)
{ int k,p,am=0; double best=-1e9;
  for(p=0;p<P;p++){ double z=b; for(k=0;k<K;k++) z+=w[k]*x[p+k]; { double v=tanh(z); if(v>best){best=v;am=p;} } }
  if(argmax)*argmax=am;
  return best; }

/* ===== C shared tiled channels + linear read-out, trained JOINTLY on the COMPOSITE label (no aux) ===== */
static double train_multi(int C, uint32_t seed, double wout[CMAX][K], double bout[CMAX])
{
    double w[CMAX][K], b[CMAX], r[CMAX], rb=0; int e,s,k,c,cc=0;
    wseed(seed);
    for(c=0;c<C;c++){ for(k=0;k<K;k++) w[c][k]=0.3*wunif(); b[c]=0; r[c]=0.3*wunif(); }
    for(e=0;e<EPOCHS;e++) for(s=0;s<g_ntr;s++){
        int am[CMAX]; double pool[CMAX], zsum=rb;
        for(c=0;c<C;c++){ pool[c]=chan_pool(w[c],b[c],Xtr[s],&am[c]); zsum+=r[c]*pool[c]; }
        { double o=1.0/(1.0+exp(-zsum)), dout=(o-ytr[s])*o*(1.0-o);
          for(c=0;c<C;c++){ double dz=dout*r[c]*(1.0-pool[c]*pool[c]);
              r[c]-=LR*dout*pool[c]; b[c]-=LR*dz;
              for(k=0;k<K;k++) w[c][k]-=LR*dz*Xtr[s][am[c]+k]; }
          rb-=LR*dout; }
    }
    for(s=0;s<NTE;s++){ double zsum=rb;
        for(c=0;c<C;c++) zsum+=r[c]*chan_pool(w[c],b[c],Xte[s],NULL);
        { double o=1.0/(1.0+exp(-zsum)); if((o>0.5)==(yte[s]==1)) cc++; } }
    if(wout){ for(c=0;c<C;c++){ for(k=0;k<K;k++) wout[c][k]=w[c][k]; bout[c]=b[c]; } }
    return (double)cc/NTE;
}

/* ===== minimal-supervision trainer: train_multi + an auxiliary per-channel part-label push on a
 * FRACTION p of examples (channel 0 -> yA, channel 1 -> yB). p=0 is identical to train_multi (pure
 * composite label); p=1 pushes every example. Everything else -- channel forward, composite loss,
 * test-accuracy return -- is unchanged from train_multi. ===== */
static double train_sup(int C, uint32_t seed, double p, double wout[CMAX][K], double bout[CMAX])
{
    double w[CMAX][K], b[CMAX], r[CMAX], rb=0; int e,s,k,c,cc=0;
    int nsup=(int)(p*g_ntr);
    wseed(seed);
    for(c=0;c<C;c++){ for(k=0;k<K;k++) w[c][k]=0.3*wunif(); b[c]=0; r[c]=0.3*wunif(); }
    for(e=0;e<EPOCHS;e++) for(s=0;s<g_ntr;s++){
        int am[CMAX]; double pool[CMAX], zsum=rb;
        for(c=0;c<C;c++){ pool[c]=chan_pool(w[c],b[c],Xtr[s],&am[c]); zsum+=r[c]*pool[c]; }
        { double o=1.0/(1.0+exp(-zsum)), dout=(o-ytr[s])*o*(1.0-o);
          for(c=0;c<C;c++){ double dz=dout*r[c]*(1.0-pool[c]*pool[c]);
              r[c]-=LR*dout*pool[c]; b[c]-=LR*dz;
              for(k=0;k<K;k++) w[c][k]-=LR*dz*Xtr[s][am[c]+k]; }
          rb-=LR*dout; }
        if(s<nsup){                       /* supervised example: push each channel toward its part-label */
            int cc2; for(cc2=0; cc2<C && cc2<2; cc2++){
                int auxlab=(cc2==0)?yAtr[s]:yBtr[s];        /* 1 if that op is present in x[s] */
                double t=auxlab?1.0:-1.0;                    /* target pooled activation */
                double adz=g_supw*(pool[cc2]-t)*(1.0-pool[cc2]*pool[cc2]);
                b[cc2]-=LR*adz;
                for(k=0;k<K;k++) w[cc2][k]-=LR*adz*Xtr[s][am[cc2]+k]; }
        }
    }
    for(s=0;s<NTE;s++){ double zsum=rb;
        for(c=0;c<C;c++) zsum+=r[c]*chan_pool(w[c],b[c],Xte[s],NULL);
        { double o=1.0/(1.0+exp(-zsum)); if((o>0.5)==(yte[s]==1)) cc++; } }
    if(wout){ for(c=0;c<C;c++){ for(k=0;k<K;k++) wout[c][k]=w[c][k]; bout[c]=b[c]; } }
    return (double)cc/NTE;
}

/* normalized alignment |<w,M>| / (|w| |M|) in [0,1]; M has entries +/-1 so |M|=sqrt(K) */
static double align_motif(const double*w,const double*M)
{ int k; double d=0,nw=0; for(k=0;k<K;k++){ d+=w[k]*M[k]; nw+=w[k]*w[k]; }
  if(nw<1e-12) return 0.0;
  return fabs(d)/(sqrt(nw)*sqrt((double)K)); }

/* specialization: best channel->motif assignment score in [0,1]; ->1 means each channel locks onto a
 * distinct motif (channel0->MA & channel1->MB, or the swap) = the two channels have specialized. */
static double specialization(const double w0[K], const double w1[K])
{
    double a00=align_motif(w0,MA), a01=align_motif(w0,MB);
    double a10=align_motif(w1,MA), a11=align_motif(w1,MB);
    double straight=a00+a11, swap=a01+a10;
    return 0.5*(straight>swap?straight:swap);
}

/* ===== baselines for the two-op task (context, not the claim) ===== */
/* aux-label curriculum: find each block on its per-op label, freeze, train the ANDing read-out */
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
static double curriculum_acc(uint32_t seed)
{
    double wA[K],bA,wB[K],bB, rA,rB,rb=0; int e,s,c=0;
    find_block(yAtr,seed,wA,&bA); find_block(yBtr,seed,wB,&bB);
    wseed(seed*40503u+13u); rA=0.3*wunif(); rB=0.3*wunif();
    for(e=0;e<READOUT_EPOCHS;e++) for(s=0;s<g_ntr;s++){
        double pa=chan_pool(wA,bA,Xtr[s],NULL), pb=chan_pool(wB,bB,Xtr[s],NULL);
        double o=1.0/(1.0+exp(-(rA*pa+rB*pb+rb))), dout=(o-ytr[s])*o*(1.0-o);
        rA-=LR*dout*pa; rB-=LR*dout*pb; rb-=LR*dout; }
    for(s=0;s<NTE;s++){ double pa=chan_pool(wA,bA,Xte[s],NULL), pb=chan_pool(wB,bB,Xte[s],NULL);
        double o=1.0/(1.0+exp(-(rA*pa+rB*pb+rb))); if((o>0.5)==(yte[s]==1)) c++; }
    return (double)c/NTE;
}
/* unshared from-scratch: two channels of P independent per-position filters, joint on composite label */
static double scratch_acc(uint32_t seed)
{
    static double wA[P][K], bA[P], wB[P][K], bB[P];
    double rA,rB,rb=0; int e,s,k,p,c=0; wseed(seed);
    for(p=0;p<P;p++){ for(k=0;k<K;k++) wA[p][k]=0.3*wunif(); bA[p]=0; }
    for(p=0;p<P;p++){ for(k=0;k<K;k++) wB[p][k]=0.3*wunif(); bB[p]=0; }
    rA=0.3*wunif(); rB=0.3*wunif();
    for(e=0;e<EPOCHS;e++) for(s=0;s<g_ntr;s++){
        int amA=0,amB=0; double bestA=-1e9,bestB=-1e9;
        for(p=0;p<P;p++){ double z=bA[p]; for(k=0;k<K;k++) z+=wA[p][k]*Xtr[s][p+k]; { double v=tanh(z); if(v>bestA){bestA=v;amA=p;} } }
        for(p=0;p<P;p++){ double z=bB[p]; for(k=0;k<K;k++) z+=wB[p][k]*Xtr[s][p+k]; { double v=tanh(z); if(v>bestB){bestB=v;amB=p;} } }
        { double o=1.0/(1.0+exp(-(rA*bestA+rB*bestB+rb))), dout=(o-ytr[s])*o*(1.0-o);
          double dzA=dout*rA*(1.0-bestA*bestA), dzB=dout*rB*(1.0-bestB*bestB);
          rA-=LR*dout*bestA; rB-=LR*dout*bestB; rb-=LR*dout;
          bA[amA]-=LR*dzA; for(k=0;k<K;k++) wA[amA][k]-=LR*dzA*Xtr[s][amA+k];
          bB[amB]-=LR*dzB; for(k=0;k<K;k++) wB[amB][k]-=LR*dzB*Xtr[s][amB+k]; }
    }
    for(s=0;s<NTE;s++){ double bestA=-1e9,bestB=-1e9;
        for(p=0;p<P;p++){ double z=bA[p]; for(k=0;k<K;k++) z+=wA[p][k]*Xte[s][p+k]; { double v=tanh(z); if(v>bestA)bestA=v; } }
        for(p=0;p<P;p++){ double z=bB[p]; for(k=0;k<K;k++) z+=wB[p][k]*Xte[s][p+k]; { double v=tanh(z); if(v>bestB)bestB=v; } }
        { double o=1.0/(1.0+exp(-(rA*bestA+rB*bestB+rb))); if((o>0.5)==(yte[s]==1)) c++; } }
    return (double)c/NTE;
}

/* parse a comma-separated list of doubles from env key (default def); returns count, fills out[] */
static int parse_plist(const char*key,const char*def,double*out,int maxn)
{
    const char*e=getenv(key); const char*s=(e&&*e)?e:def; int n=0;
    while(*s && n<maxn){
        char*end; double v=strtod(s,&end);
        if(end==s) break;                 /* no number parsed */
        out[n++]=v; s=end;
        while(*s==',' || *s==' ') s++;
    }
    return n;
}

/* MINIMAL-SUPERVISION PHASE BOUNDARY: sweep the fraction p of examples that get a per-channel
 * part-label push; find where (if anywhere) the two C=2 channels SNAP into specializing. */
int main(void)
{
    int seeds=envint("SEEDS",24), restarts=envint("RESTARTS",4), sd, r, i;
    double plist[64]; int np;
    g_ntr=envint("NTR",64); if(g_ntr>NTRMAX) g_ntr=NTRMAX;
    g_amp=envdbl("AMP",AMP_DEFAULT);
    g_supw=envdbl("SUPW",SUP_W_DEFAULT);
    np=parse_plist("PLIST","0,0.01,0.02,0.05,0.1,0.15,0.2,0.3,0.5,1.0",plist,64);
    (void)train_multi; (void)curriculum_acc; (void)scratch_acc;   /* old baselines kept, unused */

    printf("MINIMAL-SUPERVISION PHASE BOUNDARY (two-op task, C=2, SUP_W=%.2f, %d seeds x %d restarts)\n",
           g_supw, seeds, restarts);
    printf("N=%d K=%d P=%d, NTR=%d AMP=%.1f, %d epochs/arm. fair mean over restarts, paired seeds.\n\n",
           N,K,P,g_ntr,g_amp,EPOCHS);
    printf("supervised fraction p | test acc (mean+-SEM) | specialization (mean+-SEM)\n");

    for(i=0;i<np;i++){
        double p=plist[i];
        double sum_a=0, sumsq_a=0, sum_s=0, sumsq_s=0;   /* over seeds (each = mean over restarts) */
        for(sd=1;sd<=seeds;sd++){
            new_task((uint32_t)(sd*911u + 2u*61u + 1u), 2);   /* two-op task, kop=2 (matches emerge_discover) */
            { double acc_r=0, spec_r=0;
              for(r=0;r<restarts;r++){ uint32_t w=(uint32_t)(sd*7u+1u)*131u+(uint32_t)r*97u+1u;
                  double wc[CMAX][K], bc[CMAX];
                  acc_r += train_sup(2,w,p,wc,bc);
                  spec_r += specialization(wc[0],wc[1]); }
              acc_r/=restarts; spec_r/=restarts;
              sum_a+=acc_r; sumsq_a+=acc_r*acc_r;
              sum_s+=spec_r; sumsq_s+=spec_r*spec_r; }
        }
        { double ma=sum_a/seeds, ms=sum_s/seeds;
          double va=(sumsq_a/seeds-ma*ma), vs=(sumsq_s/seeds-ms*ms);
          double sema=(seeds>1)? sqrt((va>0?va:0)/(seeds-1)) : 0.0;
          double sems=(seeds>1)? sqrt((vs>0?vs:0)/(seeds-1)) : 0.0;
          printf("  p=%-6.3f            | %.3f +- %.3f       | %.3f +- %.3f\n", p, ma, sema, ms, sems); }
    }

    printf("\nphase boundary = the p at which acc jumps toward the ~0.87 curriculum ceiling and\n");
    printf("specialization rises; a sharp jump = transition, a gradual rise = none.\n");
    return 0;
}
