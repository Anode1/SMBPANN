/* emerge_relax.c -- paper 2 control: DESCEND the energy budget, or SELECT on it?
 * Self-contained C99, no dependencies. Build: make emerge_relax
 *
 * THE QUESTION. Every probe in this line asks a population to discover a compact filter by trial. But
 * non-organic emergence needs no search: crystals, convection cells and soap films produce structure
 * by relaxing an energy functional under local dynamics. If structure appears there with no search at
 * all, the search may be doing less work here than the framing assumes.
 *
 *   does the compact contiguous K-tap filter appear when the energy budget is DESCENDED rather than
 *   SELECTED ON, with the model and the scoring held fixed?
 *
 * yes -> the bottleneck in this line was the SEARCH.  no -> the OBJECTIVE decides, as suspected.
 *
 * WHY THIS PROBE IS BUILT THE WAY IT IS. The obvious version of this experiment is confounded, and the
 * first draft of this file was: it compared an L1-regularised relaxation against a GA whose energy is a
 * COUNT (L0), on a different architecture. Two changes at once. If the two land in different places
 * that could be L1 != L0, or windows != offsets, and nothing to do with search versus descent.
 *
 * So: **never compare optimizers by their own objective values. Compare the STRUCTURES they produce,
 * under one common scoring function.**
 *
 *   - ONE model for everything: weights tied by OFFSET (as emerge_prove / emerge_fsdd), so a structure
 *     is exactly a binary support over NOFF = N+H-1 offsets.
 *   - FOUR producers of supports: L1 relaxation (binarised), a GA over supports, the hand-built K-tap
 *     ideal, and a tap-matched RANDOM support (protocol item 5: the rate-matched null is a primary
 *     comparison, not an afterthought).
 *   - ONE scorer: score_support() retrains the masked kernel from scratch and reports held-out
 *     accuracy. Every producer is judged by it, including the relaxation, whose own L1 objective is
 *     demoted to what it actually is: a proposal mechanism for supports.
 *
 * L1 under plain SGD only shrinks weights toward zero, so "surviving taps" would report wherever a
 * threshold was placed. This uses a PROXIMAL step (soft-threshold / ISTA): after each update
 * w <- sign(w)*max(|w| - lr*lambda, 0), which lands taps on EXACT zero.
 *
 * PASS / FAIL, fixed before running:
 *   relaxation FOUND THE TARGET  iff at some lambda its support is exactly K taps, contiguous, and its
 *                                accuracy is within noise of the hand-built ideal.
 *   relaxation BEAT THE SEARCH   iff its support scores higher than the GA's under the common
 *                                objective (accuracy - LAMBDA*taps/NOFF).
 *   anything else is a null, and the null is informative: it puts descent and selection on the same
 *   footing for the first time in this line.
 *
 * env: SEEDS EPOCHS LR LAMBDA GENS POP PFLIP
 */
#include "common.h"

#define N     12                /* inputs                                  */
#define K     3                 /* planted filter width -- the target      */
#define H     (N - K + 1)       /* offsets the kernel is applied at        */
#define NOFF  (N + H - 1)       /* 21 shared weights, indexed by i-j+OFF0  */
#define OFF0  (H - 1)
#define NTR   64
#define NTE   1000
#define POPMAX 64
#define NLAM  9
#define ZERO  1e-9

static int    g_seeds=20, g_epochs=200, g_gens=60, g_pop=24, g_elite=4;
static double g_lr=0.05, g_lambda=0.5, g_pflip=0.10;

/* ============================== TASK ============================== */
static double wstar[K];
static double Xtr[NTR][N], Xte[NTE][N];
static int    ytr[NTR], yte[NTE];

static int label_of(const double *x)
{ int p,k; double s=0; for(p=0;p+K<=N;p++){ double a=0; for(k=0;k<K;k++) a+=wstar[k]*x[p+k]; s+=tanh(a);} return s>0; }
static void gen(double X[][N], int *y, int n)
{ int s,i; for(s=0;s<n;s++){ for(i=0;i<N;i++) X[s][i]=runif(); y[s]=label_of(X[s]); } }
static void new_task(uint32_t seed)
{ int k; rseed(seed); for(k=0;k<K;k++) wstar[k]=runif(); gen(Xtr,ytr,NTR); gen(Xte,yte,NTE); }

/* ===================== THE ONE SCORER =====================
 * Every producer's structure is judged here and nowhere else: retrain the offset kernel restricted to
 * `mask`, from scratch, and return held-out accuracy. Same architecture, trainer and budget for all
 * four producers, so the only thing differing between rows is the SUPPORT itself. */
static double score_support(const char mask[NOFF], uint32_t seed)
{
    double w[NOFF], b[H], v[H], bo=0, h[H];
    int i,j,e,s,c=0;
    wseed(seed);
    for(i=0;i<NOFF;i++) w[i] = mask[i] ? 0.3*wunif() : 0.0;
    for(j=0;j<H;j++){ b[j]=0; v[j]=0.1*wunif(); }
    for(e=0;e<g_epochs;e++) for(s=0;s<NTR;s++){
        const double *x=Xtr[s]; double opre=bo,o,dout,dw[NOFF];
        for(i=0;i<NOFF;i++) dw[i]=0;
        for(j=0;j<H;j++){ double pre=b[j];
            for(i=0;i<N;i++) if(mask[i-j+OFF0]) pre += w[i-j+OFF0]*x[i];
            h[j]=tanh(pre); opre += v[j]*h[j]; }
        o=1.0/(1.0+exp(-opre)); dout=(o-ytr[s])*o*(1.0-o);
        for(j=0;j<H;j++){ double dpre=dout*v[j]*(1.0-h[j]*h[j]);
            v[j]-=g_lr*dout*h[j]; b[j]-=g_lr*dpre;
            for(i=0;i<N;i++) if(mask[i-j+OFF0]) dw[i-j+OFF0]+=dpre*x[i]; }
        bo-=g_lr*dout;
        for(i=0;i<NOFF;i++) if(mask[i]) w[i]-=g_lr*dw[i];
    }
    for(s=0;s<NTE;s++){
        const double *x=Xte[s]; double opre=bo,o;
        for(j=0;j<H;j++){ double pre=b[j];
            for(i=0;i<N;i++) if(mask[i-j+OFF0]) pre += w[i-j+OFF0]*x[i];
            opre += v[j]*tanh(pre); }
        o=1.0/(1.0+exp(-opre)); if((o>0.5)==(yte[s]==1)) c++;
    }
    return (double)c/NTE;
}
static int taps_of(const char m[NOFF]){ int i,d=0; for(i=0;i<NOFF;i++) d+=(m[i]!=0); return d; }
static int contig_of(const char m[NOFF])
{ int i,lo=NOFF,hi=-1,d=0;
  for(i=0;i<NOFF;i++) if(m[i]){ d++; if(i<lo) lo=i; if(i>hi) hi=i; }
  return d>0 && d==hi-lo+1; }
/* the common objective: accuracy minus the energy budget, energy = fraction of offsets switched on */
static double objective(const char m[NOFF], double acc)
{ return acc - g_lambda*(double)taps_of(m)/NOFF; }

/* ============ PRODUCER 1: RELAXATION (descend the energy) ============ */
static void produce_relax(double lam, uint32_t seed, char mask_out[NOFF])
{
    double w[NOFF], b[H], v[H], bo=0, h[H];
    int i,j,e,s;
    wseed(seed);
    for(i=0;i<NOFF;i++) w[i]=0.3*wunif();
    for(j=0;j<H;j++){ b[j]=0; v[j]=0.1*wunif(); }
    for(e=0;e<g_epochs;e++) for(s=0;s<NTR;s++){
        const double *x=Xtr[s]; double opre=bo,o,dout,dw[NOFF];
        for(i=0;i<NOFF;i++) dw[i]=0;
        for(j=0;j<H;j++){ double pre=b[j];
            for(i=0;i<N;i++) pre += w[i-j+OFF0]*x[i];
            h[j]=tanh(pre); opre += v[j]*h[j]; }
        o=1.0/(1.0+exp(-opre)); dout=(o-ytr[s])*o*(1.0-o);
        for(j=0;j<H;j++){ double dpre=dout*v[j]*(1.0-h[j]*h[j]);
            v[j]-=g_lr*dout*h[j]; b[j]-=g_lr*dpre;
            for(i=0;i<N;i++) dw[i-j+OFF0]+=dpre*x[i]; }
        bo-=g_lr*dout;
        for(i=0;i<NOFF;i++) w[i]-=g_lr*dw[i];
        for(i=0;i<NOFF;i++){                     /* proximal step: the energy term, applied exactly */
            double t=g_lr*lam;
            if(w[i] >  t) w[i]-=t;
            else if(w[i] < -t) w[i]+=t;
            else w[i]=0.0;
        }
    }
    for(i=0;i<NOFF;i++) mask_out[i] = (char)(fabs(w[i])>ZERO);
}

/* ============ PRODUCER 2: THE GA (select on the energy) ============ */
/* Same model, and the common scorer as its fitness. Mutation flips offsets; no crossover, which paper
 * 1 and this direction both measured null on this class of genome. */
static void produce_ga(uint32_t seed, char best_out[NOFF])
{
    static char pop[POPMAX][NOFF], nxt[POPMAX][NOFF];
    double fit[POPMAX]; int idx[POPMAX], g,p,q,i,best=0; double bf=-1e300;
    rseed(seed);
    for(p=0;p<g_pop;p++) for(i=0;i<NOFF;i++) pop[p][i]=1;     /* dense start, as relaxation has */
    for(p=0;p<g_pop;p++) fit[p]=objective(pop[p], score_support(pop[p],(uint32_t)(seed+p*2654435761u+1u)));
    for(g=0; g<g_gens; g++){
        for(p=0;p<g_pop;p++) idx[p]=p;
        for(p=0;p<g_pop;p++) for(q=p+1;q<g_pop;q++)
            if(fit[idx[q]]>fit[idx[p]]){ int t=idx[p]; idx[p]=idx[q]; idx[q]=t; }
        for(p=0;p<g_elite;p++) memcpy(nxt[p],pop[idx[p]],NOFF);
        for(p=g_elite;p<g_pop;p++){
            int a=idx[(int)(r32()%(uint32_t)g_elite)];
            memcpy(nxt[p],pop[a],NOFF);
            for(i=0;i<NOFF;i++) if(rprob()<g_pflip) nxt[p][i]^=1;
        }
        memcpy(pop,nxt,sizeof pop);
        for(p=0;p<g_pop;p++)
            fit[p]=objective(pop[p], score_support(pop[p],(uint32_t)(seed+(uint32_t)(g*g_pop+p)+7u)));
    }
    for(p=0;p<g_pop;p++) if(fit[p]>bf){ bf=fit[p]; best=p; }
    memcpy(best_out,pop[best],NOFF);
}

/* ============ PRODUCERS 3 and 4: the ideal, and the tap-matched null ============ */
static void produce_ideal(char m[NOFF])
{ int i; for(i=0;i<NOFF;i++) m[i]=0;
  for(i=OFF0; i<OFF0+K && i<NOFF; i++) m[i]=1; }
static void produce_random(int taps, char m[NOFF])
{ int i,n=0; for(i=0;i<NOFF;i++) m[i]=0;
  while(n<taps && n<NOFF){ int r=(int)(r32()%(uint32_t)NOFF); if(!m[r]){ m[r]=1; n++; } } }

int main(void)
{
    static const double LAM[NLAM] = {0.005,0.010,0.015,0.020,0.024,0.028,0.032,0.036,0.040};
    int li, sd;
    double acc_i=0,tap_i=0,obj_i=0, acc_g=0,tap_g=0,con_g=0,obj_g=0;
    double acc_r[NLAM]={0}, tap_r[NLAM]={0}, con_r[NLAM]={0}, obj_r[NLAM]={0};
    double acc_n[NLAM]={0}, obj_n[NLAM]={0};

    g_seeds=envint("SEEDS",20); g_epochs=envint("EPOCHS",200); g_gens=envint("GENS",60);
    g_pop=envint("POP",24); g_lr=envdbl("LR",0.05); g_lambda=envdbl("LAMBDA",0.5);
    g_pflip=envdbl("PFLIP",0.10);
    if(g_pop>POPMAX) g_pop=POPMAX;

    printf("emerge_relax -- descend vs select, ONE model and ONE scorer\n");
    printf("offset-tied kernel, %d offsets, planted K=%d. %d seeds x %d epochs, lr=%.3f.\n",
           NOFF, K, g_seeds, g_epochs, g_lr);
    printf("common objective = accuracy - %.2f * taps/%d.  GA: pop %d, %d gens, pflip %.2f.\n\n",
           g_lambda, NOFF, g_pop, g_gens, g_pflip);

    for(sd=1; sd<=g_seeds; sd++){
        char mi[NOFF], mg[NOFF], mr[NOFF], mn[NOFF]; double a;
        new_task((uint32_t)(sd*911u+1u));

        produce_ideal(mi);
        a=score_support(mi,(uint32_t)(sd*7u+1u));
        acc_i+=a; tap_i+=taps_of(mi); obj_i+=objective(mi,a);

        produce_ga((uint32_t)(sd*7u+1u), mg);
        a=score_support(mg,(uint32_t)(sd*7u+1u));
        acc_g+=a; tap_g+=taps_of(mg); con_g+=contig_of(mg); obj_g+=objective(mg,a);

        for(li=0; li<NLAM; li++){
            produce_relax(LAM[li],(uint32_t)(sd*7u+1u), mr);
            a=score_support(mr,(uint32_t)(sd*7u+1u));
            acc_r[li]+=a; tap_r[li]+=taps_of(mr); con_r[li]+=contig_of(mr); obj_r[li]+=objective(mr,a);
            produce_random(taps_of(mr), mn);      /* protocol item 5: tap-matched null */
            a=score_support(mn,(uint32_t)(sd*7u+1u));
            acc_n[li]+=a; obj_n[li]+=objective(mn,a);
        }
    }

    printf("  %-22s %6s %8s %10s %11s\n", "producer", "taps", "contig", "test acc", "objective");
    printf("  %-22s %6.2f %8.2f %10.3f %11.4f\n", "ideal (hand-built K)",
           tap_i/g_seeds, 1.0, acc_i/g_seeds, obj_i/g_seeds);
    printf("  %-22s %6.2f %8.2f %10.3f %11.4f\n", "GA (select on energy)",
           tap_g/g_seeds, con_g/g_seeds, acc_g/g_seeds, obj_g/g_seeds);
    for(li=0; li<NLAM; li++){
        char nm[40];
        sprintf(nm,"relax lambda=%.3f", LAM[li]);
        printf("  %-22s %6.2f %8.2f %10.3f %11.4f  | tap-matched rnd %.3f / %.4f\n", nm,
               tap_r[li]/g_seeds, con_r[li]/g_seeds, acc_r[li]/g_seeds, obj_r[li]/g_seeds,
               acc_n[li]/g_seeds, obj_n[li]/g_seeds);
    }
    printf("\nPASS/FAIL, fixed before running:\n");
    printf("  FOUND THE TARGET if some lambda gives taps=%d, contig=1, accuracy within noise of ideal.\n", K);
    printf("  BEAT THE SEARCH  if a relax objective exceeds the GA's.\n");
    printf("  If the tap-matched random column matches a relax row, that row's STRUCTURE is doing no\n");
    printf("  work and only its SIZE is -- the same null that refuted emerge_tile.c.\n");
    return 0;
}
