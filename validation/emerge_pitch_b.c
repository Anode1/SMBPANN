/* emerge_pitch_b.c -- variant B of emerge_pitch.c: a larger-window, sparser-period sweep. Forked from
 * emerge_pitch.c with ONLY the window size, dilation alphabet, and period sweep widened; everything
 * else (task logic, engine, reuse-vs-recombine analysis) is unchanged.
 *
 * emerge_arch.c found reuse (one block type, stacked) is the tractable near-optimum -- but only for a
 * task whose difficulty is a single receptive-field requirement, which uniform stacks already meet.
 * The prediction was that reuse should cost something on a task that genuinely needs two
 * DIFFERENT operations composed, where no single reused block can serve both roles. This builds it.
 *
 * The library is again dilated conv (dilation 1/2/3/4), but the two dilations are now genuinely
 * different feature extractors, and the pattern to detect (anywhere, max-pool) needs BOTH at once:
 *   FINE op (needs adjacent taps -> dilation 1): a local motif (+A,-A,+A) at three ADJACENT positions.
 *     A block with dilation >= 2 samples with gaps and skips the middle tap -- it literally cannot
 *     tell (+,-,+) from (+,+,+). So the fine motif is invisible to coarse blocks.
 *   PERIODICITY op (needs to span the train's period -> cheaper with larger dilation): a periodic
 *     spike train at period p (spikes at q+p, q+2p, ... -- the pitch analog). Reaching successive
 *     spikes with dilation 1 costs ~p/2 blocks per hop; larger dilations cover the spacing in fewer.
 * A positive has the correct fine motif AND the periodic train at period p. Negatives break exactly one:
 *   (A) correct fine motif but the train at the WRONG period;  (B) correct train at period p
 *   but the fine motif CORRUPTED to (+,+,+). So both ops are necessary -- period alone or fine alone
 *   never separates the classes.
 *
 * B widens N to 40 (so several periods fit) and adds dilation 4 to the library, sweeping sparser
 * periods {4,6,8}. Single-block tap spacings are {d,2d} for d in 1..4 = {1,2,3,4,6,8}, so swept
 * periods 4 (=2*2 outer taps of d=2, or adjacent taps of d=4), 6 (=2*3), 8 (=2*4) are each
 * representable by one block.
 *
 * Predicted crossover (by analogy to emerge_twoop): uniform dilation-4 FAILS (blind to the fine motif);
 * uniform dilation-1 pays depth to span the period; a HETEROGENEOUS mix (e.g. [1,4]: one fine block, then a
 * coarse period-spanning block) does both at lower energy. So this is the regime where searching
 * different blocks (recombination) is predicted to beat cloning one (reuse). Actual per-period solve
 * counts and energies are measured by the sweep below.
 *
 * reuse BREAKS here if uniform-d3 fails (blind to the fine motif) and FREE finds a cheaper MIX than
 * uniform-d1 -- i.e. two different ops force heterogeneity, and cloning one block cannot express it.
 * Self-contained C99 (reuses emerge_arch's dilated-conv engine). Build: make emerge_pitch_b
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define N     40
#define K     3
#define LMAX  6
#define NTR   224
#define NTE   1500
#define TEPOCHS 300
#define LR    0.05
#define AMP   2.0

static uint32_t rs=1u; static uint32_t r32(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;rs=x;return x;}
static void rseed(uint32_t s){rs=s?s:1u;} static double runif(void){return (double)r32()/4294967296.0*2.0-1.0;}
static uint32_t urand(uint32_t n){ return r32()%n; }
static uint32_t wr=1u; static uint32_t wr32(void){uint32_t x=wr;x^=x<<13;x^=x>>17;x^=x<<5;wr=x;return x;}
static void wseed(uint32_t s){wr=s?s:1u;} static double wunif(void){return (double)wr32()/4294967296.0*2.0-1.0;}

static int envint(const char*k,int d){const char*e=getenv(k);return e&&*e?atoi(e):d;}
static double envdbl(const char*k,double d){const char*e=getenv(k);return e&&*e?atof(e):d;}
static double g_target = 0.85;
static int    g_period = 4;               /* target period of the periodic spike train */
static long   g_evals = 0;

static double Xtr[NTR][N], Xte[NTE][N];
static int    ytr[NTR], yte[NTE];

/* plant a fine motif at [q,q+1,q+2] (fine=1: (+,-,+); fine=0: (+,+,+)) and a periodic spike train
 * at q+per, q+2*per, ... (per>0). */
static void plant(double*x,int q,int fine,int per)
{
    x[q]+=AMP; x[q+1]+= fine? -AMP : AMP; x[q+2]+=AMP;
    if(per>0){ int t; for(t=q+per; t<N; t+=per) x[t]+=AMP; }
}
/* positive: fine (+,-,+) AND periodic train at g_period. neg A: fine but WRONG period. neg B:
 * corrupted fine (+,+,+) but correct period. Both ops needed to separate all three. */
static void gen(double X[][N],int*y,int n)
{
    int s,i,q,per,cls;
    for(s=0;s<n;s++){
        for(i=0;i<N;i++) X[s][i]=runif();
        cls=(int)urand(2);                       /* 0 = positive, 1 = negative */
        q=(int)urand(N/2);                        /* room for q..q+2 and several periods */
        if(cls==0){ plant(X[s],q,1,g_period); y[s]=1; }
        else {
            y[s]=0;
            if(urand(2)==0){                      /* neg A: fine ok, train at a NON-ALIASING wrong period */
                /* exclude g_period and its divisors/multiples: those trains' spikes are a subset/superset
                 * of the period-g_period train, so a coarse detector cannot separate them (false null). */
                do { per=2+(int)urand(9); }
                while(per==g_period || g_period%per==0 || per%g_period==0);   /* per in [2,10], non-aliasing */
                plant(X[s],q,1,per);
            } else {                              /* neg B: train at correct period, fine corrupted */
                plant(X[s],q,0,g_period);
            }
        }
    }
}
static void new_task(uint32_t seed,int period){ rseed(seed); g_period=period; gen(Xtr,ytr,NTR); gen(Xte,yte,NTE); }

static int arch_lens(const int*dil,int L,int*mlen)
{ int l; mlen[0]=N; for(l=0;l<L;l++){ mlen[l+1]=mlen[l]-(K-1)*dil[l]; if(mlen[l+1]<1) return -1; } return mlen[L]; }

/* dilated-conv stack (own weights per block) + max-pool readout; test acc, or -1 if invalid. */
static double train_eval(const int*dil, int L, uint32_t seed)
{
    static double w[LMAX][K], bconv[LMAX], rscale, rbias;
    static double a[LMAX+1][N];
    int mlen[LMAX+1];
    int e,s,i,k,l, mL, c=0;
    if((mL=arch_lens(dil,L,mlen))<0) return -1.0;
    g_evals++;
    wseed(seed);
    for(l=0;l<L;l++){ for(k=0;k<K;k++) w[l][k]=0.4*wunif(); bconv[l]=0; }
    rscale=0.4*wunif(); rbias=0;
    for(e=0;e<TEPOCHS;e++) for(s=0;s<NTR;s++){
        for(i=0;i<N;i++) a[0][i]=Xtr[s][i];
        for(l=0;l<L;l++){ int d=dil[l], mo=mlen[l+1];
            for(i=0;i<mo;i++){ double pre=bconv[l]; for(k=0;k<K;k++) pre+=w[l][k]*a[l][i+k*d]; a[l+1][i]=tanh(pre); } }
        { double pool=a[L][0]; int am=0; for(i=1;i<mL;i++) if(a[L][i]>pool){pool=a[L][i];am=i;}
          double o=1.0/(1.0+exp(-(rscale*pool+rbias))), dout=(o-ytr[s])*o*(1.0-o);
          static double da[LMAX+1][N], dpre[N];
          for(i=0;i<mL;i++) da[L][i]=0;
          da[L][am]=dout*rscale;
          rscale-=LR*dout*pool; rbias-=LR*dout;
          for(l=L-1;l>=0;l--){ int d=dil[l], mi=mlen[l], mno=mlen[l+1];
              for(i=0;i<mi;i++) da[l][i]=0;
              for(i=0;i<mno;i++){ dpre[i]=da[l+1][i]*(1.0-a[l+1][i]*a[l+1][i]);
                  bconv[l]-=LR*dpre[i];
                  for(k=0;k<K;k++){ da[l][i+k*d]+=dpre[i]*w[l][k]; w[l][k]-=LR*dpre[i]*a[l][i+k*d]; } } }
        }
    }
    for(s=0;s<NTE;s++){
        for(i=0;i<N;i++) a[0][i]=Xte[s][i];
        for(l=0;l<L;l++){ int d=dil[l], mo=mlen[l+1];
            for(i=0;i<mo;i++){ double pre=bconv[l]; for(k=0;k<K;k++) pre+=w[l][k]*a[l][i+k*d]; a[l+1][i]=tanh(pre); } }
        { double pool=a[L][0]; for(i=1;i<mL;i++) if(a[L][i]>pool) pool=a[L][i];
          double o=1.0/(1.0+exp(-(rscale*pool+rbias))); if((o>0.5)==(yte[s]==1)) c++; }
    }
    return (double)c/NTE;
}
static double best_eval(const int*dil,int L, uint32_t seed, int nr)
{ int r; double best=-1; for(r=0;r<nr;r++){ double a=train_eval(dil,L,seed*131u+(uint32_t)r*97u+1u); if(a>best) best=a; } return best; }
static double fitness(double acc,int L){ return acc>=g_target ? (100.0 - L) : acc; }

typedef struct { int dil[LMAX], L; double acc; } Arch;

/* best uniform stack of a FIXED dilation d (search depth) -- one "reuse one op" strategy. */
static Arch best_uniform(int d,uint32_t seed,int nr)
{ Arch b; int L,i; double bf=-1e9; b.L=1; b.acc=0; for(i=0;i<LMAX;i++) b.dil[i]=d;
  for(L=1;L<=LMAX;L++){ int dil[LMAX]; for(i=0;i<L;i++) dil[i]=d;
    { double acc=best_eval(dil,L,seed,nr);
      if(acc<0) continue;
      if(fitness(acc,L)>bf){ bf=fitness(acc,L); b.L=L; b.acc=acc; } } }
  return b; }
/* FREE: GA over arbitrary dilation sequences (recombination). */
static Arch search_free(uint32_t seed,int pop,int gens,int nr)
{
    Arch P[64]; double fit[64]; int p,g,i,q; rseed(seed^0x9e3779b9u);
    if(pop>64) pop=64;
    for(p=0;p<pop;p++){ P[p].L=1+(int)urand(LMAX); for(i=0;i<P[p].L;i++) P[p].dil[i]=1+(int)urand(4);
        P[p].acc=best_eval(P[p].dil,P[p].L,seed+(uint32_t)p*2654435761u+1u,nr); fit[p]=fitness(P[p].acc,P[p].L); }
    for(g=0;g<gens;g++){
        int idx[64]; for(p=0;p<pop;p++) idx[p]=p;
        for(p=0;p<pop;p++) for(q=p+1;q<pop;q++) if(fit[idx[q]]>fit[idx[p]]){int t=idx[p];idx[p]=idx[q];idx[q]=t;}
        for(p=pop/2;p<pop;p++){ Arch a=P[idx[(int)urand((uint32_t)(pop/2))]];
            int m=(int)urand(3);
            if(m==0 && a.L<LMAX){ for(i=a.L;i>0;i--) a.dil[i]=a.dil[i-1]; a.dil[0]=1+(int)urand(4); a.L++; }
            else if(m==1 && a.L>1){ a.L--; }
            else { a.dil[(int)urand((uint32_t)a.L)]=1+(int)urand(4); }
            a.acc=best_eval(a.dil,a.L,seed+(uint32_t)(g*pop+p)+11u,nr); P[idx[p]]=a; fit[idx[p]]=fitness(a.acc,a.L); }
    }
    { int bp=0; for(p=1;p<pop;p++) if(fit[p]>fit[bp]) bp=p; return P[bp]; }
}
static void fmt(const Arch*a,char*buf){ int i,n=0; n+=sprintf(buf+n,"["); for(i=0;i<a->L;i++) n+=sprintf(buf+n,"%s%d",i?" ":"",a->dil[i]); sprintf(buf+n,"]"); }

#define POP 12
#define GENS 10
#define FREE_NR 2
#define REUSE_NR 8

int main(void)
{
    int seeds=envint("SEEDS",24), sd, di;
    int periods[8], nper=0;                       /* default {4,6,8}; override e.g. PERIODS=8 to run one period per process */
    { const char*e=getenv("PERIODS"); char buf[64]; strncpy(buf, e&&*e?e:"4,6,8", sizeof buf-1); buf[sizeof buf-1]=0;
      { char*p=strtok(buf,","); while(p&&nper<8){ int v=atoi(p); if(v>=2) periods[nper++]=v; p=strtok(NULL,","); } }
      if(nper==0){ periods[0]=4; periods[1]=6; periods[2]=8; nper=3; } }
    g_target=envdbl("TARGET",0.85);
    printf("TWO-OPERATION task: fine motif (+,-,+, needs dilation 1) AND periodic spike train at period p\n");
    printf("(cheaper with larger dilation). No single reused block serves both roles. Equal compute.\n");
    printf("N=%d K=%d, dilations {1,2,3,4}, up to L=%d, %d seeds, target %.2f\n\n", N, K, LMAX, seeds, g_target);
    printf("  p  | uni-d1 (solve,E) | uni-d4 (solve,E) | REUSE best | FREE/recomb best   | verdict\n");
    for(di=0;di<nper;di++){
        int u1n=0,u4n=0,rn=0,fn=0; double u1L=0,u4L=0,rL=0,fL=0; char fb[128];
        Arch lastF; lastF.L=0;
        for(sd=1;sd<=seeds;sd++){
            new_task((uint32_t)(sd*911u+(unsigned)periods[di]*17u+1u),periods[di]);
            /* the reuse heuristic gets to pick its one block type: best over uniform d=1,2,3,4 */
            Arch u1=best_uniform(1,(uint32_t)(sd*7u+1u),REUSE_NR);
            Arch u2=best_uniform(2,(uint32_t)(sd*7u+1u),REUSE_NR);
            Arch u3=best_uniform(3,(uint32_t)(sd*7u+1u),REUSE_NR);
            Arch u4=best_uniform(4,(uint32_t)(sd*7u+1u),REUSE_NR);
            Arch R = (fitness(u1.acc,u1.L)>=fitness(u2.acc,u2.L)) ? u1 : u2;
            if(fitness(u3.acc,u3.L)>fitness(R.acc,R.L)) R=u3;
            if(fitness(u4.acc,u4.L)>fitness(R.acc,R.L)) R=u4;
            if(u1.acc>=g_target){u1n++;u1L+=u1.L;}
            if(u4.acc>=g_target){u4n++;u4L+=u4.L;}
            if(R.acc>=g_target){rn++;rL+=R.L;}
            g_evals=0; Arch F=search_free((uint32_t)(sd*7u+1u),POP,GENS,FREE_NR); if(F.acc>=g_target){fn++;fL+=F.L; lastF=F;}
        }
        { double u1e=u1n?u1L/u1n:0,u4e=u4n?u4L/u4n:0,re=rn?rL/rn:0,fe=fn?fL/fn:0;
          const char*verdict;
          if(fn>rn) verdict="FREE solves more";
          else if(fn==rn && fe<re-0.25) verdict="FREE cheaper -> reuse costs here";
          else if(rn>fn) verdict="REUSE solves more";
          else verdict="tie";
          printf("  %-2d | %d/%d ", periods[di], u1n, seeds); if(u1n) printf("L=%.1f",u1e); else printf(" -  ");
          printf("   | %d/%d ", u4n, seeds); if(u4n) printf("L=%.1f",u4e); else printf(" -  ");
          printf("   | %d/%d ",rn,seeds); if(rn) printf("L=%.1f",re); else printf(" - ");
          printf("  | %d/%d ",fn,seeds); if(fn) printf("L=%.1f",fe); else printf(" - ");
          printf("  | %s\n", verdict);
          if(lastF.L){ fmt(&lastF,fb); printf("       example FREE arch: %s\n", fb); } }
    }
    printf("\nreuse BREAKS if uniform-d4 fails (blind to fine motif) and FREE finds a cheaper MIX than\n");
    printf("uniform-d1 -- i.e. two different ops force heterogeneity, and cloning one block costs energy.\n");
    return 0;
}
