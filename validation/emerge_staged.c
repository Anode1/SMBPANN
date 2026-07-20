/* emerge_staged.c -- an adaptive searcher that CLONES while it pays, RECOMBINES only when it stalls.
 *
 * Section 9 (emerge_arch) showed cloning one block is the tractable near-optimum for repetitive tasks;
 * Section 10 (emerge_twoop) showed it BREAKS when a task needs two different operations, where
 * recombination is required. The two regimes were demonstrated separately. This builds the searcher
 * that handles both without being told which regime it is in -- the algorithmic "runners when scaling
 * a module, seeds when you need a new part."
 *
 * STAGED strategy: first CLONE -- enumerate reused (uniform-dilation) stacks, cheapest energy first,
 * and stop the instant one reaches target. If the whole clone sweep finishes WITHOUT solving, that is
 * the stall signal (cloning is exhausted, no single reused block works) -- switch to RECOMBINE, a GA
 * over heterogeneous dilation sequences. Note we do NOT use an "accuracy plateau" stall detector: these
 * tasks are flat-then-jump (chance until the receptive field covers the pattern), so a plateau detector
 * would misfire in the flat region. "Clone sweep finished unsolved" is the robust signal.
 *
 * We run three strategies -- pure REUSE (clone only), pure FREE (recombine only), and STAGED -- on TWO
 * task regimes at the same distance s: ONE-OP (repetitive: two spikes at distance s; uniform stacks
 * solve it) and TWO-OP (fine (+,-,+) motif AND far spike; needs two different block types). Expectation:
 *   ONE-OP : REUSE cheap+solves, FREE solves but wasteful, STAGED = cheap (solves in the clone phase).
 *   TWO-OP : REUSE fails, FREE solves but always pays, STAGED = solves (detects stall, recombines).
 *
 * Finding (24 seeds, s=8): STAGED matches or beats the best fixed strategy in each regime, and BEATS both
 * on the repetitive one: ONE-OP 22/24 (REUSE 15/24, FREE 18/24), TWO-OP 24/24 (ties FREE 24/24; REUSE
 * 11/24). Clone and recombine have partially non-overlapping success sets, so clone-then-recombine
 * catches seeds neither gets alone (ONE-OP 22 = 15 clone + 7 recombine, above FREE's 18) -- the fallback
 * backstops noise in the easy regime too, not just the hard one. Cost: adaptivity is not free --
 * STAGED pays the failed clone sweep before recombining (ONE-OP cost 114 vs REUSE 63; TWO-OP cost 154 vs
 * FREE 140). But it never has either fixed strategy's worst-case failure. Reports solve rate, mean
 * candidate-trainings, and which phase STAGED solved in.
 * Self-contained C99. Build: make emerge_staged
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define N     24
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
static int    g_sep = 8;
static int    g_task = 0;                 /* 0 = ONE-OP (repetitive), 1 = TWO-OP */
static long   g_evals = 0;

static double Xtr[NTR][N], Xte[NTE][N];
static int    ytr[NTR], yte[NTE];

/* ONE-OP: two spikes at distance g_sep (positive) vs any other distance (negative). Uniform-solvable. */
static void gen_oneop(double X[][N],int*y,int n)
{
    int s,i,dd,p;
    for(s=0;s<n;s++){
        for(i=0;i<N;i++) X[s][i]=runif();
        y[s]=(int)(r32()&1u);
        if(y[s]) dd=g_sep; else { do { dd=1+(int)urand(N-1); } while(abs(dd-g_sep)<2); }
        p=(int)urand((uint32_t)(N-dd));
        X[s][p]+=AMP; X[s][p+dd]+=AMP;
    }
}
/* TWO-OP: fine (+,-,+) motif AND far spike at distance g_sep; negatives break exactly one condition. */
static void plant(double*x,int q,int fine,int far)
{ x[q]+=AMP; x[q+1]+= fine? -AMP : AMP; x[q+2]+=AMP; if(far>=0 && q+far<N) x[q+far]+=AMP; }
static void gen_twoop(double X[][N],int*y,int n)
{
    int s,i,q,far,cls;
    for(s=0;s<n;s++){
        for(i=0;i<N;i++) X[s][i]=runif();
        cls=(int)urand(2); q=(int)urand((uint32_t)(N-g_sep-1));
        if(cls==0){ plant(X[s],q,1,g_sep); y[s]=1; }
        else { y[s]=0;
            if(urand(2)==0){ do { far=3+(int)urand((uint32_t)(N-1-q-1)); } while(abs(far-g_sep)<2 || q+far>=N); plant(X[s],q,1,far); }
            else { plant(X[s],q,0,g_sep); } }
    }
}
static void new_task(uint32_t seed,int sep)
{ rseed(seed); g_sep=sep; if(g_task==0){ gen_oneop(Xtr,ytr,NTR); gen_oneop(Xte,yte,NTE); }
  else { gen_twoop(Xtr,ytr,NTR); gen_twoop(Xte,yte,NTE); } }

static int arch_lens(const int*dil,int L,int*mlen)
{ int l; mlen[0]=N; for(l=0;l<L;l++){ mlen[l+1]=mlen[l]-(K-1)*dil[l]; if(mlen[l+1]<1) return -1; } return mlen[L]; }

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
typedef struct { Arch a; int solved; int phase; } Result;   /* phase: 0 clone, 1 recombine */

/* CLONE sweep: enumerate uniform stacks, LOWEST ENERGY (depth) first, exit the instant one solves. */
static Result clone_sweep(uint32_t seed,int nr)
{
    Arch best; int d,L,i; double bf=-1e9; best.L=1; best.acc=0; for(i=0;i<LMAX;i++) best.dil[i]=1;
    for(L=1;L<=LMAX;L++) for(d=1;d<=3;d++){ int dil[LMAX]; for(i=0;i<L;i++) dil[i]=d;
        { double acc=best_eval(dil,L,seed,nr); if(acc<0) continue;
          if(fitness(acc,L)>bf){ bf=fitness(acc,L); best.L=L; best.acc=acc; for(i=0;i<L;i++) best.dil[i]=d; }
          if(acc>=g_target){ Result r; r.a=best; r.solved=1; r.phase=0; return r; } } }   /* early exit */
    { Result r; r.a=best; r.solved=(best.acc>=g_target); r.phase=0; return r; }
}
/* RECOMBINE: GA over arbitrary dilation sequences. */
static Arch recombine(uint32_t seed,int pop,int gens,int nr)
{
    Arch P[64]; double fit[64]; int p,g,i,q; rseed(seed^0x9e3779b9u);
    if(pop>64) pop=64;
    for(p=0;p<pop;p++){ P[p].L=1+(int)urand(LMAX); for(i=0;i<P[p].L;i++) P[p].dil[i]=1+(int)urand(3);
        P[p].acc=best_eval(P[p].dil,P[p].L,seed+(uint32_t)p*2654435761u+1u,nr); fit[p]=fitness(P[p].acc,P[p].L); }
    for(g=0;g<gens;g++){
        int idx[64]; for(p=0;p<pop;p++) idx[p]=p;
        for(p=0;p<pop;p++) for(q=p+1;q<pop;q++) if(fit[idx[q]]>fit[idx[p]]){int t=idx[p];idx[p]=idx[q];idx[q]=t;}
        for(p=pop/2;p<pop;p++){ Arch a=P[idx[(int)urand((uint32_t)(pop/2))]];
            int m=(int)urand(3);
            if(m==0 && a.L<LMAX){ for(i=a.L;i>0;i--) a.dil[i]=a.dil[i-1]; a.dil[0]=1+(int)urand(3); a.L++; }
            else if(m==1 && a.L>1){ a.L--; }
            else { a.dil[(int)urand((uint32_t)a.L)]=1+(int)urand(3); }
            a.acc=best_eval(a.dil,a.L,seed+(uint32_t)(g*pop+p)+11u,nr); P[idx[p]]=a; fit[idx[p]]=fitness(a.acc,a.L); }
    }
    { int bp=0; for(p=1;p<pop;p++) if(fit[p]>fit[bp]) bp=p; return P[bp]; }
}
/* STAGED: clone; if the sweep stalls (no uniform solved), recombine. */
static Result staged(uint32_t seed,int nr,int pop,int gens,int gnr)
{
    Result c=clone_sweep(seed,nr);
    if(c.solved) return c;                                    /* clone regime, cheap */
    { Arch f=recombine(seed,pop,gens,gnr); Result r; r.a=f; r.solved=(f.acc>=g_target); r.phase=1; return r; }
}

#define CLONE_NR 6
#define POP 12
#define GENS 10
#define GA_NR 2

int main(void)
{
    int seeds=envint("SEEDS",24), sd, t;
    const char *tn[2]={"ONE-OP (repetitive)","TWO-OP (two different ops)"};
    g_target=envdbl("TARGET",0.85); g_sep=envint("SEP",8);
    printf("STAGED searcher: clone while it pays, recombine when it stalls. s=%d, %d seeds, target %.2f\n", g_sep, seeds, g_target);
    printf("N=%d K=%d dilations {1,2,3} up to L=%d. cost = candidate trainings.\n\n", N, K, LMAX);
    printf("  task regime               | strategy | solve | mean-cost | note\n");
    for(t=0;t<2;t++){ g_task=t;
        int rSolve=0,fSolve=0,sSolve=0, sClone=0,sRecomb=0; long rC=0,fC=0,sC=0;
        for(sd=1;sd<=seeds;sd++){
            new_task((uint32_t)(sd*911u+(unsigned)g_sep*17u+1u),g_sep);
            g_evals=0; Result R=clone_sweep((uint32_t)(sd*7u+1u),CLONE_NR); rC+=g_evals; if(R.solved) rSolve++;
            g_evals=0; Arch F=recombine((uint32_t)(sd*7u+1u),POP,GENS,GA_NR); fC+=g_evals; if(F.acc>=g_target) fSolve++;
            g_evals=0; Result S=staged((uint32_t)(sd*7u+1u),CLONE_NR,POP,GENS,GA_NR); sC+=g_evals;
            if(S.solved){ sSolve++; if(S.phase==0) sClone++; else sRecomb++; }
        }
        printf("  %-25s | REUSE    | %d/%d  | %6ld    | clone only\n", tn[t], rSolve, seeds, rC/seeds);
        printf("  %-25s | FREE     | %d/%d  | %6ld    | recombine only\n", "", fSolve, seeds, fC/seeds);
        printf("  %-25s | STAGED   | %d/%d  | %6ld    | solved: %d via clone, %d via recombine\n",
               "", sSolve, seeds, sC/seeds, sClone, sRecomb);
        if(t==0) printf("  %-25s |\n","");
    }
    printf("\nSTAGED wins if it matches REUSE's low cost on ONE-OP (solves in clone) and FREE's solve\n");
    printf("rate on TWO-OP (detects the stall, recombines) -- adapting to the regime without being told.\n");
    return 0;
}
