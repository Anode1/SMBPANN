/* emerge_arch.c -- does the REUSE heuristic tame the combinatorics of composition?
 *
 * emerge_compose.c imposed one reused block type and searched only depth: emergent depth tracked the
 * task's required depth. That assumed the user's heuristic -- "reuse the same block, many, wired
 * input->output" -- rather than testing it. This probe tests it: give the search a LIBRARY of
 * different block types and ask whether letting the composition be heterogeneous actually helps, and
 * at what search cost.
 *
 * Block library: a dilated conv (K taps, dilation d in {1,2,3}). Dilation is the standard way to buy
 * receptive field per layer: block d adds (K-1)*d to the RF but samples with gaps of d, so a large
 * dilation reaches far cheaply yet may MISS the exact spacing it must detect. Energy = block count.
 *
 * Task (same fair depth test as emerge_compose): fire iff two spikes are exactly distance s apart,
 * max-pool readout so position/count cannot be used. Detecting distance s needs a unit whose sampled
 * taps land on BOTH spikes -> a real interaction of receptive-field reach (favoring big dilation) and
 * tap alignment (favoring dilation 1). So neither "all dilation 1" nor "all dilation 3" is obviously
 * best, and a heterogeneous mix MIGHT win -- if the search can find it.
 *
 * Two searches at EQUAL COMPUTE (candidate-training budgets matched):
 *   REUSE: one block type reused -- enumerate (dilation d, depth L), extra restarts to spend the
 *          budget. Tiny space (3*Lmax).
 *   FREE : a GA over arbitrary dilation sequences and length. Combinatorial (up to 3^Lmax).
 * We report, per s, solve-rate and the energy (block count) among solved. The honest question: at
 * equal compute, does searching the big heterogeneous space solve MORE or CHEAPER than enumerating
 * uniform reused blocks -- or is reuse the tractable near-optimum? Self-contained C99. make emerge_arch
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define N     24
#define K     3
#define LMAX  6
#define NTR   192
#define NTE   1200
#define TEPOCHS 250
#define LR    0.05
#define AMP   2.5

static uint32_t rs=1u; static uint32_t r32(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;rs=x;return x;}
static void rseed(uint32_t s){rs=s?s:1u;} static double runif(void){return (double)r32()/4294967296.0*2.0-1.0;}
static uint32_t urand(uint32_t n){ return r32()%n; }
static uint32_t wr=1u; static uint32_t wr32(void){uint32_t x=wr;x^=x<<13;x^=x>>17;x^=x<<5;wr=x;return x;}
static void wseed(uint32_t s){wr=s?s:1u;} static double wunif(void){return (double)wr32()/4294967296.0*2.0-1.0;}

static int envint(const char*k,int d){const char*e=getenv(k);return e&&*e?atoi(e):d;}
static double envdbl(const char*k,double d){const char*e=getenv(k);return e&&*e?atof(e):d;}
static double g_target = 0.85;
static int    g_sep = 4;
static long   g_evals = 0;               /* candidate trainings spent (search cost) */

static double Xtr[NTR][N], Xte[NTE][N];
static int    ytr[NTR], yte[NTE];

static void gen(double X[][N],int*y,int n)
{
    int s,i,dd,p;
    for(s=0;s<n;s++){
        for(i=0;i<N;i++) X[s][i]=runif();
        y[s]=(int)(r32()&1u);
        if(y[s]) dd=g_sep;
        else { do { dd=1+(int)urand(N-1); } while(abs(dd-g_sep)<2); }
        p=(int)urand((uint32_t)(N-dd));
        X[s][p]+=AMP; X[s][p+dd]+=AMP;
    }
}
static void new_task(uint32_t seed,int sep){ rseed(seed); g_sep=sep; gen(Xtr,ytr,NTR); gen(Xte,yte,NTE); }

/* length after a dilated block of dilation d over length m */
static int len_after(int m,int d){ return m - (K-1)*d; }
/* lengths of every layer given a dilation sequence; returns final length or -1 if any layer < 1 */
static int arch_lens(const int*dil,int L,int*mlen)
{ int l; mlen[0]=N; for(l=0;l<L;l++){ mlen[l+1]=len_after(mlen[l],dil[l]); if(mlen[l+1]<1) return -1; } return mlen[L]; }

/* train a dilated-conv stack (own weights per block) + max-pool readout; return test acc, or -1 if
 * the architecture is invalid (a layer shrinks below length 1). */
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

/* fitness: reaching target is worth (100 - energy); below target, just the accuracy */
static double fitness(double acc,int L){ return acc>=g_target ? (100.0 - L) : acc; }

typedef struct { int dil[LMAX], L; double acc; } Arch;

/* REUSE search: one reused block type, enumerate (dilation, depth), nr restarts per config. */
static Arch search_reuse(uint32_t seed,int nr)
{
    Arch best; int d,L,i; double bf=-1e9; best.L=1; best.acc=0; for(i=0;i<LMAX;i++) best.dil[i]=1;
    for(d=1;d<=3;d++) for(L=1;L<=LMAX;L++){ int dil[LMAX]; for(i=0;i<L;i++) dil[i]=d;
        { double acc=best_eval(dil,L,seed,nr); if(acc<0) continue;
          if(fitness(acc,L)>bf){ bf=fitness(acc,L); best.L=L; best.acc=acc; for(i=0;i<L;i++) best.dil[i]=d; } } }
    return best;
}
/* FREE search: GA over arbitrary dilation sequences and length, nr restarts per candidate. */
static Arch search_free(uint32_t seed,int pop,int gens,int nr)
{
    Arch P[64]; double fit[64]; int p,g,i,q; rseed(seed^0x9e3779b9u);
    if(pop>64) pop=64;
    for(p=0;p<pop;p++){ P[p].L=1+(int)urand(LMAX); for(i=0;i<P[p].L;i++) P[p].dil[i]=1+(int)urand(3);
        P[p].acc=best_eval(P[p].dil,P[p].L,seed+(uint32_t)p*2654435761u+1u,nr); fit[p]=fitness(P[p].acc,P[p].L); }
    for(g=0;g<gens;g++){
        int idx[64]; for(p=0;p<pop;p++) idx[p]=p;
        for(p=0;p<pop;p++) for(q=p+1;q<pop;q++) if(fit[idx[q]]>fit[idx[p]]){int t=idx[p];idx[p]=idx[q];idx[q]=t;}
        for(p=pop/2;p<pop;p++){ Arch a=P[idx[(int)urand((uint32_t)(pop/2))]];   /* mutate a top-half parent */
            int m=(int)urand(3);
            if(m==0 && a.L<LMAX){ for(i=a.L;i>0;i--) a.dil[i]=a.dil[i-1]; a.dil[0]=1+(int)urand(3); a.L++; }   /* prepend block */
            else if(m==1 && a.L>1){ a.L--; }                                    /* drop last block */
            else { a.dil[(int)urand((uint32_t)a.L)]=1+(int)urand(3); }          /* change a dilation */
            a.acc=best_eval(a.dil,a.L,seed+(uint32_t)(g*pop+p)+11u,nr); P[idx[p]]=a; fit[idx[p]]=fitness(a.acc,a.L); }
    }
    { int bp=0; for(p=1;p<pop;p++) if(fit[p]>fit[bp]) bp=p; return P[bp]; }
}
static void fmt(const Arch*a,char*buf){ int i,n=0; n+=sprintf(buf+n,"["); for(i=0;i<a->L;i++) n+=sprintf(buf+n,"%s%d",i?" ":"",a->dil[i]); sprintf(buf+n,"]"); }

/* REUSE gets more restarts per config so its total evals ~ FREE's -- an EQUAL-COMPUTE comparison.
 * FREE: (pop + gens*pop/2) candidates * REUSE-restarts... we just set REUSE restarts to balance. */
#define POP 12
#define GENS 10
#define FREE_NR 2
#define REUSE_NR 8    /* 18 configs * 8 ~= 144 ~= FREE (12 + 10*6)*2 = 144 evals */

int main(void)
{
    int seeds=envint("SEEDS",8), sd, di;
    int seps[3]={4,8,12};
    g_target=envdbl("TARGET",0.85);
    printf("REUSE vs FREE composition at EQUAL COMPUTE: does heterogeneity beat the reuse heuristic?\n");
    printf("N=%d K=%d, dilations {1,2,3}, up to L=%d, %d seeds, target %.2f\n", N, K, LMAX, seeds, g_target);
    printf("REUSE enumerates uniform (dilation,depth) with %d restarts; FREE runs a GA over arbitrary\n", REUSE_NR);
    printf("dilation sequences -- budgets matched (~equal candidate trainings). Rank: solve-rate, then\n");
    printf("energy (block count) among solved. Winner needs to actually SOLVE, then be cheaper.\n\n");
    printf("  s  | REUSE solve  energy  evals | FREE solve  energy  evals | verdict\n");
    for(di=0;di<3;di++){
        double rLs=0,fLs=0; long rE=0,fE=0; int rn=0,fn=0; char rb[128],fb[128];
        Arch lastR, lastF; lastR.L=0; lastF.L=0;
        for(sd=1;sd<=seeds;sd++){
            new_task((uint32_t)(sd*911u+(unsigned)seps[di]*17u+1u),seps[di]);
            g_evals=0; Arch R=search_reuse((uint32_t)(sd*7u+1u),REUSE_NR); rE+=g_evals;
            if(R.acc>=g_target){ rn++; rLs+=R.L; lastR=R; }
            g_evals=0; Arch F=search_free((uint32_t)(sd*7u+1u),POP,GENS,FREE_NR); fE+=g_evals;
            if(F.acc>=g_target){ fn++; fLs+=F.L; lastF=F; }
        }
        { double rEn = rn? rLs/rn : 0, fEn = fn? fLs/fn : 0;
          const char *verdict;
          if(fn>rn+0) verdict = "FREE solves more";
          else if(rn>fn+0) verdict = "REUSE solves more";
          else if(fEn < rEn-0.25) verdict = "tie-solve, FREE cheaper";
          else if(rEn < fEn-0.25) verdict = "tie-solve, REUSE cheaper";
          else verdict = "tie (same solve & energy)";
          printf("  %-2d | %d/%d  ", seps[di], rn, seeds);
          if(rn) printf("L=%.1f", rEn); else printf("  -  "); printf("  %4ld | %d/%d  ", rE/seeds, fn, seeds);
          if(fn) printf("L=%.1f", fEn); else printf("  -  "); printf("  %4ld | %s\n", fE/seeds, verdict); }
        if(lastR.L){ fmt(&lastR,rb); } else strcpy(rb,"[-]");
        if(lastF.L){ fmt(&lastF,fb); } else strcpy(fb,"[-]");
        printf("       example solved arch: reuse %s   free %s\n", rb, fb);
    }
    printf("\nreuse heuristic validated if, at equal compute, FREE does not solve more or cheaper than\n");
    printf("enumerating uniform reused blocks -- i.e. block diversity buys nothing but a bigger search.\n");
    return 0;
}
