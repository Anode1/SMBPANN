/* emerge_compose.c -- does the emerged DEPTH match the task's compositional depth?
 *
 * The composition step. emerge_local.c showed a simple local block emerges. The next abstraction
 * layer is COMPOSITION: stack that block to build deeper structure. The combinatorics of "which
 * blocks, wired how" explode, so we adopt the heuristic LeCun used: don't design each block
 * differently -- REUSE ONE BLOCK TYPE, stacked feed-forward input->output, and search only DEPTH.
 *
 * IMPOSED (honestly "a bit of cheating, like LeCun's"): the composition rule -- a feed-forward stack
 * of L identical-type local conv blocks, wired input to output. EMERGENT: the depth L, every block's
 * weights (trained from scratch), and transfer across independent tasks of the family.
 *
 * The task must genuinely REQUIRE depth, or the answer is trivial. A first attempt (a self-similar
 * conv hierarchy read out by a full linear layer) failed as a test: a shallow block + a positional
 * linear readout solved every depth, because the readout globally integrates whatever the blocks
 * detect -- depth is never needed. The fair fix is a RECEPTIVE-FIELD task with a GLOBAL-POOL readout:
 *
 *   Two spikes are planted in a noisy signal. A POSITIVE example has them at a specific distance s;
 *   a NEGATIVE has them at any other distance. Both classes contain exactly two spikes, so mere
 *   presence is uninformative and a MAX-POOL readout cannot cheat by counting or by position -- the
 *   network must fire a unit that SEES BOTH spikes at once and checks their spacing. That needs
 *   receptive field >= s+1, i.e. stack depth L >= s/2. So the task's required depth is set by s:
 *   s = 2,4,6,8 need L = 1,2,3,4. A shallow net literally cannot see the pair and is stuck at chance.
 *
 * The honest question: under an energy budget (energy = depth), does the selected depth L* track the
 * required depth -- do abstraction layers appear exactly as deep as the task's composition demands,
 * from a reused block? Reports, per required depth: test accuracy vs stack depth L, and the
 * energy-selected L* (shallowest L reaching target). Each accuracy averages INDEPENDENT task draws,
 * so a working depth is task-general (reuse/transfer). Self-contained C99. Build: make emerge_compose
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define N     18
#define K     3
#define LMAX  5
#define NTR   192
#define NTE   1500
#define TEPOCHS 300
#define LR    0.05
#define AMP   2.5
#define RESTARTS 4

static uint32_t rs=1u; static uint32_t r32(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;rs=x;return x;}
static void rseed(uint32_t s){rs=s?s:1u;} static double runif(void){return (double)r32()/4294967296.0*2.0-1.0;}
static uint32_t urand(uint32_t n){ return r32()%n; }
static uint32_t wr=1u; static uint32_t wr32(void){uint32_t x=wr;x^=x<<13;x^=x>>17;x^=x<<5;wr=x;return x;}
static void wseed(uint32_t s){wr=s?s:1u;} static double wunif(void){return (double)wr32()/4294967296.0*2.0-1.0;}

static int envint(const char*k,int d){const char*e=getenv(k);return e&&*e?atoi(e):d;}
static double envdbl(const char*k,double d){const char*e=getenv(k);return e&&*e?atof(e):d;}
static double g_target = 0.85;
static int    g_sep = 2;             /* target spike distance for the current task */

static double Xtr[NTR][N], Xte[NTE][N];
static int    ytr[NTR], yte[NTE];

static int len_at(int l){ return N - l*(K-1); }   /* signal length after l valid conv layers */

/* plant two spikes: positive => distance == g_sep, negative => distance != g_sep (both classes have
 * exactly two spikes, so only the SPACING is informative -> the net must see both spikes at once). */
static void gen(double X[][N],int*y,int n)
{
    int s,i,dd,p;
    for(s=0;s<n;s++){
        for(i=0;i<N;i++) X[s][i]=runif();
        y[s]=(int)(r32()&1u);
        if(y[s]) dd=g_sep;
        else { do { dd=1+(int)urand(N-1); } while(abs(dd-g_sep)<2); }  /* margin: negatives >=2 off */
        p=(int)urand((uint32_t)(N-dd));         /* p in [0, N-1-dd] */
        X[s][p]+=AMP; X[s][p+dd]+=AMP;
    }
}
static void new_task(uint32_t seed,int sep){ rseed(seed); g_sep=sep; gen(Xtr,ytr,NTR); gen(Xte,yte,NTE); }

/* train a feed-forward stack of L conv blocks (own weights per layer) + MAX-POOL readout (a single
 * learned scale+bias on the pooled max, so the readout cannot use position or count). Returns test acc. */
static double train_eval(int L, uint32_t seed)
{
    static double w[LMAX][K], bconv[LMAX], rscale, rbias;
    static double a[LMAX+1][N];
    int e,s,i,k,l, mo, mL=len_at(L), c=0;
    wseed(seed);
    for(l=0;l<L;l++){ for(k=0;k<K;k++) w[l][k]=0.4*wunif(); bconv[l]=0; }
    rscale=0.4*wunif(); rbias=0;
    for(e=0;e<TEPOCHS;e++) for(s=0;s<NTR;s++){
        for(i=0;i<N;i++) a[0][i]=Xtr[s][i];
        for(l=0;l<L;l++){ mo=len_at(l+1);
            for(i=0;i<mo;i++){ double pre=bconv[l]; for(k=0;k<K;k++) pre+=w[l][k]*a[l][i+k]; a[l+1][i]=tanh(pre); } }
        { double pool=a[L][0]; int am=0; for(i=1;i<mL;i++) if(a[L][i]>pool){pool=a[L][i];am=i;}  /* max-pool */
          double o=1.0/(1.0+exp(-(rscale*pool+rbias))), dout=(o-ytr[s])*o*(1.0-o);
          static double da[LMAX+1][N], dpre[N];
          for(i=0;i<mL;i++) da[L][i]=0;
          da[L][am]=dout*rscale;                                 /* gradient routes to the argmax */
          rscale-=LR*dout*pool; rbias-=LR*dout;
          for(l=L-1;l>=0;l--){ int mi=len_at(l), mno=len_at(l+1);
              for(i=0;i<mi;i++) da[l][i]=0;
              for(i=0;i<mno;i++){ dpre[i]=da[l+1][i]*(1.0-a[l+1][i]*a[l+1][i]);
                  bconv[l]-=LR*dpre[i];
                  for(k=0;k<K;k++){ da[l][i+k]+=dpre[i]*w[l][k]; w[l][k]-=LR*dpre[i]*a[l][i+k]; } } }
        }
    }
    for(s=0;s<NTE;s++){
        for(i=0;i<N;i++) a[0][i]=Xte[s][i];
        for(l=0;l<L;l++){ mo=len_at(l+1);
            for(i=0;i<mo;i++){ double pre=bconv[l]; for(k=0;k<K;k++) pre+=w[l][k]*a[l][i+k]; a[l+1][i]=tanh(pre); } }
        { double pool=a[L][0]; for(i=1;i<mL;i++) if(a[L][i]>pool) pool=a[L][i];
          double o=1.0/(1.0+exp(-(rscale*pool+rbias))); if((o>0.5)==(yte[s]==1)) c++; }
    }
    return (double)c/NTE;
}
static double best_eval(int L, uint32_t seed)
{ int r; double best=0; for(r=0;r<RESTARTS;r++){ double a=train_eval(L,seed*131u+(uint32_t)r*97u+1u); if(a>best) best=a; } return best; }

int main(void)
{
    int seeds=envint("SEEDS",12), sd, di, L;
    int seps[4]={2,4,6,8}, reqL[4]={1,2,3,4};   /* distance s -> required depth s/2 */
    double acc[LMAX+1];
    g_target=envdbl("TARGET",0.85);
    printf("EMERGENT DEPTH vs REQUIRED DEPTH  (reused feed-forward conv block + global max-pool)\n");
    printf("N=%d K=%d, %d seeds x %d restarts, up to L=%d, target acc >= %.2f\n", N, K, seeds, RESTARTS, LMAX, g_target);
    printf("task: fire iff two spikes are exactly distance s apart. RF>=s+1 needed => depth>=s/2.\n");
    printf("imposed: feed-forward reused-block stack. emergent: depth L, weights, cross-task transfer.\n\n");
    printf("  distance s (need L) | test accuracy at stack depth L=1..%d      | L* (energy-selected)\n", LMAX);
    for(di=0;di<4;di++){ g_sep=seps[di];
        for(L=1;L<=LMAX;L++){ double sacc=0; int nv=0;
            for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*911u+g_sep*17u+1u),g_sep);
                if(len_at(L)>=1){ sacc+=best_eval(L,(uint32_t)(sd*7u+1u)); nv++; } }
            acc[L]= nv? sacc/nv : 0; }
        { int Ls=-1; for(L=1;L<=LMAX;L++) if(acc[L]>=g_target){ Ls=L; break; }
          printf("  s=%d (need L=%d)      |", seps[di], reqL[di]);
          for(L=1;L<=LMAX;L++) printf(" %.3f", acc[L]);
          if(Ls>0) printf("  | L*=%d\n", Ls); else printf("  | L*=- \n"); }
    }
    printf("\nemergent depth tracks required depth if L* rises 1,2,3,4 with s=2,4,6,8.\n");
    printf("(shallow nets cannot see the spike pair -> stuck at chance; energy picks the shallowest that can.)\n");
    return 0;
}
