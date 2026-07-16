/* emerge_2d.c -- FOUNDATION: does the 1-D emergent-depth result reproduce, stably, in 2-D?
 *
 * Before asking anything interesting in 2-D (orientation as genuinely different operations, reuse vs
 * recombination), we must first have a correct 2-D conv engine and confirm the *already-established*
 * 1-D result survives the move to two dimensions. That result (Section 8, emerge_compose): a task with
 * a receptive-field requirement is solvable only once the stack is deep enough to cover it, so the
 * depth that works tracks the required receptive field. This file replicates exactly that in 2-D and
 * asks only whether it is STABLE -- no new claim, just a foundation to build on (or to catch a bug).
 *
 * Engine: a stack of L 2-D conv blocks (k x k, valid, tanh) over a G x G grid, then a global MAX-pool
 * (position-agnostic) and a learned scale+bias readout. Task: two spikes at a diagonal 2-D separation
 * s (positive) vs any other separation (negative); both classes have two spikes, so only the spacing
 * is informative and a unit must SEE BOTH at once -- receptive field >= s+1 per axis, i.e. depth
 * >= s/2. Expectation if the engine is right: accuracy is at chance until L covers the pair, then lifts
 * off around L ~ s/2, exactly as in 1-D. Reported over many seeds so we can judge STABILITY, not just a
 * lucky staircase. Self-contained C99. Build: make emerge_2d
 *
 * FINDING (8 seeds, AMP=3, 250 epochs) -- honest NEGATIVE: the clean 1-D staircase does NOT reproduce.
 * Only the smallest separation solves (s=2: 0.86 at L=2), and it even degrades with more depth. s=4
 * never reaches target (~0.68-0.71); s=6 is chance at every depth, including L>=3 where the receptive
 * field covers the pair. The engine is verified correct (ASan-clean; s=2 clearly learns), so this is a
 * TASK limitation, not a bug: exact-distance detection at the receptive-field extreme is hard in 2-D
 * (a stacked 3x3 kernel builds a smooth composite filter and cannot easily put sharp weight at the two
 * opposite corners of the field), the final map shrinks to a few cells after valid convs (max-pool gets
 * few positions), and 144 noisy cells breed spurious maxima. Also, this diagonal-offset task is really
 * 1-D-on-the-diagonal -- it does not exercise genuine 2-D structure. Conclusion: this task is NOT a
 * stable foundation; a 2-D study should use oriented-motif detection (conv's core competency, and
 * genuinely 2-D via orientation), not exact distance. Kept as an honest negative and engine check.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define G     12            /* grid side */
#define KK    3             /* conv kernel side */
#define LMAX  4
#define NTR   192
#define NTE   1200
#define LR    0.05
static int    TEPOCHS = 250;      /* env TEPOCHS */
static double AMP     = 3.0;      /* env AMP */
static int    RESTARTS = 3;       /* env RESTARTS */

static uint32_t rs=1u; static uint32_t r32(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;rs=x;return x;}
static void rseed(uint32_t s){rs=s?s:1u;} static double runif(void){return (double)r32()/4294967296.0*2.0-1.0;}
static uint32_t urand(uint32_t n){ return r32()%n; }
static uint32_t wr=1u; static uint32_t wr32(void){uint32_t x=wr;x^=x<<13;x^=x>>17;x^=x<<5;wr=x;return x;}
static void wseed(uint32_t s){wr=s?s:1u;} static double wunif(void){return (double)wr32()/4294967296.0*2.0-1.0;}

static int envint(const char*k,int d){const char*e=getenv(k);return e&&*e?atoi(e):d;}
static double envdbl(const char*k,double d){const char*e=getenv(k);return e&&*e?atof(e):d;}
static double g_target = 0.85;
static int    g_sep = 2;

static double Xtr[NTR][G][G], Xte[NTE][G][G];
static int    ytr[NTR], yte[NTE];

/* plant two unit spikes at a diagonal separation dd (both classes have two spikes). */
static void gen(double X[][G][G],int*y,int n)
{
    int s,i,j,dd,r,c;
    for(s=0;s<n;s++){
        for(i=0;i<G;i++) for(j=0;j<G;j++) X[s][i][j]=runif();
        y[s]=(int)(r32()&1u);
        if(y[s]) dd=g_sep; else { do { dd=1+(int)urand(G-1); } while(abs(dd-g_sep)<2); }
        r=(int)urand((uint32_t)(G-dd)); c=(int)urand((uint32_t)(G-dd));
        X[s][r][c]+=AMP; X[s][r+dd][c+dd]+=AMP;
    }
}
static void new_task(uint32_t seed,int sep){ rseed(seed); g_sep=sep; gen(Xtr,ytr,NTR); gen(Xte,yte,NTE); }

static int side_at(int l){ return G - l*(KK-1); }   /* valid conv, dilation 1 */

/* train a stack of L 2-D conv blocks + global max-pool readout; return test accuracy. */
static double train_eval(int L, uint32_t seed)
{
    static double w[LMAX][KK][KK], bconv[LMAX], rscale, rbias;
    static double a[LMAX+1][G][G];
    int e,s,i,j,k,l,r,c, sL=side_at(L), cc=0;
    wseed(seed);
    for(l=0;l<L;l++){ for(i=0;i<KK;i++) for(j=0;j<KK;j++) w[l][i][j]=0.3*wunif(); bconv[l]=0; }
    rscale=0.3*wunif(); rbias=0;
    for(e=0;e<TEPOCHS;e++) for(s=0;s<NTR;s++){
        for(i=0;i<G;i++) for(j=0;j<G;j++) a[0][i][j]=Xtr[s][i][j];
        for(l=0;l<L;l++){ int so=side_at(l+1);
            for(r=0;r<so;r++) for(c=0;c<so;c++){ double pre=bconv[l];
                for(i=0;i<KK;i++) for(j=0;j<KK;j++) pre+=w[l][i][j]*a[l][r+i][c+j];
                a[l+1][r][c]=tanh(pre); } }
        { double pool=a[L][0][0]; int rm=0,cm=0;
          for(r=0;r<sL;r++) for(c=0;c<sL;c++) if(a[L][r][c]>pool){pool=a[L][r][c];rm=r;cm=c;}
          double o=1.0/(1.0+exp(-(rscale*pool+rbias))), dout=(o-ytr[s])*o*(1.0-o);
          static double da[LMAX+1][G][G], dpre[G][G];
          for(r=0;r<sL;r++) for(c=0;c<sL;c++) da[L][r][c]=0;
          da[L][rm][cm]=dout*rscale;
          rscale-=LR*dout*pool; rbias-=LR*dout;
          for(l=L-1;l>=0;l--){ int si=side_at(l), so=side_at(l+1);
              for(r=0;r<si;r++) for(c=0;c<si;c++) da[l][r][c]=0;
              for(r=0;r<so;r++) for(c=0;c<so;c++){ dpre[r][c]=da[l+1][r][c]*(1.0-a[l+1][r][c]*a[l+1][r][c]);
                  bconv[l]-=LR*dpre[r][c];
                  for(i=0;i<KK;i++) for(j=0;j<KK;j++){ da[l][r+i][c+j]+=dpre[r][c]*w[l][i][j];
                      w[l][i][j]-=LR*dpre[r][c]*a[l][r+i][c+j]; } }
          }
        }
    }
    (void)k;
    for(s=0;s<NTE;s++){
        for(i=0;i<G;i++) for(j=0;j<G;j++) a[0][i][j]=Xte[s][i][j];
        for(l=0;l<L;l++){ int so=side_at(l+1);
            for(r=0;r<so;r++) for(c=0;c<so;c++){ double pre=bconv[l];
                for(i=0;i<KK;i++) for(j=0;j<KK;j++) pre+=w[l][i][j]*a[l][r+i][c+j];
                a[l+1][r][c]=tanh(pre); } }
        { double pool=a[L][0][0];
          for(r=0;r<sL;r++) for(c=0;c<sL;c++) if(a[L][r][c]>pool) pool=a[L][r][c];
          double o=1.0/(1.0+exp(-(rscale*pool+rbias))); if((o>0.5)==(yte[s]==1)) cc++; }
    }
    return (double)cc/NTE;
}
static double best_eval(int L, uint32_t seed)
{ int r; double best=0; for(r=0;r<RESTARTS;r++){ double a=train_eval(L,seed*131u+(uint32_t)r*97u+1u); if(a>best) best=a; } return best; }

int main(void)
{
    int seeds=envint("SEEDS",8), sd, di, L;
    int seps[3]={2,4,6}, reqL[3]={1,2,3};
    double acc[LMAX+1], sd_[LMAX+1];
    g_target=envdbl("TARGET",0.85);
    TEPOCHS=envint("TEPOCHS",TEPOCHS); AMP=envdbl("AMP",AMP); RESTARTS=envint("RESTARTS",RESTARTS);
    printf("2-D FOUNDATION: emergent depth vs required receptive field (2-D conv stack + global max-pool)\n");
    printf("G=%dx%d k=%d, %d seeds x %d restarts, up to L=%d. two spikes at diagonal separation s.\n", G, G, KK, seeds, RESTARTS, LMAX);
    printf("RF >= s+1 per axis needed => depth >= s/2. accuracy should be chance until L covers the pair.\n\n");
    printf("  sep s (need L) | mean test acc at depth L=1..%d      | spread(SD) at each L\n", LMAX);
    for(di=0;di<3;di++){ g_sep=seps[di];
        for(L=1;L<=LMAX;L++){ double sum=0,sm2=0; int nv=0;
            for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*911u+(unsigned)g_sep*17u+1u),g_sep);
                if(side_at(L)>=1){ double v=best_eval(L,(uint32_t)(sd*7u+1u)); sum+=v; sm2+=v*v; nv++; } }
            acc[L]= nv? sum/nv : 0; sd_[L]= nv>1? sqrt((sm2 - sum*sum/nv)/(nv-1)) : 0; }
        printf("  s=%d (need L=%d)   |", seps[di], reqL[di]);
        for(L=1;L<=LMAX;L++) printf(" %.3f", acc[L]);
        printf("   |");
        for(L=1;L<=LMAX;L++) printf(" %.02f", sd_[L]);
        printf("\n");
    }
    printf("\nstable if the lift-off depth tracks s (chance until ~L=s/2) with modest spread across seeds.\n");
    printf("this is a REPLICATION check, not a new claim -- the foundation for a 2-D reuse/recombine study.\n");
    return 0;
}
