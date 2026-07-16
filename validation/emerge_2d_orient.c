/* emerge_2d_orient.c -- STABILITY PROBE: can the 2-D engine stably discriminate orientation?
 *
 * emerge_2d.c found the diagonal-distance task is not a stable 2-D foundation (it is 1-D-on-the-diagonal
 * and hits the receptive-field-extreme corner problem). Before building any 2-D reuse/recombination
 * study we need a signal that is genuinely 2-D AND stable. The natural one is conv's core competency and
 * the canonical 2-D primitive (Hubel-Wiesel, Gabor, LeCun): ORIENTATION. This probe asks only whether
 * the engine can reliably tell a horizontal bar from a vertical one -- a single oriented 3x3 filter
 * should do it at L=1. If this is stably high across seeds, the engine is sound for 2-D oriented tasks
 * and we have a foundation; if not, 2-D is simply hard in this setup and we stop and say so.
 *
 * Task: both classes contain one 3-cell bar (so "a bar exists" is uninformative -- only ORIENTATION
 * separates them). Positive = horizontal bar; negative = vertical bar. Global max-pool readout, so the
 * net must fire an orientation-selective unit somewhere. Reports accuracy vs depth over many seeds with
 * spread, to judge STABILITY. Self-contained C99. Build: make emerge_2d_orient
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define G     12
#define KK    3
#define LMAX  3
#define NTR   192
#define NTE   1200
#define LR    0.05

static uint32_t rs=1u; static uint32_t r32(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;rs=x;return x;}
static void rseed(uint32_t s){rs=s?s:1u;} static double runif(void){return (double)r32()/4294967296.0*2.0-1.0;}
static uint32_t urand(uint32_t n){ return r32()%n; }
static uint32_t wr=1u; static uint32_t wr32(void){uint32_t x=wr;x^=x<<13;x^=x>>17;x^=x<<5;wr=x;return x;}
static void wseed(uint32_t s){wr=s?s:1u;} static double wunif(void){return (double)wr32()/4294967296.0*2.0-1.0;}

static int envint(const char*k,int d){const char*e=getenv(k);return e&&*e?atoi(e):d;}
static double envdbl(const char*k,double d){const char*e=getenv(k);return e&&*e?atof(e):d;}
static int    TEPOCHS = 200;
static double AMP     = 3.0;
static int    RESTARTS = 3;
static double g_target = 0.85;

static double Xtr[NTR][G][G], Xte[NTE][G][G];
static int    ytr[NTR], yte[NTE];

/* both classes plant ONE 3-cell bar; class 1 = horizontal, class 0 = vertical. Only orientation differs. */
static void gen(double X[][G][G],int*y,int n)
{
    int s,i,j,r,c;
    for(s=0;s<n;s++){
        for(i=0;i<G;i++) for(j=0;j<G;j++) X[s][i][j]=runif();
        y[s]=(int)(r32()&1u);
        r=(int)urand(G-2); c=(int)urand(G-2);       /* room for a 3-bar either way */
        if(y[s]){ X[s][r][c]+=AMP; X[s][r][c+1]+=AMP; X[s][r][c+2]+=AMP; }     /* horizontal */
        else    { X[s][r][c]+=AMP; X[s][r+1][c]+=AMP; X[s][r+2][c]+=AMP; }     /* vertical */
    }
}
static void new_task(uint32_t seed){ rseed(seed); gen(Xtr,ytr,NTR); gen(Xte,yte,NTE); }

static int side_at(int l){ return G - l*(KK-1); }

static double train_eval(int L, uint32_t seed)
{
    static double w[LMAX][KK][KK], bconv[LMAX], rscale, rbias;
    static double a[LMAX+1][G][G];
    int e,s,i,j,l,r,c, sL=side_at(L), cc=0;
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
    int seeds=envint("SEEDS",8), sd, L;
    g_target=envdbl("TARGET",0.85);
    TEPOCHS=envint("TEPOCHS",TEPOCHS); AMP=envdbl("AMP",AMP); RESTARTS=envint("RESTARTS",RESTARTS);
    printf("2-D STABILITY PROBE: orientation discrimination (horizontal vs vertical 3-bar)\n");
    printf("G=%dx%d k=%d, %d seeds x %d restarts, AMP=%.1f, %d epochs. both classes have one bar.\n\n",
           G, G, KK, seeds, RESTARTS, AMP, TEPOCHS);
    printf("  depth L | mean test acc | spread(SD) | min over seeds\n");
    for(L=1;L<=LMAX;L++){ double sum=0,sm2=0,mn=1.0; int nv=0;
        for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*911u+1u));
            double v=best_eval(L,(uint32_t)(sd*7u+1u)); sum+=v; sm2+=v*v; if(v<mn)mn=v; nv++; }
        { double m=sum/nv, sdv= nv>1? sqrt((sm2 - sum*sum/nv)/(nv-1)) : 0;
          printf("   L=%d    | %.3f         | %.03f      | %.3f\n", L, m, sdv, mn); }
    }
    printf("\nSTABLE foundation if L=1 is high (single oriented filter suffices) with small spread and a\n");
    printf("high per-seed minimum -- then a genuinely-2-D reuse/recombination study can be built on it.\n");
    return 0;
}
